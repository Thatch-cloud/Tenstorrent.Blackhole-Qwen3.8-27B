"""Native serial convolution/gates feeding one device-loop recurrence/norm call."""

from gdn_multitoken import execute


def validate_projected(shape, states):
    if len(shape) != 3 or shape[0] != 1 or shape[1] not in (1, 2, 4, 8, 16) or shape[2] not in (8240, 8256):
        raise ValueError('Expected projected TP2 rows [1,T,8240/8256]')
    if len(states) != 4 or any(tuple(state.shape) != (1, 1, 5120) for state in states):
        raise ValueError('Four compact B1 convolution states required')
    return shape[1]


def addresses(operations, tensor):
    shards = operations.get_device_tensors(tensor)
    if len(shards) != 2:
        raise ValueError('Both chips required')
    return tuple(shard.buffer_address() for shard in shards)


def release_owned(operations, tensors):
    unique = {addresses(operations, tensor): tensor for tensor in tensors}
    for tensor in unique.values():
        operations.deallocate(tensor)


def restore_prefix(operations, result, entry, destinations, accepted):
    rows = result['states'].shape[0]
    if rows not in (1, 2, 4, 8, 16) or tuple(result['states'].shape) != (rows, 24, 128, 128):
        raise ValueError('Supported recurrent prefix geometry required')
    if type(accepted) is not int or not 0 <= accepted <= rows:
        raise ValueError('Accepted prefix must lie within the candidate block')
    if len(entry) != 5 or len(destinations) != 5 or len(result['conv_prefixes']) != rows:
        raise ValueError('Complete recurrent and four-convolution state required')
    if any(len(prefix) != 4 or any(tuple(value.shape) != (1, 1, 5120) for value in prefix)
           for prefix in result['conv_prefixes']):
        raise ValueError('Complete compact convolution prefixes required')
    expected = [(1, 24, 128, 128), *[(1, 1, 5120)] * 4]
    if any(tuple(value.shape) != shape for values in (entry, destinations)
           for value, shape in zip(values, expected, strict=True)):
        raise ValueError('Compact B1 restore shapes required')
    destination_addresses = [addresses(operations, value) for value in destinations]
    protected = [*entry, result['states'], *[value for prefix in result['conv_prefixes'] for value in prefix]]
    protected_addresses = {addresses(operations, value) for value in protected}
    def overlaps(left, right):
        return any(first == second for first, second in zip(left, right, strict=True))

    if any(overlaps(address, protected) for address in destination_addresses for protected in protected_addresses):
        raise ValueError('Restore destinations must not alias immutable snapshots')
    if any(overlaps(address, other) for index, address in enumerate(destination_addresses)
           for other in destination_addresses[index + 1:]):
        raise ValueError('Restore destinations must be independent')
    sliced = None
    try:
        if accepted:
            sliced = operations.slice(result['states'], (accepted - 1, 0, 0, 0),
                (accepted, 24, 128, 128), memory_config=operations.DRAM_MEMORY_CONFIG)
            sources = [sliced, *result['conv_prefixes'][accepted - 1]]
        else:
            sources = entry
        for source, destination in zip(sources, destinations, strict=True):
            operations.copy(source, destination)
        if [addresses(operations, value) for value in destinations] != destination_addresses:
            raise AssertionError('Restore changed stable state addresses')
    finally:
        if sliced is not None and not any(overlaps(addresses(operations, sliced), protected) for protected in protected_addresses):
            operations.deallocate(sliced)


def finish_output(gdn, result, operations, reduce):
    rows = result['output'].shape[1]
    partial = gdn._row_proj(result['output'], gdn.tw['out'])
    partial = operations.reshape(partial, (1, 1, rows, partial.shape[-1]))
    output = reduce(partial, gdn.mesh, gdn.tt_ccl, cluster_axis=0, dim=3,
                    topology=gdn.args.ccl_topology(), memory_config=operations.DRAM_MEMORY_CONFIG)
    result['layer_output'] = output
    result['owned'].append(output)
    return result


def run_projected(mesh, projected, initial, conv_states, taps, dt_bias, neg_exp_A, norm_w, kernels, operations=None):
    if operations is None:
        import ttnn as operations
    rows = validate_projected(tuple(projected.shape), conv_states)
    if len(taps) != 4 or any(tuple(tap.shape) != (1, 1, 5120) for tap in taps):
        raise ValueError('Four channel-wise convolution taps required')
    owned, conv_prefixes = [], []
    streams = [[], [], []]
    initial_addresses = [addresses(operations, state) for state in conv_states]
    source_address = addresses(operations, projected)
    try:
        for token in range(rows):
            view = operations.slice(projected, (0, token, 0), (1, token + 1, projected.shape[-1]),
                                    memory_config=operations.DRAM_MEMORY_CONFIG)
            if addresses(operations, view) != source_address:
                owned.append(view)
            row = operations.clone(view, memory_config=operations.DRAM_MEMORY_CONFIG)
            owned.append(row)
            results = operations.transformer.gdn_decode_conv_gates(row, conv_states, taps, row, row,
                dt_bias, neg_exp_A, batch=1, memory_config=operations.L1_MEMORY_CONFIG,
                channels=5120, a_col=8192, b_col=8216)
            owned.extend(results)
            for stream, result in zip(streams, results, strict=True):
                stream.append(result)
            prefix = [operations.clone(state, memory_config=operations.DRAM_MEMORY_CONFIG) for state in conv_states]
            owned.extend(prefix)
            conv_prefixes.append(prefix)
        if [addresses(operations, state) for state in conv_states] != initial_addresses:
            raise AssertionError('Convolution state addresses changed')
        packed = [operations.concat(stream, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG) for stream in streams]
        owned.extend(packed)
        z = operations.slice(projected, (0, 0, 5120), (1, rows, 8192), memory_config=operations.DRAM_MEMORY_CONFIG)
        owned.append(z)
        weights = operations.to_memory_config(norm_w, operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, weights) != addresses(operations, norm_w):
            owned.append(weights)
        output, states = execute(mesh, *packed, initial, kernels, z=z, norm_w=weights)
        owned.extend([output, states])
        return dict(output=output, states=states, conv_prefixes=conv_prefixes, owned=owned)
    except BaseException:
        release_owned(operations, owned)
        raise
