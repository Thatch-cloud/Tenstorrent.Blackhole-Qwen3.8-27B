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
