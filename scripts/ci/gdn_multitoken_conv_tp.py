"""gdn_multitoken_conv's shape-checking helpers at the width this process serves at.

validate_projected, restore_prefix and run_projected carry the pair's widths as literals (a projected row of 8240 / 8256 columns,
24 value heads, 5120 conv channels) and gdn_multitoken_conv.py is one of the 42 sources the TP2 evidence pins, so
these are twins that read the widths from tp_shapes instead. tp_addresses.install() rebinds the pinned names to them
at QWEN_FAST_TP=4 only; at the pair they are not installed and the pinned functions run as before. Written call for
call after the pinned bodies, so a difference between the two at the pair is a bug (test_gdn_tp_twins holds them
equal at the pair).
"""

from gdn_multitoken_conv import convolution_checkpoints
from gdn_multitoken_tp import execute
from gdn_prefix import independent_row
import tp_shapes
from tp_addresses import addresses, release_owned


def validate_projected(shape, states):
    found = tp_shapes.active()
    if (len(shape) != 3 or shape[0] != 1 or shape[1] not in (1, 2, 4, 8, 16, 32)
            or shape[2] not in (found.gdn_qkvzab, found.gdn_qkvzab_padded)):
        raise ValueError('Expected projected TP%d rows [1,T,%d/%d]'
                         % (found.tp, found.gdn_qkvzab, found.gdn_qkvzab_padded))
    if len(states) != 4 or any(tuple(state.shape) != (1, 1, found.gdn_qkv) for state in states):
        raise ValueError('Four compact B1 convolution states required')
    return shape[1]


def restore_prefix(operations, result, entry, destinations, accepted):
    found = tp_shapes.active()
    rows = result['states'].shape[0]
    packed_checkpoints = result.get('packed_checkpoints', False)
    prefix_zero_reuse = result.get('prefix_zero_reuse', False)
    if type(prefix_zero_reuse) is not bool or (prefix_zero_reuse and not packed_checkpoints):
        raise ValueError('Prefix zero reuse requires explicit bool and packed checkpoints')
    windows = result.get('packed_conv_states', []) if packed_checkpoints else []
    if rows not in (1, 2, 4, 8, 16, 32) or tuple(result['states'].shape) != (rows, found.gdn_nv, 128, 128):
        raise ValueError('Supported recurrent prefix geometry required')
    if type(accepted) is not int or not 0 <= accepted <= rows:
        raise ValueError('Accepted prefix must lie within the candidate block')
    if len(entry) != 5 or len(destinations) != 5 or len(result['conv_prefixes']) != rows:
        raise ValueError('Complete recurrent and four-convolution state required')
    if any(prefix is not None
           and (len(prefix) != 4 or any(tuple(value.shape) != (1, 1, found.gdn_qkv) for value in prefix))
           for prefix in result['conv_prefixes']):
        raise ValueError('Complete compact convolution prefixes required')
    if accepted and result['conv_prefixes'][accepted - 1] is None and not packed_checkpoints:
        raise ValueError('Requested convolution prefix was not materialized')
    expected = [(1, found.gdn_nv, 128, 128), *[(1, 1, found.gdn_qkv)] * 4]
    if any(tuple(value.shape) != shape for values in (entry, destinations)
           for value, shape in zip(values, expected, strict=True)):
        raise ValueError('Compact B1 restore shapes required')
    if packed_checkpoints:
        from gdn_conv_prefix_copy import copy_prefix, validate_prefix
        validate_prefix([tuple(value.shape) for value in windows],
                        [tuple(value.shape) for value in destinations[1:]], max(1, accepted))
        if windows[0].shape[1] != rows or result.get('mesh') is None:
            raise ValueError('Packed convolution and recurrent prefix geometry must match')
    destination_addresses = [addresses(operations, value) for value in destinations]
    protected = [*entry, result['states'], *windows,
                 *[value for prefix in result['conv_prefixes'] if prefix is not None for value in prefix]]
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
                (accepted, found.gdn_nv, 128, 128), memory_config=operations.DRAM_MEMORY_CONFIG)
            if packed_checkpoints:
                copy_prefix(result['mesh'], windows, destinations[1:], accepted,
                    **(dict(reuse_zero_tile=True) if prefix_zero_reuse else {}))
                sources = [sliced]
            else:
                sources = [sliced, *result['conv_prefixes'][accepted - 1]]
        else:
            sources = entry
        targets = destinations[:1] if accepted and packed_checkpoints else destinations
        for source, destination in zip(sources, targets, strict=True):
            operations.copy(source, destination)
        if [addresses(operations, value) for value in destinations] != destination_addresses:
            raise AssertionError('Restore changed stable state addresses')
    finally:
        if sliced is not None and not any(overlaps(addresses(operations, sliced), protected)
                                          for protected in protected_addresses):
            operations.deallocate(sliced)


def run_projected(mesh, projected, initial, conv_states, taps, dt_bias, neg_exp_A, norm_w, kernels, operations=None,
                  *, conv_checkpoints=None, hoist_input=False):
    """gdn_multitoken_conv.run_projected (one native row at a time, the sequential engine's T1 path and the
    single-user block) with the conv channel width, the a / b columns and the z slice from tp_shapes, launching
    through gdn_multitoken_tp.execute."""
    if operations is None:
        import ttnn as operations
    found = tp_shapes.active()
    rows = validate_projected(tuple(projected.shape), conv_states)
    selected = convolution_checkpoints(rows, conv_checkpoints)
    if hoist_input and rows == 1:
        raise ValueError('Hoisted input is a multirow-only experiment')
    if len(taps) != 4 or any(tuple(tap.shape) != (1, 1, found.gdn_qkv) for tap in taps):
        raise ValueError('Four channel-wise convolution taps required')
    owned, conv_prefixes = [], []
    streams = [[], [], []]
    initial_addresses = [addresses(operations, state) for state in conv_states]
    source_address = addresses(operations, projected)
    try:
        row_source = projected
        if hoist_input:
            row_source = operations.to_layout(projected, operations.ROW_MAJOR_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG)
            independent_row(operations, projected, row_source)
            owned.append(row_source)
            source_address = addresses(operations, row_source)
        for token in range(rows):
            view = operations.slice(row_source, (0, token, 0), (1, token + 1, projected.shape[-1]),
                                    memory_config=operations.DRAM_MEMORY_CONFIG)
            if addresses(operations, view) != source_address:
                owned.append(view)
            if hoist_input:
                independent_row(operations, row_source, view)
                row = operations.to_layout(view, operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG)
                independent_row(operations, view, row)
                independent_row(operations, projected, row)
            else:
                row = operations.clone(view, memory_config=operations.DRAM_MEMORY_CONFIG)
            owned.append(row)
            results = operations.transformer.gdn_decode_conv_gates(row, conv_states, taps, row, row,
                dt_bias, neg_exp_A, batch=1, memory_config=operations.L1_MEMORY_CONFIG,
                channels=found.gdn_qkv, a_col=found.gdn_a_col, b_col=found.gdn_b_col)
            owned.extend(results)
            for stream, result in zip(streams, results, strict=True):
                stream.append(result)
            prefix = [operations.clone(state, memory_config=operations.DRAM_MEMORY_CONFIG) for state in conv_states] if token + 1 in selected else None
            if prefix is not None:
                owned.extend(prefix)
            conv_prefixes.append(prefix)
        if [addresses(operations, state) for state in conv_states] != initial_addresses:
            raise AssertionError('Convolution state addresses changed')
        packed = [operations.concat(stream, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG) for stream in streams]
        owned.extend(packed)
        z = operations.slice(projected, (0, 0, found.gdn_qkv), (1, rows, found.gdn_a_col),
                             memory_config=operations.DRAM_MEMORY_CONFIG)
        owned.append(z)
        weights = operations.to_memory_config(norm_w, operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, weights) != addresses(operations, norm_w):
            owned.append(weights)
        output, states = execute(mesh, *packed, initial, kernels, z=z, norm_w=weights)
        owned.extend([output, states])
        return dict(output=output, states=states, conv_prefixes=conv_prefixes, owned=owned,
                    materialized_conv_prefixes=selected, hoisted_input=hoist_input)
    except BaseException:
        release_owned(operations, owned)
        raise
