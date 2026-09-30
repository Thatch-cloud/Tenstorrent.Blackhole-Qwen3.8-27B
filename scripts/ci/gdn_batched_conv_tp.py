"""gdn_batched_conv.run_batched_projected at any served width.

The pair's module is pinned (gdn_direct_window_hardware_sources.BATCH_SHA256) and stays as it was; tp_addresses
rebinds its run_batched_projected to this one at four cards. Call for call the pair's body with the projected row's
widths (qkv, the a and b columns) from tp_shapes.
"""

from gdn_batched_conv import history_windows, norm_batch_enabled
from gdn_multitoken import execute
from gdn_multitoken_conv import addresses, convolution_checkpoints, release_owned, run_projected, validate_projected
from gdn_prefix import independent_row
import tp_shapes


def run_batched_projected(mesh, projected, initial, conv_states, taps, dt_bias, neg_exp_A, norm_w, kernels,
                          operations=None, *, conv_checkpoints=None, hoist_input=False, dma_windows=False,
                          packed_checkpoints=False, norm_batch=False, norm_source_root=None, prefix_zero_reuse=False,
                          defer_conv_publication=False):
    if type(defer_conv_publication) is not bool or (defer_conv_publication and not (packed_checkpoints and dma_windows)):
        raise ValueError('Deferred convolution publication requires packed DMA checkpoints and explicit bool')
    if type(prefix_zero_reuse) is not bool or (prefix_zero_reuse and not packed_checkpoints):
        raise ValueError('Prefix zero reuse requires explicit bool and packed checkpoints')
    if operations is None:
        import ttnn as operations
    found = tp_shapes.active()
    rows = validate_projected(tuple(projected.shape), conv_states)
    use_norm_batch = norm_batch_enabled(rows, norm_batch)
    if use_norm_batch and tp_shapes.chip_count() != tp_shapes.PAIR:
        # gdn_vsplit's 96 recurrence / 24 norm-gate workers and 384 state pages are the pair's (S2T-05b, not ported)
        raise ValueError('The value-split norm batch is written for the (1, 2) pair')
    selected = convolution_checkpoints(rows, conv_checkpoints)
    if rows == 1:
        return run_projected(mesh, projected, initial, conv_states, taps, dt_bias, neg_exp_A, norm_w, kernels,
                             operations, conv_checkpoints=conv_checkpoints)
    if len(taps) != 4 or any(tuple(tap.shape) != (1, 1, found.gdn_qkv) for tap in taps):
        raise ValueError('Four channel-wise convolution taps required')
    owned = []
    original_addresses = [addresses(operations, state) for state in conv_states]

    def own(value):
        owned.append(value)
        return value

    def layout(value, target):
        converted = operations.to_layout(value, target, memory_config=operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, converted) != addresses(operations, value):
            own(converted)
        return converted

    def sliced(value, start, end):
        result = operations.slice(value, start, end, memory_config=operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, result) != addresses(operations, value):
            own(result)
        return result

    try:
        if dma_windows:
            from gdn_conv_windows import build_windows
            windows = build_windows(mesh, projected, conv_states)
            owned.extend(windows)
        else:
            source = layout(projected, operations.ROW_MAJOR_LAYOUT)
            projected_qkv = sliced(source, (0, 0, 0), (1, rows, found.gdn_qkv))
            history = own(operations.concat([*[layout(state, operations.ROW_MAJOR_LAYOUT) for state in conv_states],
                                             projected_qkv], dim=1, memory_config=operations.DRAM_MEMORY_CONFIG))
            windows = []
            for start, end in history_windows(rows):
                window = layout(sliced(history, (0, start, 0), (1, end, found.gdn_qkv)), operations.TILE_LAYOUT)
                independent_row(operations, history, window)
                windows.append(window)
        packed = operations.transformer.gdn_decode_conv_gates(projected, windows, taps, projected, projected,
            dt_bias, neg_exp_A, batch=rows, memory_config=operations.DRAM_MEMORY_CONFIG,
            channels=found.gdn_qkv, a_col=found.gdn_a_col, b_col=found.gdn_b_col)
        owned.extend(packed)
        prefixes = [None] * rows
        if not packed_checkpoints:
            shifted = [layout(window, operations.ROW_MAJOR_LAYOUT) for window in windows]
            for token in range(rows):
                if token + 1 in selected:
                    prefixes[token] = [layout(sliced(window, (0, token, 0), (1, token + 1, found.gdn_qkv)), operations.TILE_LAYOUT)
                                       for window in shifted]
        z = sliced(projected, (0, 0, found.gdn_qkv), (1, rows, found.gdn_a_col))
        weights = operations.to_memory_config(norm_w, operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, weights) != addresses(operations, norm_w):
            own(weights)
        if use_norm_batch:
            from gdn_vsplit import execute as split_execute
            output, states, bridge = split_execute(mesh, *packed, initial, z=z, norm_w=weights,
                experimental=True, batch_norm=True, synchronize=False, output_memory=operations.L1_MEMORY_CONFIG,
                **(dict(root=norm_source_root) if norm_source_root is not None else {}))
            owned.extend([output, states, bridge])
        else:
            output, states = execute(mesh, *packed, initial, kernels, z=z, norm_w=weights)
            owned.extend([output, states])
        if packed_checkpoints and not defer_conv_publication:
            from gdn_conv_prefix_copy import copy_prefix
            copy_prefix(mesh, windows, conv_states, rows, **(dict(reuse_zero_tile=True) if prefix_zero_reuse else {}))
        elif not packed_checkpoints:
            for source, destination in zip(prefixes[-1], conv_states, strict=True):
                operations.copy(source, destination)
        if [addresses(operations, state) for state in conv_states] != original_addresses:
            raise AssertionError('Batched convolution changed stable state addresses')
        return dict(output=output, states=states, conv_prefixes=prefixes, owned=owned, packed_conv_states=windows,
                    mesh=mesh, packed_checkpoints=packed_checkpoints, prefix_zero_reuse=prefix_zero_reuse,
                    materialized_conv_prefixes=() if packed_checkpoints else selected,
                    available_conv_prefixes=tuple(range(1, rows + 1)) if packed_checkpoints else selected,
                    hoisted_input=True, batched_convolution=True, dma_windows=dma_windows,
                    norm_batch=use_norm_batch, deferred_conv_publication=defer_conv_publication)
    except BaseException:
        release_owned(operations, owned)
        raise
