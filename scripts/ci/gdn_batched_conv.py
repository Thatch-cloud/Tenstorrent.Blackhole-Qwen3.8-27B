"""Parallel causal convolution windows using the unmodified native BF16 kernel."""

from gdn_multitoken import execute
from gdn_multitoken_conv import addresses, convolution_checkpoints, release_owned, run_projected, validate_projected
from gdn_prefix import independent_row


def history_windows(rows):
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16):
        raise ValueError('Supported single-sequence token width required')
    return tuple((slot, slot + rows) for slot in range(4))


def run_batched_projected(mesh, projected, initial, conv_states, taps, dt_bias, neg_exp_A, norm_w, kernels,
                          operations=None, *, conv_checkpoints=None, hoist_input=False, dma_windows=False):
    if operations is None:
        import ttnn as operations
    rows = validate_projected(tuple(projected.shape), conv_states)
    selected = convolution_checkpoints(rows, conv_checkpoints)
    if rows == 1:
        return run_projected(mesh, projected, initial, conv_states, taps, dt_bias, neg_exp_A, norm_w, kernels,
                             operations, conv_checkpoints=conv_checkpoints)
    if len(taps) != 4 or any(tuple(tap.shape) != (1, 1, 5120) for tap in taps):
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
            projected_qkv = sliced(source, (0, 0, 0), (1, rows, 5120))
            history = own(operations.concat([*[layout(state, operations.ROW_MAJOR_LAYOUT) for state in conv_states],
                                             projected_qkv], dim=1, memory_config=operations.DRAM_MEMORY_CONFIG))
            windows = []
            for start, end in history_windows(rows):
                window = layout(sliced(history, (0, start, 0), (1, end, 5120)), operations.TILE_LAYOUT)
                independent_row(operations, history, window)
                windows.append(window)
        packed = operations.transformer.gdn_decode_conv_gates(projected, windows, taps, projected, projected,
            dt_bias, neg_exp_A, batch=rows, memory_config=operations.DRAM_MEMORY_CONFIG,
            channels=5120, a_col=8192, b_col=8216)
        owned.extend(packed)
        shifted = [layout(window, operations.ROW_MAJOR_LAYOUT) for window in windows]
        prefixes = []
        for token in range(rows):
            prefix = None
            if token + 1 in selected:
                prefix = [layout(sliced(window, (0, token, 0), (1, token + 1, 5120)), operations.TILE_LAYOUT)
                          for window in shifted]
            prefixes.append(prefix)
        z = sliced(projected, (0, 0, 5120), (1, rows, 8192))
        weights = operations.to_memory_config(norm_w, operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, weights) != addresses(operations, norm_w):
            own(weights)
        output, states = execute(mesh, *packed, initial, kernels, z=z, norm_w=weights)
        owned.extend([output, states])
        for source, destination in zip(prefixes[-1], conv_states, strict=True):
            operations.copy(source, destination)
        if [addresses(operations, state) for state in conv_states] != original_addresses:
            raise AssertionError('Batched convolution changed stable state addresses')
        return dict(output=output, states=states, conv_prefixes=prefixes, owned=owned,
                    materialized_conv_prefixes=selected, hoisted_input=True, batched_convolution=True, dma_windows=dma_windows)
    except BaseException:
        release_owned(operations, owned)
        raise
