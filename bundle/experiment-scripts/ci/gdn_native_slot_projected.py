"""Unqualified complete T16 GDN math with read-only native recurrent and convolution state."""

from gdn_multitoken_conv import addresses, release_owned
from gdn_native_slot_recurrence import execute as recurrence
from gdn_native_slot_windows import build_windows, validate_geometry


def execute(operations, mesh, projected, state, taps, dt_bias, neg_exp_A, norm_w, *, root, experimental=False):
    if experimental is not True:
        raise ValueError('Explicit native-slot GDN experiment required')
    if len(state) != 5 or tuple(state[0].shape) != (8, 24, 128, 128):
        raise ValueError('Complete native B8 GDN state required')
    validate_geometry(tuple(projected.shape), [tuple(value.shape) for value in state[1:]])
    if len(taps) != 4 or any(tuple(tap.shape) != (1, 1, 5120) for tap in taps):
        raise ValueError('Four native convolution taps required')
    bindings = [addresses(operations, value) for value in state]
    owned = []
    try:
        windows = build_windows(mesh, projected, state[1:], experimental=True)
        owned.extend(windows)
        packed = operations.transformer.gdn_decode_conv_gates(projected, windows, taps, projected, projected,
            dt_bias, neg_exp_A, batch=16, memory_config=operations.DRAM_MEMORY_CONFIG,
            channels=5120, a_col=8192, b_col=8216)
        owned.extend(packed)
        z = operations.slice(projected, (0, 0, 5120), (1, 16, 8192), memory_config=operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, z) != addresses(operations, projected):
            owned.append(z)
        weights = operations.to_memory_config(norm_w, operations.DRAM_MEMORY_CONFIG)
        if addresses(operations, weights) != addresses(operations, norm_w):
            owned.append(weights)
        output, prefixes, bridge = recurrence(operations, mesh, *packed, state[0], z=z, norm_w=weights,
            root=root, experimental=True)
        owned.extend((output, prefixes, bridge))
        if [addresses(operations, value) for value in state] != bindings:
            raise AssertionError('Native GDN state bindings changed')
        return dict(output=output, states=prefixes, conv_prefixes=[None] * 16, owned=owned,
            packed_conv_states=windows, mesh=mesh, packed_checkpoints=True, prefix_zero_reuse=False,
            materialized_conv_prefixes=(), available_conv_prefixes=tuple(range(1, 17)),
            hoisted_input=True, batched_convolution=True, dma_windows=True, norm_batch=True,
            deferred_conv_publication=True, commit_only_gdn=True, native_slot_read=True)
    except BaseException:
        release_owned(operations, owned)
        raise
