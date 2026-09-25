"""Unqualified native T16 MLP-down column distribution; no weight relayout."""


def widen(operations, original):
    grid = original.compute_with_storage_grid_size
    expected = dict(in0_block_w=8, out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=5, fuse_batch=True, fused_activation=None, mcast_in0=True)
    if ((grid.x, grid.y) != (11, 3)
            or any(getattr(original, name) != value for name, value in expected.items())):
        raise ValueError('Pinned T16 8704x5120 native down-projection configuration required')
    return operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, 8), **dict(expected, per_core_N=2))
