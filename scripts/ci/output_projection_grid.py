"""Unqualified output-column redistribution; preserve the native reduction loop."""

import math


def widen(operations, original, *, output_width=5120, grid=(11, 6)):
    if output_width != 5120 or grid != (11, 6):
        raise ValueError('Only the declared 5120-column, 11x6 experiment is supported')
    if (tuple(original.compute_with_storage_grid_size) != (11, 3)
            or original.per_core_M != 1 or original.per_core_N != 5
            or original.out_subblock_h != 1 or original.out_subblock_w != 1
            or original.mcast_in0 is not True or original.fuse_batch is not True
            or original.fused_activation is not None):
        raise ValueError('Unchanged native 33-core output projection configuration required')
    return operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=original.in0_block_w,
        out_subblock_h=original.out_subblock_h, out_subblock_w=original.out_subblock_w,
        per_core_M=original.per_core_M, per_core_N=math.ceil((output_width // 32) / math.prod(grid)),
        fuse_batch=original.fuse_batch, fused_activation=original.fused_activation,
        mcast_in0=original.mcast_in0)
