"""Unqualified full-width learned K/V history projection candidate; no serving integration."""

from dspark_projection import compute_config, require_tensor


POLICY = 'Full learned TP2 K/V projection; 8x8 two-dimensional multicast; HiFi4 FP32 result then BF16 cast'


def geometry(rows):
    if type(rows) is not int or not 32 <= rows <= 4096 or rows % 32:
        raise ValueError('Tile-aligned historical rows from 32 through 4096 required')
    return dict(compute_with_storage_grid_size=(8,8),in0_block_w=4,
        out_subblock_h=1,out_subblock_w=2,per_core_M=(rows+255)//256,per_core_N=2,
        transpose_mcast=False,fused_activation=None)


def project(operations, mesh, context, weight, retain):
    if not callable(retain) or mesh is None:
        raise ValueError('Explicit history projection mesh and caller-owned tensors required')
    shape = tuple(context.shape)
    if len(shape)!=4 or shape[:2]!=(1,1) or shape[-1]!=5120:
        raise ValueError('Complete 5120-channel historical context required')
    configuration = geometry(shape[2])
    grid = mesh.compute_with_storage_grid_size()
    if grid.x<8 or grid.y<8:
        raise ValueError('History projection requires an available 8x8 worker grid')
    require_tensor(operations,context,shape,operations.bfloat16)
    require_tensor(operations,weight,(1,1,5120,512),operations.bfloat16)
    program = operations.MatmulMultiCoreReuseMultiCastProgramConfig(**configuration)
    partial = retain(operations.matmul(context,weight,dtype=operations.float32,program_config=program,
        compute_kernel_config=compute_config(operations),memory_config=operations.DRAM_MEMORY_CONFIG))
    return dict(partial=partial,projected=retain(operations.typecast(partial,operations.bfloat16)))
