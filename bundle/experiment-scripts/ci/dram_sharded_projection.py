"""Unqualified native DRAM-sharded projection; no programmable-DRAM prefetcher."""

from tiny_tile_matmul import PROJECTIONS


def geometry(name, banks):
    if name not in PROJECTIONS or type(banks) is not int or banks not in (7, 8):
        raise ValueError('Known TP2 projection and measured seven/eight-bank geometry required')
    inner, width, unused, dtype, activation = PROJECTIONS[name]
    workers = 4 if name in ('gate', 'up') else 2
    block = 8
    if inner % (workers * 32 * block) or width % (workers * 32):
        raise ValueError('Native width shards must have complete K blocks and output tiles')
    shard_width = ((width + banks * 32 - 1) // (banks * 32)) * 32
    return dict(inner=inner, width=width, banks=banks, workers=workers,
        weight_shard=(inner, shard_width), input_shard=(32, inner // workers),
        output_shard=(32, width // workers), in0_block_w=block,
        per_core_M=1, per_core_N=width // workers // 32,
        dtype=dtype, activation=activation)


def configurations(operations, mesh, name):
    dram_grid = mesh.dram_grid_size()
    if dram_grid.y != 1:
        raise ValueError('One measured DRAM bank row required')
    plan = geometry(name, dram_grid.x)
    bank_cores = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0),
        operations.CoreCoord(dram_grid.x - 1, 0))])
    spec = operations.ShardSpec(bank_cores, plan['weight_shard'], operations.ShardOrientation.ROW_MAJOR)
    weights = operations.MemoryConfig(operations.TensorMemoryLayout.WIDTH_SHARDED,
        operations.BufferType.DRAM, spec)
    inputs = operations.create_sharded_memory_config((1, 1, 32, plan['inner']),
        core_grid=operations.CoreGrid(x=plan['workers'], y=1), strategy=operations.ShardStrategy.WIDTH,
        orientation=operations.ShardOrientation.ROW_MAJOR)
    outputs = operations.MemoryConfig(operations.TensorMemoryLayout.WIDTH_SHARDED, operations.BufferType.L1)
    program = operations.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=plan['in0_block_w'], per_core_M=1, per_core_N=plan['per_core_N'],
        fused_activation=operations.UnaryOpType.SILU if plan['activation'] else None)
    return dict(plan=plan, weights=weights, inputs=inputs, outputs=outputs, program=program)


def execute(operations, source, weight, config, compute, retain, *, preserve_partials=False):
    if type(preserve_partials) is not bool:
        raise ValueError('Explicit FP32 reload correction selection required')
    plan = config['plan']
    if (tuple(source.shape) != (1, 1, 16, plan['inner']) or source.dtype != operations.bfloat16
            or tuple(weight.shape)[-2:] != (plan['inner'], plan['width'])
            or weight.dtype != getattr(operations, plan['dtype'])
            or weight.memory_config() != config['weights']):
        raise ValueError('T16 BF16 input and unchanged-precision DRAM-sharded weights required')
    staged = retain(operations.to_memory_config(source, config['inputs']))
    if preserve_partials:
        from dram_projection_reload import execute as corrected
        partial = corrected(operations, staged, weight, config, compute, retain)
    else:
        partial = retain(operations.linear(staged, weight, program_config=config['program'],
            compute_kernel_config=compute, memory_config=config['outputs']))
    return retain(operations.to_memory_config(partial, operations.L1_MEMORY_CONFIG))
