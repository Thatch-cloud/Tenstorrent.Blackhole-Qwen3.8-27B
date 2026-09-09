"""Replicated native DSpark Markov prototype; full vocabulary and device feedback, no target integration."""

import math


def validate(operations, anchor, base_logits, predecessor, successor):
    shape = tuple(base_logits.shape)
    if (len(shape) != 4 or shape[:2] != (1, 1) or shape[2] not in (3, 7, 15)
            or shape[3] not in (64, 248320) or tuple(anchor.shape) != (1, 1, 1, 1)
            or tuple(predecessor.shape) != (1, 1, shape[3], 256)
            or tuple(successor.shape) != (1, 1, 256, shape[3])):
        raise ValueError('One anchor, explicit proposal width and complete supported vocabulary/rank required')
    for tensor, dtype, layout in ((anchor, operations.uint32, operations.ROW_MAJOR_LAYOUT),
            (base_logits, operations.float32, operations.TILE_LAYOUT),
            (predecessor, operations.bfloat16, operations.ROW_MAJOR_LAYOUT),
            (successor, operations.bfloat16, operations.TILE_LAYOUT)):
        if tensor.dtype != dtype or tensor.layout != layout or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG:
            raise ValueError('Explicit interleaved DRAM operand dtype and layout required')
    return shape[2], shape[3]


def execute(operations, anchor, base_logits, predecessor, successor, owned):
    steps, vocabulary = validate(operations, anchor, base_logits, predecessor, successor)
    grid = (2, 1) if vocabulary == 64 else (10, 10)
    program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid,
        in0_block_w=1, out_subblock_h=1, out_subblock_w=1, per_core_M=1,
        per_core_N=math.ceil(vocabulary / (32 * math.prod(grid))),
        fuse_batch=True, fused_activation=None, mcast_in0=True)
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)

    def retain(value):
        owned.append(value)
        return value

    previous = anchor
    records = []
    for step in range(steps):
        embedding = retain(operations.embedding(previous, predecessor, layout=operations.ROW_MAJOR_LAYOUT))
        latent = retain(operations.to_layout(embedding, operations.TILE_LAYOUT))
        bias = retain(operations.matmul(latent, successor, dtype=operations.float32,
            program_config=program, compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
        base = retain(operations.slice(base_logits, (0, 0, step, 0), (1, 1, step + 1, vocabulary)))
        scores = retain(operations.add(base, bias, memory_config=operations.DRAM_MEMORY_CONFIG))
        linear = retain(operations.untilize(scores, use_multicore=True))
        token = retain(operations.argmax(linear, dim=-1, keepdim=False))
        previous = retain(operations.reshape(token, (1, 1, 1, 1)))
        records.append(dict(token=token, scores=scores))
    return records
