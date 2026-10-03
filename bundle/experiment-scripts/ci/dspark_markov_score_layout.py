"""Opt-in Markov feedback with fused score layout; native dot product and argmax."""

import math

from dspark_markov_device import validate
from dspark_score_layout import execute as score_layout


def execute(operations, mesh, anchor, base_logits, predecessor, successor, owned, *, on_step_enqueued=None):
    steps, vocabulary = validate(operations, anchor, base_logits, predecessor, successor)
    if on_step_enqueued is not None and not callable(on_step_enqueued):
        raise ValueError('Step observer must be callable')
    if list(mesh.shape) != [1, 2]:
        raise ValueError('Two-chip score-layout feedback requires a 1x2 mesh')
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
        scores = score_layout(operations, mesh, base_logits, bias, step, retain)
        token = retain(operations.argmax(scores, dim=-1, keepdim=False))
        previous = retain(operations.reshape(token, (1, 1, 1, 1)))
        records.append(dict(token=token, scores=scores))
        if on_step_enqueued is not None:
            on_step_enqueued(step)
    return records
