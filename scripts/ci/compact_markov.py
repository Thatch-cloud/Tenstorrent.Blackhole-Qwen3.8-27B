"""Simulator-only compact Markov feedback with retained per-step validity records."""

import math
import os

from compact_score_device import execute_local_winners, reduce_winners
from dspark_markov_device import validate


def execute(operations, mesh, anchor, base_logits, predecessor, successor, owned, *, on_step_enqueued=None):
    if os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR'):
        raise ValueError('Compact Markov feedback is simulator-only')
    steps, vocabulary = validate(operations, anchor, base_logits, predecessor, successor)
    if list(mesh.shape) != [1, 2] or (on_step_enqueued is not None and not callable(on_step_enqueued)):
        raise ValueError('Two-chip mesh and callable step observer required')
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
        winners = execute_local_winners(operations, mesh, base_logits, bias, step, retain)
        diagnostic = reduce_winners(operations, mesh, winners, vocabulary, retain)
        token = retain(operations.slice(diagnostic, (0, 0, 0, 3), (1, 1, 1, 4)))
        previous = token
        records.append(dict(token=token, diagnostic=diagnostic))
        if on_step_enqueued is not None:
            on_step_enqueued(step)
    return records


def validate_readback(diagnostics, tokens, vocabulary):
    if vocabulary not in (64, 248320) or len(diagnostics) not in (3, 7, 15) or len(tokens) != len(diagnostics):
        raise ValueError('Complete supported Markov chain required')
    for records, selected in zip(diagnostics, tokens, strict=True):
        if len(records) != 2 or len(selected) != 2:
            raise ValueError('Both chip replicas required')
        for record, token in zip(records, selected, strict=True):
            if (len(record) != 8 or record[1] != 0 or not 0 <= record[0] < vocabulary
                    or record[0] != record[3] or record[3] != token or any(record[4:])
                    or (record[2] & 0x7f800000) == 0x7f800000):
                raise ValueError('Invalid compact feedback chain; discard entire proposal')
    return True
