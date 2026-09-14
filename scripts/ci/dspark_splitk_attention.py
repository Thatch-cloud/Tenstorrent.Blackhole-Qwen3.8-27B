"""Unqualified folded-query native split-K attention; no serving admission."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_ladder_attention
from dspark_full_attention import validate_inputs


def execute_folded(operations, query, key, value, mask, owned, *, audit=None,
        key_chunk_size=32, max_cores_per_head=16):
    def retain(tensor):
        owned.append(tensor)
        return tensor

    memory = operations.DRAM_MEMORY_CONFIG
    linear = retain(operations.to_layout(query, operations.ROW_MAJOR_LAYOUT, memory_config=memory))
    grouped = retain(operations.reshape(linear, (4, 4, 32, 128)))
    swapped = retain(operations.permute(grouped, (0, 2, 1, 3), memory_config=memory))
    folded = retain(operations.reshape(swapped, (1, 4, 128, 128)))
    folded = retain(operations.to_layout(folded, operations.TILE_LAYOUT, memory_config=memory))
    mask_rows = retain(operations.repeat_interleave(mask, 4, dim=2, memory_config=memory))
    folded_mask = retain(operations.repeat(mask_rows, (4, 1, 1, 1), memory_config=memory))
    key_lanes = retain(operations.reshape(key, (4, 1, key.shape[2], 128)))
    value_lanes = retain(operations.reshape(value, (4, 1, value.shape[2], 128)))
    if audit is not None:
        audit(operations, (query, key, value, mask), (folded, key_lanes, value_lanes, folded_mask))
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8),
        q_chunk_size=0, k_chunk_size=key_chunk_size, exp_approx_mode=False,
        max_cores_per_head_batch=max_cores_per_head)
    output = retain(operations.transformer.scaled_dot_product_attention_decode(folded, key_lanes, value_lanes,
        attn_mask=folded_mask, is_causal=False, scale=128 ** -.5, program_config=program,
        compute_kernel_config=kernel, memory_config=memory))
    linear = retain(operations.to_layout(output, operations.ROW_MAJOR_LAYOUT, memory_config=memory))
    grouped = retain(operations.reshape(linear, (4, 32, 4, 128)))
    swapped = retain(operations.permute(grouped, (0, 2, 1, 3), memory_config=memory))
    restored = retain(operations.reshape(swapped, (1, 16, 32, 128)))
    return retain(operations.to_layout(restored, operations.TILE_LAYOUT, memory_config=memory))


def adapter(context, output_tokens=1024):
    fixture = dspark_ladder_attention.geometry(context, output_tokens)

    def execute(operations, mesh, query, key, value, mask, owned, *, context_rows,
            proposals, mask_validated=False):
        if (mesh is None or not isinstance(owned, list) or context_rows != fixture['capacity']
                or proposals != fixture['proposals'] or key.shape[2] != fixture['storage_keys']):
            raise ValueError('Exact full-history split-K ladder geometry required')
        validate_inputs(operations, query, key, value, mask, context_rows, proposals, mask_validated)
        padding = fixture['extra_masked_keys']
        if padding:
            key = operations.pad(key, [(0, 0), (0, 0), (0, padding), (0, 0)], 8192.)
            owned.append(key)
            value = operations.pad(value, [(0, 0), (0, 0), (0, padding), (0, 0)], -8192.)
            owned.append(value)
            mask = operations.pad(mask, [(0, 0), (0, 0), (0, 0), (0, padding)], float('-inf'))
            owned.append(mask)
        if key.shape[2] % 256:
            raise ValueError('Aligned split-K history required without truncation')
        return execute_folded(operations, query, key, value, mask, owned)

    return execute


@contextmanager
def splitk_scope():
    with patch.object(dspark_ladder_attention, 'adapter', adapter):
        yield
