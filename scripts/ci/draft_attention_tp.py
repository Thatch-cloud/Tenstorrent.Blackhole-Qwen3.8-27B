"""draft_attention's GQA validation and native SDPA call at any served width.

draft_attention.py is frozen evidence (its sha256 is pinned by the TP2 attach) and carries the pair's drafter geometry: 16
query heads on 4 KV heads per chip. The drafter is TP-sharded, not replicated (dflash_device.py: its heads, MLP columns,
embedding width and vocabulary are split over the chips), so a four-card chip holds 8 query heads on 2 KV heads - the same
group of four. This twin is validate_attention and draft_sdpa with the head counts from tp_shapes (draft_heads,
draft_kv_heads); the SDPA call is verbatim (HiFi4, fp32 accumulation, an 8x8 grid, q chunk 32, exp exact, scale 128^-0.5).
The composed path (composed_draft_attention, its fused dot and row-sum kernels) stays the pinned module's and keeps its
own 16 / 4 check, so it refuses at four cards rather than run at a geometry nobody qualified.

tp_addresses.install() rebinds draft_attention.validate_attention / draft_sdpa to these at QWEN_FAST_TP=4 only, so the
unedited importers (draft_attention_branch, proposal_native_attention, dflash_t16_native_attention, whose bytes are hashed
by the T16 native gate) reach the twin.
"""

import tp_shapes

def validate_attention(operations, query, key, value, mask):
    found = tp_shapes.active()
    heads, kv_heads = found.draft_heads, found.draft_kv_heads
    query_shape, key_shape = tuple(query.shape), tuple(key.shape)
    if (query_shape != (1, heads, 32, 128) or len(key_shape) != 4 or key_shape[:2] != (1, kv_heads)
            or key_shape[2] % 32 or key_shape[3] != 128 or tuple(value.shape) != key_shape
            or tuple(mask.shape) != (1, 1, 32, key_shape[2])):
        raise ValueError('TP%d draft GQA uses %d query and %s KV heads with padded block32'
                         % (found.tp, heads, tp_shapes.number_word(kv_heads)))
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 draft attention operands required')


def draft_sdpa(operations, query, key, value, mask, *, streaming=False, key_chunk_size=32):
    validate_attention(operations, query, key, value, mask)
    if type(key_chunk_size) is not int or key_chunk_size not in (32, 64) or key.shape[2] % key_chunk_size:
        raise ValueError('Aligned 32/64-key native draft chunk required')
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=not streaming, packer_l1_acc=False)
    program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32,
        k_chunk_size=key_chunk_size, exp_approx_mode=False)
    return operations.transformer.scaled_dot_product_attention(query, key, value,
        attn_mask=mask, is_causal=False, scale=128 ** -0.5, program_config=program,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG)
