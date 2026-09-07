"""DFlash2 noncausal proposal block with sliding historical context."""


def draft_attention_mask(context_rows, block_rows=8, *, padded_rows=32, key_multiple=32, window=2048):
    import torch

    if (type(context_rows) is not int or not 0 <= context_rows <= 262144
            or type(block_rows) is not int or block_rows not in (1, 8, 32)
            or padded_rows != 32 or key_multiple not in (32, 128) or window != 2048):
        raise ValueError('Bounded context and configured DFlash2 attention geometry required')
    key_rows = ((context_rows + block_rows + key_multiple - 1) // key_multiple) * key_multiple
    query = context_rows + torch.arange(block_rows)[:, None]
    key = torch.arange(key_rows)[None, :]
    allowed = ((key < context_rows) & (query - key < window)) | ((key >= context_rows) & (key < context_rows + block_rows))
    mask = torch.full((1, 1, padded_rows, key_rows), float('-inf'), dtype=torch.bfloat16)
    mask[0, 0, :block_rows] = torch.where(allowed, 0., float('-inf')).bfloat16()
    mask[0, 0, block_rows:, context_rows] = 0
    return mask


def draft_sdpa(operations, query, key, value, mask):
    query_shape, key_shape = tuple(query.shape), tuple(key.shape)
    if (query_shape != (1, 16, 32, 128) or len(key_shape) != 4 or key_shape[:2] != (1, 4)
            or key_shape[2] % 32 or key_shape[3] != 128 or tuple(value.shape) != key_shape
            or tuple(mask.shape) != (1, 1, 32, key_shape[2])):
        raise ValueError('TP2 draft GQA uses 16 query and four KV heads with padded block32')
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 draft attention operands required')
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32,
        k_chunk_size=32, exp_approx_mode=False)
    return operations.transformer.scaled_dot_product_attention(query, key, value,
        attn_mask=mask, is_causal=False, scale=128 ** -0.5, program_config=program,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG)
