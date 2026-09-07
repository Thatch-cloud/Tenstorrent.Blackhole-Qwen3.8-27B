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


def validate_attention(operations, query, key, value, mask):
    query_shape, key_shape = tuple(query.shape), tuple(key.shape)
    if (query_shape != (1, 16, 32, 128) or len(key_shape) != 4 or key_shape[:2] != (1, 4)
            or key_shape[2] % 32 or key_shape[3] != 128 or tuple(value.shape) != key_shape
            or tuple(mask.shape) != (1, 1, 32, key_shape[2])):
        raise ValueError('TP2 draft GQA uses 16 query and four KV heads with padded block32')
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 draft attention operands required')


def draft_sdpa(operations, query, key, value, mask, *, streaming=False):
    validate_attention(operations, query, key, value, mask)
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=not streaming, packer_l1_acc=False)
    program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32,
        k_chunk_size=32, exp_approx_mode=False)
    return operations.transformer.scaled_dot_product_attention(query, key, value,
        attn_mask=mask, is_causal=False, scale=128 ** -0.5, program_config=program,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG)


def pairwise_column_sum(operations, value, retain):
    total = value
    tails = []
    while total.shape[-1] > 1:
        shape = tuple(total.shape)
        half = shape[-1] // 2
        if shape[-1] % 2:
            tails.append(retain(operations.slice(total, (0, 0, 0, shape[-1] - 1), shape)))
        left = retain(operations.slice(total, (0, 0, 0, 0), (*shape[:-1], half)))
        right = retain(operations.slice(total, (0, 0, 0, half), (*shape[:-1], half * 2)))
        total = retain(operations.add(left, right, dtype=operations.float32))
    for tail in tails:
        total = retain(operations.add(total, tail, dtype=operations.float32))
    return total


def pairwise_dot(operations, left, right, retain):
    batch, heads, rows, width = tuple(left.shape)
    key_rows = right.shape[2]
    if batch != 1 or tuple(right.shape)[:2] != (1, heads) or right.shape[-1] != width:
        raise ValueError('Matching single-request head batches required')
    if heads * rows * key_rows * width > 8388608:
        raise ValueError('Outer-product diagnostic exceeds its memory bound; a tiled kernel is required')
    wide_left, wide_right = [retain(operations.typecast(value, operations.float32)) for value in (left, right)]
    left_view = retain(operations.reshape(wide_left, (heads, rows, 1, width)))
    right_view = retain(operations.reshape(wide_right, (heads, 1, key_rows, width)))
    products = retain(operations.multiply(left_view, right_view, dtype=operations.float32))
    reduced = pairwise_column_sum(operations, products, retain)
    return retain(operations.reshape(reduced, (1, heads, rows, key_rows)))


def composed_draft_attention(operations, mesh, query, key, value, mask, *, inspect=None, explicit_softmax=False, wide_operands=False, pairwise_sum=False, pairwise_dots=False, fused_row_sum=False, fused_dots=False):
    from gdn_multitoken_conv import addresses, release_owned

    validate_attention(operations, query, key, value, mask)
    if fused_row_sum and (not explicit_softmax or pairwise_sum):
        raise ValueError('Fused row sum requires explicit softmax and replaces pairwise sum')
    if fused_dots and pairwise_dots:
        raise ValueError('Fused and pairwise dots are separate controls')
    if pairwise_sum and not explicit_softmax:
        raise ValueError('Pairwise reduction requires the explicit softmax control')
    protected = {addresses(operations, tensor) for tensor in (query, key, value, mask)}
    owned = []

    def retain(tensor):
        if addresses(operations, tensor) not in protected:
            owned.append(tensor)
        return tensor

    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    output = None
    try:
        keys = retain(operations.repeat_interleave(key, 4, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG))
        values = retain(operations.repeat_interleave(value, 4, dim=1, memory_config=operations.DRAM_MEMORY_CONFIG))
        if wide_operands or fused_dots:
            query, keys, values = [retain(operations.typecast(tensor, operations.float32)) for tensor in (query, keys, values)]
        if fused_dots:
            from draft_dot import fused_dot

            scores = fused_dot(mesh, query, keys, owned)
        elif pairwise_dots:
            scores = pairwise_dot(operations, query, keys, retain)
        else:
            transposed = retain(operations.transpose(keys, -1, -2))
            scores = retain(operations.matmul(query, transposed, dtype=operations.float32,
                compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
        scaled = retain(operations.multiply(scores, 128 ** -0.5, dtype=operations.float32))
        wide_mask = retain(operations.typecast(mask, operations.float32))
        masked = retain(operations.add(scaled, wide_mask, dtype=operations.float32))
        if explicit_softmax:
            maximum = retain(operations.max(masked, dim=-1, keepdim=True))
            centered = retain(operations.subtract(masked, maximum, dtype=operations.float32))
            exponentials = retain(operations.exp(centered, fast_and_approximate_mode=False))
            if fused_row_sum:
                from draft_row_sum import row_sum

                total = row_sum(mesh, exponentials, owned)
            elif pairwise_sum:
                total = pairwise_column_sum(operations, exponentials, retain)
            else:
                total = retain(operations.sum(exponentials, dim=-1, keepdim=True))
            inverse = retain(operations.reciprocal(total))
            probabilities = retain(operations.multiply(exponentials, inverse, dtype=operations.float32))
        else:
            probabilities = retain(operations.softmax(masked, dim=-1, numeric_stable=True,
                compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
        if fused_dots:
            transposed_values = retain(operations.transpose(values, -1, -2))
            output = fused_dot(mesh, probabilities, transposed_values, owned)
        elif pairwise_dots:
            transposed_values = retain(operations.transpose(values, -1, -2))
            output = pairwise_dot(operations, probabilities, transposed_values, retain)
        else:
            output = retain(operations.matmul(probabilities, values, dtype=operations.float32,
                compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
        operations.synchronize_device(mesh)
        if inspect is not None:
            inspect(query, keys, values, scores, masked, probabilities, output)
    except BaseException:
        release_owned(operations, owned)
        raise
    release_owned(operations, [tensor for tensor in owned if addresses(operations, tensor) != addresses(operations, output)])
    return output
