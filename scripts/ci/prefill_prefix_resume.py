"""Explicit eager suffix execution after a complete hybrid-prefix restore."""


def resume_eager_prefill(operations, model, token_ids, page_table, *, actual_len,
        prefix_position, restore, chunk_size=2048, flex_sdpa=True):
    if (type(actual_len) is not int or not 1 < actual_len <= 262144
            or type(prefix_position) is not int or not 0 < prefix_position < actual_len
            or type(chunk_size) is not int or chunk_size != 2048
            or prefix_position % chunk_size or not callable(restore)
            or tuple(token_ids.shape) != (1, actual_len)):
        raise ValueError('A complete aligned prefix and nonempty text suffix are required')
    num_full, tail_real = divmod(actual_len, chunk_size)
    restore(prefix_position)
    last_hidden = None
    try:
        for chunk_index in range(prefix_position // chunk_size, num_full):
            chunk_start = chunk_index * chunk_size
            tokens = token_ids[:, chunk_start:chunk_start + chunk_size]
            model._set_vision_merge(tokens, None, 0)
            last_hidden = model._forward_prefill_chunk_masked_tp(
                tokens, chunk_size, chunk_start, page_table, chunk_size, flex_sdpa=flex_sdpa)
            operations.synchronize_device(model.device)
            if chunk_index + 1 < num_full or tail_real:
                operations.deallocate(last_hidden)
                last_hidden = None
        if tail_real:
            chunk_start = num_full * chunk_size
            return model.prefill_masked_bucket(token_ids[:, chunk_start:actual_len], page_table,
                actual_len=tail_real, chunk_start=chunk_start, flex_sdpa=flex_sdpa,
                vision_tokens=None, vis_row_offset=0)
        return model._masked_bucket_logits_tp(last_hidden, chunk_size, chunk_size)
    finally:
        if last_hidden is not None:
            operations.deallocate(last_hidden)
