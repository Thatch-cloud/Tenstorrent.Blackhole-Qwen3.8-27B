"""Explicit eager suffix execution after a complete hybrid-prefix restore."""

from contextlib import contextmanager

from model_batch import instance_overrides


@contextmanager
def native_resume_scope(operations, model, expected_tokens, expected_pages, *, prefix_position, restore):
    import torch

    if (getattr(model, 'num_devices', None) != 2 or getattr(model, '_chunked_trace_id', 'missing') is not None
            or hasattr(model, '_qwen_prefix_native_resume') or not callable(restore)
            or not isinstance(expected_tokens, torch.Tensor) or expected_tokens.device.type != 'cpu'
            or expected_tokens.ndim != 2 or expected_tokens.shape[0] != 1
            or expected_tokens.dtype not in (torch.int32, torch.int64)
            or not isinstance(expected_pages, torch.Tensor) or expected_pages.device.type != 'cpu'
            or expected_pages.ndim != 2 or expected_pages.shape[0] != 1
            or expected_pages.dtype not in (torch.int32, torch.int64)
            or type(prefix_position) is not int or not 0 < prefix_position < expected_tokens.shape[1]
            or prefix_position % 2048):
        raise ValueError('Explicit TP2 eager text request and host cache-hit inputs required')
    tokens, pages = expected_tokens.clone(), expected_pages.clone()
    original = model._prefill_chunked_eager_tp
    evidence = dict(calls=0, completed=False, restored=False, prefix_tokens=prefix_position,
        suffix_tokens=tokens.shape[1] - prefix_position, performance_qualified=False)

    def eager(token_ids, page_table, actual_len, num_full, chunk_size, tail_real,
            flex_sdpa=True, vision_tokens=None):
        if (evidence['calls'] or vision_tokens is not None or chunk_size != 2048
                or actual_len != tokens.shape[1] or (num_full, tail_real) != divmod(actual_len, 2048)
                or not torch.equal(token_ids, tokens) or not torch.equal(page_table, pages)):
            raise ValueError('Exactly one unchanged admitted text request may reuse this prefix')
        evidence['calls'] += 1
        result = resume_eager_prefill(operations, model, token_ids, page_table,
            actual_len=actual_len, prefix_position=prefix_position, restore=restore,
            chunk_size=chunk_size, flex_sdpa=flex_sdpa)
        evidence['completed'] = True
        return result

    try:
        with instance_overrides([(model, '_qwen_prefix_native_resume', evidence),
                (model, '_prefill_chunked_eager_tp', eager)]):
            yield evidence
            if not evidence['completed']:
                raise ValueError('Native prefix-resume route was not executed')
    finally:
        evidence['restored'] = model._prefill_chunked_eager_tp == original


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
