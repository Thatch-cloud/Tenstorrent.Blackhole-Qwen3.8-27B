"""Capture GDN at a completed cold-prefill chunk, before later suffix mutation."""

from contextlib import contextmanager

from model_batch import instance_overrides


@contextmanager
def checkpoint_boundary(model, checkpoint, position):
    if (type(position) is not int or position <= 0 or position >= 262144 or position % 2048
            or hasattr(model, '_qwen_prefix_checkpoint_boundary')):
        raise ValueError('One aligned cold-prefill checkpoint boundary required')
    original = model._forward_prefill_chunk_masked_tp
    evidence = dict(position=position, captured=False, complete=False, restored=False)
    cursor = 0

    def chunk(token_buf, valid_len, chunk_start, page_table, bucket, *args, **kwargs):
        nonlocal cursor
        if (type(chunk_start) is not int or chunk_start != cursor
                or type(valid_len) is not int or valid_len < 1):
            raise ValueError('One ordered cold-prefill request required')
        end = chunk_start + valid_len
        if not evidence['captured'] and end > position:
            raise ValueError('Native chunk crosses requested prefix checkpoint')
        output = original(token_buf, valid_len, chunk_start, page_table, bucket, *args, **kwargs)
        cursor = end
        if end == position:
            try:
                checkpoint.capture(position)
            except BaseException:
                checkpoint.operations.deallocate(output)
                raise
            evidence['captured'] = True
        return output

    try:
        with instance_overrides([(model, '_qwen_prefix_checkpoint_boundary', evidence),
                (model, '_forward_prefill_chunk_masked_tp', chunk)]):
            yield evidence
            if not evidence['captured']:
                raise ValueError('Requested prefix checkpoint was not reached')
            evidence['complete'] = True
    finally:
        evidence['restored'] = model._forward_prefill_chunk_masked_tp == original
