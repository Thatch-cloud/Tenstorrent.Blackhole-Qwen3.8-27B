"""Scoped borrowing of owned native prefix features for suffix-only prefill."""

from contextlib import contextmanager

from dspark_prefill import FullHistoryCapture, validate_chunks
from model_batch import instance_overrides


@contextmanager
def prefix_features(operations, model, position, owner, prefix_position):
    if (not isinstance(owner, FullHistoryCapture) or owner.operations is not operations
            or owner.model is not model or type(prefix_position) is not int
            or prefix_position <= 0 or prefix_position % 2048
            or type(position) is not int or prefix_position >= position
            or hasattr(owner, '_qwen_prefix_feature_borrower')):
        raise ValueError('Exclusive same-model aligned prefix feature owner required')
    chunks = tuple(chunk for chunk in owner.outputs()
        if chunk.start + chunk.rows <= prefix_position)
    validate_chunks(chunks, start=0, rows=prefix_position)
    capture = FullHistoryCapture(operations, model, position)
    capture.chunks.extend(chunks)
    capture.cursor = prefix_position

    def reject_close():
        raise ValueError('Cannot release prefix features while a request borrows them')

    with instance_overrides([(owner, '_qwen_prefix_feature_borrower', capture),
            (owner, 'close', reject_close)]):
        try:
            yield capture
        finally:
            capture.close()
