"""Audit-only operation completion fences; never used for measured request timing."""

from contextlib import contextmanager
import inspect
from pathlib import Path

from model_batch import instance_overrides


@contextmanager
def audit_operations(operations, mesh, progress):
    if not callable(progress):
        raise ValueError('Explicit audit reporter required')
    bindings = []
    groups = ((operations, ('add', 'multiply', 'matmul', 'transpose', 'reshape', 'slice', 'concat',
        'typecast', 'rms_norm', 'zeros_like', 'softmax', 'max', 'sum', 'exp', 'reciprocal', 'pad', 'generic_op')),
        (operations.experimental, ('rotary_embedding_hf', 'all_gather_async', 'nlp_create_qkv_heads', 'nlp_concat_heads')))
    def wrap(name, original):
        def call(*args, **kwargs):
            frame = inspect.currentframe().f_back
            try:
                source, line = Path(frame.f_code.co_filename).name, frame.f_lineno
            finally:
                del frame
            progress('draft-operation', operation=name, source=source, line=line, phase='enqueue')
            result = original(*args, **kwargs)
            progress('draft-operation', operation=name, source=source, line=line, phase='synchronize')
            operations.synchronize_device(mesh)
            progress('draft-operation', operation=name, source=source, line=line, phase='complete')
            return result
        return call
    for module, names in groups:
        for name in names:
            original = getattr(module, name, None)
            if callable(original):
                bindings.append((module, name, wrap(name, original)))
    with instance_overrides(bindings):
        yield
