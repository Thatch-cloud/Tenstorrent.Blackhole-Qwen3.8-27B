"""Simulator-only T32 reuse of the T16 fused score kernel; dot product and argmax unchanged."""

import os

from dspark_markov_score_layout import execute as fused
from dspark_projection import require_tensor


def execute(operations, anchor, base_logits, predecessor, successor, owned, *, mesh, on_step_enqueued=None):
    if (os.environ.get('QWEN_SIM_ONLY') != '1'
            or os.environ.get('QWEN_HARDWARE_TESTS') == '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'
            or list(mesh.shape) != [1, 2]):
        raise ValueError('Fused T32 feedback requires a simulator-only two-chip mesh')
    shape = tuple(base_logits.shape)
    if (len(shape) != 4 or shape[:3] != (1, 1, 31) or shape[3] not in (64, 248320)
            or not isinstance(owned, list)
            or (on_step_enqueued is not None and not callable(on_step_enqueued))):
        raise ValueError('Explicit experimental 31-query full-vocabulary feedback required')
    require_tensor(operations, base_logits, shape, operations.float32)
    records = []
    previous = anchor
    for start, width in ((0, 7), (7, 7), (14, 7), (21, 7), (28, 3)):
        local = operations.slice(base_logits, (0, 0, start, 0), (1, 1, start + width, shape[3]))
        owned.append(local)
        observer = None if on_step_enqueued is None else lambda step, offset=start: on_step_enqueued(offset + step)
        segment = fused(operations, mesh, previous, local, predecessor, successor, owned,
            on_step_enqueued=observer)
        if len(segment) != width:
            raise AssertionError('Every query must produce exactly one feedback token')
        records.extend(segment)
        previous = operations.reshape(segment[-1]['token'], (1, 1, 1, 1))
        owned.append(previous)
    return records
