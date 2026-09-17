"""Unqualified folded-verifier component geometry; not full-request admission."""

from contextlib import contextmanager
from unittest.mock import patch

from matched_context_geometry import CONTEXTS


def geometry(context):
    if type(context) is not int or context not in CONTEXTS:
        raise ValueError('Explicit context ladder member required')
    return dict(context=context, capacity=context + 256, rows=16,
        starts=(context, context + 17, context + 240, context))


def validate_ticket(context, start, rows, capacity, *, short_context=False):
    expected = geometry(context)
    if (short_context is not False or any(type(value) is not int for value in (start, rows, capacity))
            or capacity != expected['capacity'] or not 1 <= rows <= 16
            or start < context or start + rows > capacity):
        raise ValueError('Complete context-specific folded-verifier component required')


@contextmanager
def geometry_scope(context):
    import attention_mask_replay
    import attention_replay

    geometry(context)

    def validate(start, rows, capacity, *, short_context=False):
        return validate_ticket(context, start, rows, capacity, short_context=short_context)

    with patch.object(attention_mask_replay, 'validate_ticket', validate), \
            patch.object(attention_replay, 'validate_ticket', validate):
        yield
