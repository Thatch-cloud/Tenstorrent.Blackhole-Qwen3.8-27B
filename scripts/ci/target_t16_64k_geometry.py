"""Isolated folded-verifier geometry experiments, not request admission."""

from contextlib import contextmanager
from unittest.mock import patch


def geometry(*, hardware):
    if type(hardware) is not bool:
        raise ValueError('Explicit execution backend required')
    context = 65536 if hardware else 256
    return dict(context=context, capacity=context + 256, rows=16,
        starts=(context, context + 17, context + 240, context))


def validate_ticket(start, rows, capacity, *, short_context=False, hardware):
    expected = geometry(hardware=hardware)
    if (short_context is not False or any(type(value) is not int for value in (start, rows, capacity))
            or capacity != expected['capacity'] or not 1 <= rows <= 16
            or start < expected['context'] or start + rows > capacity):
        raise ValueError('Explicit fixed-capacity T16 verifier experiment required')


@contextmanager
def geometry_scope(*, hardware):
    import attention_mask_replay
    import attention_replay

    geometry(hardware=hardware)

    def validate(start, rows, capacity, *, short_context=False):
        return validate_ticket(start, rows, capacity, short_context=short_context, hardware=hardware)

    with patch.object(attention_mask_replay, 'validate_ticket', validate), \
            patch.object(attention_replay, 'validate_ticket', validate):
        yield
