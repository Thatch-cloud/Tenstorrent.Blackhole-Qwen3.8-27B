"""Offline full-window tail routing without extending fixed-width draft positions."""

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def runtime_scope():
    from attention_request_plan import ReplayPlan

    original = ReplayPlan.available
    evidence = dict(position_limit=262144, draft_rows=15, target_only_calls=0,
        restored=False, performance_qualified=False)

    def available(plan, position, remaining):
        choices = original(plan, position, remaining)
        if plan.start != 261888 or plan.stop > 262144:
            raise ValueError('Explicit full-window request plan required')
        if position + 15 > 262144:
            choices = tuple(choice for choice in choices if choice.rows == 1)
            if len(choices) != 1:
                raise ValueError('Prepared singleton target fallback required')
            evidence['target_only_calls'] += 1
        return choices

    try:
        with patch.object(ReplayPlan, 'available', available):
            yield evidence
    finally:
        evidence['restored'] = ReplayPlan.available is original
