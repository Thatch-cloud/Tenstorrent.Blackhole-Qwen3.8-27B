"""Diagnostic-only stop after one complete exact request, before a second arm."""

from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch


class ControlProfileComplete(Exception):
    pass


@contextmanager
def stop_after_control(module, requests):
    original = module.measure_dspark_request

    @wraps(original)
    def measure(*args, **kwargs):
        if requests:
            raise ValueError('Only one control request may execute')
        result = original(*args, **kwargs)
        if (any(result.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))
                or result.get('length') != 65536 or not result.get('blocks')
                or result.get('committed_decode_tokens', 0) <= 0):
            raise ValueError('Complete exact 64K control required before diagnostic stop')
        requests.append(result)
        raise ControlProfileComplete('Complete control retained; no candidate or performance acceptance')

    with patch.object(module, 'measure_dspark_request', measure):
        yield
