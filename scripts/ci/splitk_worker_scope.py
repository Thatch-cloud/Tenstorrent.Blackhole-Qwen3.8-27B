"""Isolated worker-limit experiment; inputs, precision and key chunks unchanged."""

from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch


@contextmanager
def worker_scope(module, records):
    original = module.execute_folded

    @wraps(original)
    def execute(*args, **kwargs):
        if (kwargs.get('key_chunk_size') != 256 or kwargs.get('max_cores_per_head') != 8
                or kwargs.get('stripe_keys') is not False or kwargs.get('fp32_dest_acc') is not True):
            raise ValueError('Exact admitted eight-worker control configuration required')
        records.append(dict(key_chunk_size=256, requested_worker_limit=8, selected_worker_limit=16,
            stripe_keys=False, fp32_dest_acc=True))
        return original(*args, **dict(kwargs, max_cores_per_head=16))

    with patch.object(module, 'execute_folded', execute):
        yield
