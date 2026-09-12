"""Explicit context selection for bounded offline combined-runtime experiments."""

import os


def request_context():
    value = os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT', '4096')
    if value not in ('4096', '8192'):
        raise ValueError('Only explicit 4096 or 8192 request contexts are supported')
    return int(value)
