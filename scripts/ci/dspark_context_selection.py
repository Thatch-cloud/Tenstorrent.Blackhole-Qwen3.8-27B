"""Explicit context selection for bounded offline combined-runtime experiments."""

import os


def request_context():
    value = os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT', '4096')
    if value not in ('4096', '8192'):
        raise ValueError('Only explicit 4096 or 8192 request contexts are supported')
    return int(value)


def validate_history_capacity(context, output_tokens):
    if (type(context) is not int or type(output_tokens) is not int
            or not 2 <= output_tokens <= 513 or not 1 <= context <= 8192 - output_tokens):
        raise ValueError('Draft history supports 8192 total rows including output; longer contexts need a new numerical gate')
