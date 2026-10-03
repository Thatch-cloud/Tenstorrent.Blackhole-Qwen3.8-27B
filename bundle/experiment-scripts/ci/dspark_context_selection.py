from frozen_context_geometry import selected_geometry
"""Explicit context selection for bounded offline combined-runtime experiments."""

import os
from frozen_context_geometry import CONTEXTS


def request_context():
    value = os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT', '8192')
    if value not in tuple(map(str, CONTEXTS)):
        raise ValueError('Explicit same-recipe ladder context required; admission is checked separately')
    return int(value)


def validate_history_capacity(context, output_tokens):
    from dspark_8k_admission import history_limit, validate_request
    if history_limit() == selected_geometry()['capacity']:
        validate_request(context, output_tokens)
        return
    if (type(context) is not int or type(output_tokens) is not int
            or not 2 <= output_tokens <= 513 or not 1 <= context <= 8192 - output_tokens):
        raise ValueError('Draft history supports 8192 total rows including output; longer contexts need a new numerical gate')
