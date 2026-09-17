"""Explicit context selection for bounded offline combined-runtime experiments."""

import os


def request_context():
    value = os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT', '4096')
    if value == '65536' and os.environ.get('QWEN_DSPARK_64K_TRIAL') == '1':
        return 65536
    if value not in ('4096', '8192'):
        raise ValueError('Only explicit 4096 or 8192 request contexts are supported')
    return int(value)


def validate_history_capacity(context, output_tokens):
    if context == 65536:
        from dspark_64k_admission import current_admission, validate_request as validate_64k
        validate_64k(context, output_tokens)
        evidence = current_admission()
        if evidence is None or evidence.get('capacity') != 66560:
            raise ValueError('64K history requires an active qualified request scope')
        return
    from dspark_8k_admission import history_limit, validate_request
    if history_limit() == 8448:
        validate_request(context, output_tokens)
        return
    if (type(context) is not int or type(output_tokens) is not int
            or not 2 <= output_tokens <= 513 or not 1 <= context <= 8192 - output_tokens):
        raise ValueError('Draft history supports 8192 total rows including output; longer contexts need a new numerical gate')
