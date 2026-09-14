"""64K full-request comparison with captured publication and native target attention."""

from dspark_norm_request_variants import POLICIES as BASE_POLICIES, SCHEDULE
from dspark_norm_request_variants import summarize_variants as summarize_norm


POLICIES = {name: dict(policy, captured_publication=True) for name, policy in BASE_POLICIES.items()}


def summarize_variants(requests):
    for value in requests:
        evidence = value.get('captured_publication')
        if not isinstance(evidence, dict) or evidence.get('enabled') is not True:
            raise ValueError('Both 64K arms must execute captured publication')
        checks = evidence.get('checks', [])
        expected = 1 + len(value['blocks']) if value.get('instrumented_timing') is True else 0
        if len(checks) != expected or any(check.get('exact') is not True or check.get('tensors') != 20 for check in checks):
            raise ValueError('Complete captured publication correctness checks required')
        tokens = value.get('prompt_tokens')
        if (not isinstance(tokens, list) or len(tokens) != 65536
                or any(type(token) is not int or token < 0 for token in tokens)
                or value.get('length') != 65536):
            raise ValueError('Exact 64K context required in both arms')
    return summarize_norm(requests)
