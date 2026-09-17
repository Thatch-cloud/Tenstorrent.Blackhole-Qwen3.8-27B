"""Compare folded target attention with and without the qualified norm reader."""

from dspark_norm_request_variants import POLICIES as NORM_POLICIES, SCHEDULE
from dspark_norm_request_variants import summarize_variants as summarize_norm
from dspark_target_attention_variants import validate_route


POLICIES = {arm: dict(policy, target_attention_t16=True) for arm, policy in NORM_POLICIES.items()}


def summarize_variants(requests):
    for value in requests:
        validate_route(value, 'parallel')
    result = summarize_norm(requests)
    result['comparison'] = 'Folded T16 target attention versus folded attention plus scatter norm'
    return result
