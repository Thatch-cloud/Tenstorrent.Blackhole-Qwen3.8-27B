"""Require explicit executed fusion policy while preserving all packed-weight checks."""

from fused_t16_admission import REPORT_SHA256 as BASELINE_SHA256
from mlp_register_epilogue_gate import REPORT_SHA256 as REGISTER_SHA256
from mlp_weight_pipeline_report import validate_fusion
from cumulative_register_scope import validate_request as validate_register


def validate_fusion_policy(request, policy='baseline'):
    if policy not in ('baseline', 'register'):
        raise ValueError('Known explicit cumulative fusion policy required')
    expected = REGISTER_SHA256 if policy == 'register' else BASELINE_SHA256
    validate_fusion(request, expected)
    identity = request.get('register_epilogue')
    if policy == 'register':
        if not isinstance(identity, dict) or identity.get('register_resident') is not True:
            raise ValueError('Executed register epilogue identity required')
        validate_register(request, identity)
    elif identity is not None:
        if identity != dict(register_resident=False, report_sha256=None,
                            constructions=0, calls=0, restored=True):
            raise ValueError('Baseline request cannot execute a register epilogue')
    return expected
