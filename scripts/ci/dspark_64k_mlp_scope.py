"""Explicit audit-only 64K MLP reintegration; preserves the existing control route."""

from contextlib import contextmanager
import inspect
import os
from unittest.mock import patch


def validate_result(audit):
    if (audit.get('rows') != 16 or audit.get('layers') != 64
            or audit.get('restored') is not True or audit.get('native_bindings_unchanged') is not True
            or len(audit.get('hits', [])) != 64
            or any(type(count) is not int or count <= 0 for count in audit['hits'])
            or audit.get('extra_weight_allocations') != 0):
        raise ValueError('All target layers must execute fused T16 and restore their native weights')


@contextmanager
def audit_scope(module, arm_factory, collective):
    required = ('QWEN_SPLITK_COMBINED', 'QWEN_64K_MLP_AUDIT', 'QWEN_DSPARK_SFPU_REQUEST_SCREEN')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Explicit combined 64K MLP correctness experiment required')
    original = module.measure_dspark_request
    signature = inspect.signature(original)

    def measure(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        arguments = bound.arguments
        if (len(arguments['prompt']) != 65536 or arguments.get('audit_features') is not True
                or any(arguments.get(name) is not True for name in
                    ('captured_publication', 'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention'))
                or any(arguments.get(name, False) for name in
                    ('fused_t16_mlp', 'gdn_shared_qk', 'score_layout', 'banked_proposal', 'native_slot_gdn'))):
            raise ValueError('Isolated full-history audited control route required')
        arm = arm_factory(arguments['operations'], arguments['model'], collective)
        with arm.install():
            result = original(*args, **kwargs)
        validate_result(arm.audit)
        result['mlp_64k_reintegration'] = dict(audit=arm.audit,
            performance_qualified=False, serving_qualified=False)
        return result

    with patch.object(module, 'measure_dspark_request', measure):
        yield
