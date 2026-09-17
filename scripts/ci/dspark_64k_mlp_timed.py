"""Clean full-response MLP measurement tied to its combined correctness audit."""

from contextlib import contextmanager
import inspect
import os
from unittest.mock import patch

import dspark_splitk_timed as splitk
from dspark_64k_mlp_gate import SCREEN_RUN, SCREEN_SHA256, qualify as qualify_mlp
from dspark_64k_mlp_scope import validate_result


@contextmanager
def identity_scope():
    with patch.object(splitk, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(splitk, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(splitk, 'qualify_splitk', qualify_mlp):
        yield


def qualify(directory, report_directory=None):
    with identity_scope():
        return splitk.qualify(directory, report_directory)


@contextmanager
def measurement_scope(module, arm_factory, collective):
    splitk.require_timed()
    if os.environ.get('QWEN_64K_MLP_TIMED') != '1' or os.environ.get('QWEN_64K_MLP_AUDIT', '0') != '0':
        raise ValueError('Explicit clean MLP timing selection required')
    original = module.measure_dspark_request
    signature = inspect.signature(original)

    def measure(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        if (len(arguments['prompt']) != 65536 or arguments.get('audit_features') is not False
                or arguments.get('max_new_tokens') != 256
                or any(arguments.get(name) is not True for name in
                    ('captured_publication', 'target_attention_t16', 'commit_only_gdn', 'proposal_trace', 'native_attention'))
                or any(arguments.get(name, False) for name in
                    ('fused_t16_mlp', 'gdn_shared_qk', 'score_layout', 'banked_proposal', 'native_slot_gdn'))):
            raise ValueError('Same complete clean 64K runtime required')
        arm = arm_factory(arguments['operations'], arguments['model'], collective)
        with arm.install():
            result = original(*args, **kwargs)
        validate_result(arm.audit)
        result['mlp_64k_reintegration'] = dict(audit=arm.audit,
            performance_qualified=False, serving_qualified=False)
        return result

    with patch.object(module, 'measure_dspark_request', measure):
        yield
