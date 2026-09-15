"""Audit-only score materialization fusion in a full-history combined request."""

from contextlib import ExitStack, contextmanager
import inspect
import os
from unittest.mock import patch


@contextmanager
def score_scope(request_module, device_class, arm_factory, hardware_audit, records):
    required = ('QWEN_64K_SCORE_AUDIT', 'QWEN_64K_SHARED_QK_AUDIT', 'QWEN_DSPARK_SFPU_REQUEST_SCREEN')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Explicit audited complete 64K score-layout candidate required')
    original = request_module.measure_dspark_request
    signature = inspect.signature(original)

    def measure(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        if (len(arguments['prompt']) != 65536 or arguments.get('audit_features') is not True
                or arguments.get('proposal_trace') is not True or arguments.get('score_layout', False)):
            raise ValueError('Full-history audited native-score control route required')
        operations, model = arguments['operations'], arguments['model']
        evidence = hardware_audit(operations, model.mesh_device, arguments['predecessor'], arguments['successor'])
        prepare = device_class.prepare_trace
        arms = []
        with ExitStack() as stack:
            def prepared(device, anchor, *, audit=False):
                if arms or not audit:
                    raise ValueError('Exactly one fully audited proposal owner required')
                arm = arm_factory(device, hardware_audit=evidence)
                stack.enter_context(arm.install())
                arms.append(arm)
                return prepare(device, anchor, audit=audit)

            with patch.object(device_class, 'prepare_trace', prepared):
                result = original(*args, **kwargs)
        if len(arms) != 1:
            raise ValueError('Score-layout candidate was not installed')
        summary = arms[0].summary()
        records.append(summary)
        result['score_64k_reintegration'] = summary
        return result

    with patch.object(request_module, 'measure_dspark_request', measure):
        yield
