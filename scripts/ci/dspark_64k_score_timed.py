"""Clean combined score-layout timing; no per-block bulk KV or feature auditing."""

from contextlib import ExitStack, contextmanager
from functools import wraps
import inspect
import json
import os
from time import perf_counter
from unittest.mock import patch

import dspark_64k_shared_timed as shared
from dspark_64k_score_gate import SCREEN_RUN, SCREEN_SHA256, qualify as qualify_score


@contextmanager
def identity_scope():
    with patch.object(shared, 'SCREEN_RUN', SCREEN_RUN), patch.object(shared, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(shared, 'qualify_shared', qualify_score):
        yield


def qualify(directory, report_directory=None):
    with identity_scope():
        return shared.qualify(directory, report_directory)


@contextmanager
def measurement_scope(module, device_class, arm_factory, hardware_audit, records):
    if (os.environ.get('QWEN_64K_SCORE_TIMED') != '1'
            or os.environ.get('QWEN_64K_SHARED_QK_TIMED') != '1'
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED') != '1'
            or os.environ.get('QWEN_DSPARK_SFPU_REQUEST_SCREEN') != '0'):
        raise ValueError('Explicit clean combined score-layout timing required')
    original = module.measure_dspark_request
    signature = inspect.signature(original)
    loaded_audit = []

    @wraps(original)
    def measure(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        if (len(arguments['prompt']) != 65536 or arguments.get('audit_features') is not False
                or arguments.get('max_new_tokens') != 256 or arguments.get('proposal_trace') is not True
                or arguments.get('score_layout', False)):
            raise ValueError('Same complete clean 64K native-score control required')
        owners = (arguments['operations'], arguments['model'].mesh_device,
            arguments['predecessor'], arguments['successor'])
        reused = bool(loaded_audit and all(before is after
            for before, after in zip(loaded_audit[0], owners)))
        started = perf_counter()
        if not reused:
            loaded_audit[:] = [owners, hardware_audit(*owners)]
        evidence = loaded_audit[1]
        setup = dict(event='score_loaded_weight_audit', reused=reused,
            elapsed_ms=(perf_counter() - started) * 1000, performance_qualified=False)
        print(json.dumps(setup), flush=True)
        prepare = device_class.prepare_trace
        arms = []
        with ExitStack() as stack:
            def prepared(device, anchor, *, audit=False):
                if arms or audit:
                    raise ValueError('One clean proposal trace owner required')
                arm = arm_factory(device, hardware_audit=evidence)
                stack.enter_context(arm.install())
                arms.append(arm)
                return prepare(device, anchor, audit=False)
            with patch.object(device_class, 'prepare_trace', prepared):
                result = original(*args, **kwargs)
        if len(arms) != 1:
            raise ValueError('Score-layout candidate was not captured')
        summary = arms[0].summary()
        summary['loaded_weight_admission'] = setup
        records.append(summary)
        result['score_64k_reintegration'] = summary
        return result

    with patch.object(module, 'measure_dspark_request', measure):
        yield
