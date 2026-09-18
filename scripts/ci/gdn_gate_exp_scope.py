"""Opt-in compute-only fusion scope; ordinary serving remains unchanged."""

from contextlib import contextmanager
import hashlib

from gdn_gate_exp_fusion import transform
from gdn_gate_exp_gate import REPORT_SHA256
from gdn_gate_exp_report import KERNEL


@contextmanager
def scoped_gate_exp(admission):
    import gdn_shared_qk_pipeline as pipeline
    if admission.get('report_sha256') != REPORT_SHA256 or admission.get('kernel') != KERNEL:
        raise ValueError('Source-qualified gate-exp admission required')
    original = pipeline.load_kernels
    if getattr(original, '_gate_exp_fusion', False):
        raise ValueError('Nested gate-exp scopes are forbidden')
    audit = dict(report_sha256=REPORT_SHA256, kernels=[], restored=False)

    def kernels(root):
        result = {stage: dict(parts) for stage, parts in original(root).items()}
        before = result['recurrence']['compute']
        after = transform(before)
        if (hashlib.sha256(before.encode()).hexdigest() != KERNEL['control_sha256']
                or hashlib.sha256(after.encode()).hexdigest() != KERNEL['candidate_sha256']):
            raise ValueError('Constructed recurrence differs from admitted gate-exp kernel')
        result['recurrence']['compute'] = after
        audit['kernels'].append(dict(KERNEL))
        return result

    kernels._gate_exp_fusion = True
    pipeline.load_kernels = kernels
    try:
        yield audit
    finally:
        if pipeline.load_kernels is not kernels:
            raise ValueError('Recurrence loader changed outside its owning scope')
        pipeline.load_kernels = original
        audit['restored'] = True
