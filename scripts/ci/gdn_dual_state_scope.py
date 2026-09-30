"""Copy-only recurrence construction scope; compose with the existing norm-prefetch builder."""

from contextlib import contextmanager
import hashlib

from gdn_dual_state_copy import transform
from gdn_dual_state_gate import KERNEL, REPORT_SHA256


@contextmanager
def scoped_copy(admission):
    import gdn_shared_qk_pipeline as pipeline
    if admission.get('report_sha256') != REPORT_SHA256 or admission.get('kernel') != KERNEL:
        raise ValueError('Source-qualified dual-state admission required')
    original = pipeline.load_kernels
    if getattr(original, '_dual_state_copy', False):
        raise ValueError('Nested dual-state scopes are forbidden')
    audit = dict(report_sha256=REPORT_SHA256, kernels=[], restored=False)

    def kernels(root):
        result = {stage: dict(parts) for stage, parts in original(root).items()}
        before = result['recurrence']['compute']
        after = transform(before)
        if (hashlib.sha256(before.encode()).hexdigest() != KERNEL['control_sha256']
                or hashlib.sha256(after.encode()).hexdigest() != KERNEL['candidate_sha256']):
            raise ValueError('Constructed recurrence differs from admitted simulator kernel')
        result['recurrence']['compute'] = after
        audit['kernels'].append(dict(KERNEL))
        return result

    kernels._dual_state_copy = True
    pipeline.load_kernels = kernels
    try:
        yield audit
    finally:
        if pipeline.load_kernels is not kernels:
            raise ValueError('Recurrence loader changed outside its owning scope')
        pipeline.load_kernels = original
        audit['restored'] = True
