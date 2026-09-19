"""Reader and buffer recurrence construction scope; compose with the existing norm-prefetch builder."""

from contextlib import contextmanager
import hashlib

from gdn_qk_double_buffer import reader as transform, buffers
from gdn_qk_double_buffer_gate import KERNEL, REPORT_SHA256


@contextmanager
def scoped_double_buffer(admission):
    import gdn_shared_qk_pipeline as pipeline
    if admission.get('report_sha256') != REPORT_SHA256 or admission.get('kernel') != KERNEL:
        raise ValueError('Source-qualified qk-double-buffer admission required')
    original, original_buffers = pipeline.load_kernels, pipeline.cb_plan
    if getattr(original, '_qk_double_buffer_copy', False):
        raise ValueError('Nested qk-double-buffer scopes are forbidden')
    audit = dict(report_sha256=REPORT_SHA256, kernels=[], buffer_builds=0, restored=False)

    def kernels(root):
        result = {stage: dict(parts) for stage, parts in original(root).items()}
        before = result['recurrence']['reader']
        after = transform(before)
        if (hashlib.sha256(before.encode()).hexdigest() != KERNEL['control_sha256']
                or hashlib.sha256(after.encode()).hexdigest() != KERNEL['candidate_sha256']):
            raise ValueError('Constructed recurrence differs from admitted simulator kernel')
        result['recurrence']['reader'] = after
        audit['kernels'].append(dict(KERNEL))
        return result

    kernels._qk_double_buffer_copy = True
    def plan(stage, *, prefetch_inputs=False):
        result = buffers(original_buffers, stage, prefetch_inputs=prefetch_inputs)
        audit['buffer_builds'] += 1
        return result

    pipeline.load_kernels = kernels
    pipeline.cb_plan = plan
    try:
        yield audit
    finally:
        if pipeline.load_kernels is not kernels or pipeline.cb_plan is not plan:
            raise ValueError('Recurrence loader changed outside its owning scope')
        pipeline.load_kernels = original
        pipeline.cb_plan = original_buffers
        if audit['buffer_builds'] != len(audit['kernels']):
            raise ValueError('Every transformed reader must construct its two-slot rings')
        audit['restored'] = True
