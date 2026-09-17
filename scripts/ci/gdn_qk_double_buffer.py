"""Unqualified two-slot normalized input rings; unchanged recurrence arithmetic."""

import hashlib
from unittest.mock import patch

from gdn_multitoken import replace_once


BUILD_RECORDS = []
ORIGINAL_ZERO = 'if (token == 0) { zero(target_base, 4 * 1024); }'
CANDIDATE_ZERO = 'if (token < 2) { zero(target_base, 4 * 1024); }'


def reader(source):
    return replace_once(source, ORIGINAL_ZERO, CANDIDATE_ZERO)


def buffers(original, stage, *, prefetch_inputs=False):
    io, fp32 = original(stage, prefetch_inputs=prefetch_inputs)
    io, fp32 = dict(io), dict(fp32)
    if stage != 'recurrence' or prefetch_inputs or fp32.get(10) != 4 or fp32.get(11) != 4:
        raise ValueError('Original single-slot shared-Q/K recurrence required')
    fp32[10] = fp32[11] = 8
    return io, fp32


def build(operations, mesh, tensors, *, root):
    import gdn_shared_qk_pipeline as pipeline

    original_kernels, original_buffers = pipeline.load_kernels, pipeline.cb_plan

    def kernels(requested_root):
        result = {stage: dict(parts) for stage, parts in original_kernels(requested_root).items()}
        before = result['recurrence']['reader']
        after = reader(before)
        result['recurrence']['reader'] = after
        BUILD_RECORDS.append(dict(control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(after.encode()).hexdigest(),
            extra_cb_bytes=32768, normalized_input_pages=8, initialized_slots=2))
        return result

    def plan(stage, *, prefetch_inputs=False):
        return buffers(original_buffers, stage, prefetch_inputs=prefetch_inputs)

    with patch.object(pipeline, 'load_kernels', kernels), patch.object(pipeline, 'cb_plan', plan):
        return pipeline.build(operations, mesh, tensors, root=root)
