"""Unqualified reader scheduling change; preserve arithmetic and buffer footprint."""

import hashlib
from unittest.mock import patch

from gdn_multitoken import replace_once
from gdn_shared_qk_recurrence import INPUTS


BUILD_RECORDS = []
NORMALIZED = ('        gather_normalized(20, 10, token);\n'
    '        gather_normalized(21, 11, token);\n')


def reader(source):
    if not INPUTS.startswith(NORMALIZED) or not INPUTS.endswith('\n\n'):
        raise ValueError('Unchanged shared-Q/K input sequence required')
    reordered = INPUTS[len(NORMALIZED):].rstrip('\n') + '\n' + NORMALIZED + '\n'
    return replace_once(source, INPUTS, reordered)


def build(operations, mesh, tensors, *, root):
    import gdn_shared_qk_pipeline as pipeline

    original = pipeline.load_kernels
    def kernels(requested_root):
        result = {stage: dict(parts) for stage, parts in original(requested_root).items()}
        before = result['recurrence']['reader']
        after = reader(before)
        result['recurrence']['reader'] = after
        BUILD_RECORDS.append(dict(control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(after.encode()).hexdigest(), extra_cb_bytes=0))
        return result

    with patch.object(pipeline, 'load_kernels', kernels):
        return pipeline.build(operations, mesh, tensors, root=root)
