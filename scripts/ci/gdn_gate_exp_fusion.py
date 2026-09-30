"""Unqualified BF16 gate-to-FP32 exp fusion for the shared-Q/K recurrence."""

import hashlib
from unittest.mock import patch

from gdn_multitoken import replace_once


BUILD_RECORDS = []
COPY = '        copy_tiles(cb_g, cb_gf, 1);\n'
EXP = '        expc(cb_gf, cb_gexp, 1);  // [0,0] = exp(g_h)\n'


def transform(source):
    result = replace_once(source, COPY, '        expc(cb_g, cb_gexp, 1);\n')
    for removed in ('        WAIT(cb_gf, 1);\n', EXP, '        POP(cb_gf, 1);\n'):
        result = replace_once(result, removed, '')
    for anchor in ('        WAIT(cb_g, 1);', '        POP(cb_g, 1);',
                   '        WAIT(cb_gexp, 1);', '        POP(cb_gexp, 1);'):
        if result.count(anchor) != 1:
            raise ValueError('Native gate input/output lifetime required')
    if not (result.index('        WAIT(cb_g, 1);')
            < result.index('        expc(cb_g, cb_gexp, 1);')
            < result.index('        POP(cb_g, 1);')
            < result.index('        WAIT(cb_gexp, 1);')
            < result.index('        POP(cb_gexp, 1);')):
        raise ValueError('Native gate input/output ordering required')
    return result


def build(operations, mesh, tensors, *, root):
    import gdn_shared_qk_pipeline as pipeline

    original = pipeline.load_kernels
    io, fp32 = pipeline.cb_plan('recurrence', prefetch_inputs=False)
    if io.get(3) != 1 or fp32.get(13) != 1 or fp32.get(23) != 1:
        raise ValueError('Native BF16 gate and FP32 intermediate/exp buffers required')

    def kernels(requested_root):
        result = {stage: dict(parts) for stage, parts in original(requested_root).items()}
        before = result['recurrence']['compute']
        after = transform(before)
        result['recurrence']['compute'] = after
        BUILD_RECORDS.append(dict(control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(after.encode()).hexdigest(),
            removed_intermediate_cb=23, extra_cb_bytes=0, precision_changed=False,
            numerical_qualification_required=True))
        return result

    with patch.object(pipeline, 'load_kernels', kernels):
        return pipeline.build(operations, mesh, tensors, root=root)
