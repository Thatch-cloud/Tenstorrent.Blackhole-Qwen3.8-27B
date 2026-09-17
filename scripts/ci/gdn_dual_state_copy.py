"""Unqualified shared-Q/K recurrence copy fan-out; preserve both BF16 state outputs."""

from gdn_multitoken import replace_once
from gdn_copy_pairs import ORIGINAL
import hashlib
from unittest.mock import patch


BUILD_RECORDS = []


FEEDBACK = '''        copy_tiles(cb_snew, cb_sout, kv);
        if (it + 1 < n_inst) { copy_tiles(cb_snew, 30, kv); }'''


def transform(source):
    start = 'void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {'
    end = '    cb_push_back(o, n);\n}'
    if source.count(start) != 1 or source.count(FEEDBACK) != 1:
        raise ValueError('Exact native helper and separate output/feedback copies required')
    offset = source.index(start)
    finish = source.index(end, offset) + len(end)
    original = source[offset:finish]
    if original.count(ORIGINAL) != 1:
        raise ValueError('Unchanged single-tile register handoffs required')
    helper = replace_once(original, start,
        'void copy_state_twice(uint32_t in, uint32_t o, uint32_t feedback, uint32_t n) {')
    helper = replace_once(helper, '    cb_reserve_back(o, n);',
        '    cb_reserve_back(o, n);\n    cb_reserve_back(feedback, n);')
    helper = replace_once(helper, '        pack_tile(0, o, i);',
        '        pack_tile(0, o, i);\n        pack_tile(0, feedback, i);')
    helper = replace_once(helper, '    cb_push_back(o, n);',
        '    cb_push_back(o, n);\n    cb_push_back(feedback, n);')
    result = source[:finish] + '\n\n' + helper + source[finish:]
    return replace_once(result, FEEDBACK, '''        if (it + 1 < n_inst) {
            copy_state_twice(cb_snew, cb_sout, 30, kv);
        } else {
            copy_tiles(cb_snew, cb_sout, kv);
        }''')


def build(operations, mesh, tensors, *, root):
    import gdn_shared_qk_pipeline as pipeline
    original = pipeline.load_kernels
    io, fp32 = pipeline.cb_plan('recurrence', prefetch_inputs=False)
    if io.get(18) != 4 or io.get(30) != 4 or 18 in fp32 or 30 in fp32:
        raise ValueError('Both state destinations must retain four native BF16 tiles')

    def kernels(requested_root):
        result = {stage: dict(parts) for stage, parts in original(requested_root).items()}
        before = result['recurrence']['compute']
        after = transform(before)
        result['recurrence']['compute'] = after
        BUILD_RECORDS.append(dict(control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(after.encode()).hexdigest(),
            state_tiles=4, output_cb=18, feedback_cb=30))
        return result

    with patch.object(pipeline, 'load_kernels', kernels):
        return pipeline.build(operations, mesh, tensors, root=root)
