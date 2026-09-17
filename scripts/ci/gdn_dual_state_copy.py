"""Unqualified shared-Q/K recurrence copy fan-out; preserve both BF16 state outputs."""

from gdn_multitoken import replace_once
from gdn_copy_pairs import ORIGINAL


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
