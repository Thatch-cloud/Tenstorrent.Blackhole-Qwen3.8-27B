"""Select the failing folded row in existing root DPRINT snapshots only."""

import re


def transform(source):
    pattern = re.compile(
        r'(TSLICE\((cb_prev_sum|cb_precise_reciprocal|cb_out_accumulate_im), 0,\s*)'
        r'\(SliceRange\{\.h0=0, \.h1=4, \.hs=1, \.w0=0, \.w1=([14]), \.ws=1\}\)')
    matches = list(pattern.finditer(source))
    counts = {buffer: sum(match[2] == buffer for match in matches)
        for buffer in ('cb_prev_sum', 'cb_precise_reciprocal', 'cb_out_accumulate_im')}
    if counts != {'cb_prev_sum': 1, 'cb_precise_reciprocal': 1, 'cb_out_accumulate_im': 3}:
        raise ValueError('Exact five root normalization diagnostic snapshots required')

    def select(match):
        column = 8 if match[2] == 'cb_out_accumulate_im' else 0
        return (match[1] + '(SliceRange{.h0=23, .h1=24, .hs=1, '
            f'.w0={column}, .w1={column + 1}, .ws=1' + '})')

    return pattern.sub(select, source)
