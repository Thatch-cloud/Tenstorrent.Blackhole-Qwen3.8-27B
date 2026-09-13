"""Read-only native stage snapshots for the bounded ladder diagnostic, not timing evidence."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


INCLUDE = '#include <cstdint>'
INCLUDE_AFTER = INCLUDE + '\n#include "api/debug/dprint.h"'
REDUCE = '        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);'
SNAPSHOT = '''
        if constexpr (!QWEN_DRAFT_EXP_APPROX) {
            UNPACK((
                CircularBuffer(alias_prev_sum).wait_front(Sq_chunk_t);
                CircularBuffer(alias_prev_max).wait_front(Sq_chunk_t);
                CircularBuffer(alias_mm2_prev_out).wait_front(out_chunk_tiles);
                DEVICE_PRINT("QWEN_STAGE q={} numerator={:.9f} denominator={:.9f} maximum={:.9f}\\n",
                    local_q_start + q_iter - iter_q_start,
                    TSLICE(alias_mm2_prev_out, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=5, .w1=6, .ws=1}),
                        true, true),
                    TSLICE(alias_prev_sum, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=1, .ws=1}),
                        true, true),
                    TSLICE(alias_prev_max, 0,
                        (SliceRange{.h0=2, .h1=3, .hs=1, .w0=0, .w1=1, .ws=1}),
                        true, true));
            ));
        }
'''


@contextmanager
def stage_snapshots():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += (
            (INCLUDE, INCLUDE_AFTER), (REDUCE, REDUCE + SNAPSHOT))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
