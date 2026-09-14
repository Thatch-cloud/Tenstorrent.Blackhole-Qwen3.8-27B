"""Unqualified device tile-fill replacing the per-chunk scalar scratch clear."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_score_sfpu
import native_draft_sdpa


START = 'void qwen_prepare_center_scratch(uint32_t maxima_cb, uint32_t scratch_cb) {'
CLEAR = '''    for (uint32_t index = 0; index < 1024; ++index) {
        scratch[index] = 0;
    }
'''


def transform(source):
    if source.count(START) != 1:
        raise ValueError('Unique centering scratch helper required')
    prefix, center = source.split(START)
    substitutions = (
        ('    reconfig_data_format_srca(maxima_cb);\n    copy_tile_init(maxima_cb);',
         '    qwen_copy_fp32_init(scratch_cb);\n    fill_tile_init();'),
        ('    copy_tile(maxima_cb, 0, 0);', '    fill_tile(0, 0.0f);'),
        (CLEAR, ''))
    for before, after in substitutions:
        if center.count(before) != 1:
            raise ValueError('Original centering initialization required')
        center = center.replace(before, after)
    return prefix + START + center


@contextmanager
def center_fill_scope():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        header = '#include "api/compute/bcast.h"'
        substitutions['compute_common.hpp'] += ((header,
            header + '\n#include "api/compute/eltwise_unary/fill.h"'),)
        return substitutions

    with patch.object(dspark_score_sfpu, 'HELPER', transform(dspark_score_sfpu.HELPER)), \
            patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
