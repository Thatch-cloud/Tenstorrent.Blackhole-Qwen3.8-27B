"""Scoped draft SDPA pack fix and accurate first-column correction exponential."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


BEFORE = '    sub_init(in0_cb, in1_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'
AFTER = '    sub_init(in0_cb, in1_cb);\n    pack_reconfig_data_format(out_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'
CORRECTION_BEFORE = '        MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));'
CORRECTION_AFTER = '        MATH((exp_tile_first_column<(QWEN_DRAFT_EXP_APPROX ? EXP_APPROX_MODE : true), scale_bf16>(0)));'


@contextmanager
def scoped_stats_pack():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += ((BEFORE, AFTER),
            (CORRECTION_BEFORE, CORRECTION_AFTER))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
