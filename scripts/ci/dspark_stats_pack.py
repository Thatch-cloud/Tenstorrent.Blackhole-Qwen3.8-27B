"""Scoped precise-SDPA graft adding an explicit maximum-difference pack format."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


BEFORE = '    sub_init(in0_cb, in1_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'
AFTER = '    sub_init(in0_cb, in1_cb);\n    pack_reconfig_data_format(out_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'


@contextmanager
def scoped_stats_pack():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        substitutions['compute_common.hpp'] += ((BEFORE, AFTER),
            ('        MATH((recip_tile_first_column(0)));',
             '        MATH((recip_tile_first_column<false>(0)));'))
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
