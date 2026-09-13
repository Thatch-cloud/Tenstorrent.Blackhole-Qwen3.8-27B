"""Scoped draft SDPA explicit maximum-difference pack format."""

from contextlib import contextmanager
from unittest.mock import patch

import native_draft_sdpa


BEFORE = '    sub_init(in0_cb, in1_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'
AFTER = '    sub_init(in0_cb, in1_cb);\n    pack_reconfig_data_format(out_cb);\n    exp_tile_init<EXP_APPROX_MODE>();'
SELECTOR_ASSERT = '''static_assert(!QWEN_DRAFT_EXP_APPROX,
    "8K diagnostic must execute the precise draft specialization");
static_assert(get_compile_time_arg_val(3) == 272 && get_compile_time_arg_val(8) == 8,
    "8K diagnostic must use 8704 keys and 256-key chunks");
'''


@contextmanager
def scoped_stats_pack():
    original = native_draft_sdpa.replacements

    def replacements():
        substitutions = original()
        before, after = substitutions['sdpa.cpp'][0]
        after = after.replace('#include "compute_common.hpp"', SELECTOR_ASSERT + '#include "compute_common.hpp"')
        substitutions['sdpa.cpp'] = ((before, after),)
        substitutions['compute_common.hpp'] += ((BEFORE, AFTER),)
        return substitutions

    with patch.object(native_draft_sdpa, 'replacements', replacements):
        yield
