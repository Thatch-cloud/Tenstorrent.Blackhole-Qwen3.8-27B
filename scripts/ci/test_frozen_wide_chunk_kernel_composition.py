"""Cross-module test: the four kernel-level fixes this port composes into
dspark-native-8k-attention-probe.py's main() - scalar_reciprocal (already
active via --scalar-reciprocal, reproduced here rather than imported since
it belongs to dspark_ladder_scalar_reciprocal.py, untouched by this port),
frozen_wide_chunk_sum_update.sum_update_scope, frozen_wide_chunk_score_center.
score_center_scope, frozen_wide_chunk_scratch.kernel_scope - must apply, in
that order, against a compute_common.hpp containing every anchor each one
targets, with every anchor found exactly once at the point its own patch
runs. This is what "the substitution anchors do not collide with the
scratch-CB substitution already in place" actually means operationally:
each later patch's anchor must survive every earlier patch's output intact.

This does not use the real compute_common.hpp (not vendored here) - the
fixture below is built directly from each module's own BEFORE/anchor
constants, so it is not an independent check that those anchors exist in
the real file (frozen_wide_chunk_scratch.py's own tests already establish
that risk profile - this file is scoped to composition order and anchor
survival between the four patches, not anchor-vs-real-source correctness).
"""

import unittest

import frozen_wide_chunk_scratch as scratch
import frozen_wide_chunk_score_center as score_center
import frozen_wide_chunk_sum_update as sum_update


# dspark_ladder_scalar_reciprocal.py's own anchor (the reciprocal fix,
# already active via --scalar-reciprocal before this port's three run) -
# reproduced, not imported: that module belongs to the existing frozen
# recipe, not this port, and is not staged/patched by any code here.
RECIP_BEFORE = 'void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {'
RECIP_BODY = '\n    if constexpr (!QWEN_DRAFT_EXP_APPROX) {\n        /* ... */\n        return;\n    }\n'


def synthetic_kernel_source():
    """A compute_common.hpp fixture containing exactly the anchor text each
    of the four fixes targets, each copied from the modules' own constants
    (not hand-retyped), assembled in one function body the way the real
    kernel's normalization/accumulation code is structured."""
    return ('#include "api/compute/bcast.h"\n'
        '#include <cstdint>\n'
        '\n'
        'enum SDPAType {\n'
        '    kFake\n'
        '};\n'
        '\n' + RECIP_BEFORE + '\n'
        '    // original body\n'
        '}\n'
        '\n'
        'void other_stuff() {\n' + sum_update.BEFORE + '\n'
        + score_center.MASK + '\n'
        + score_center.INIT + '\n'
        + score_center.SUBTRACT + '\n'
        + scratch.KERNEL_FINAL_CALL + '\n'
        '}\n')


def apply_once(source, before, after, *, label):
    count = source.count(before)
    if count != 1:
        raise AssertionError(f'{label}: anchor found {count} times (expected exactly 1): {before[:80]!r}')
    return source.replace(before, after, 1)


class CompositionOrderTests(unittest.TestCase):
    """The ladder's own order, dspark-ladder-attention-probe.py:77-85:
    scalar_reciprocal, scalar_sum_update, scalar_score_center,
    scratch_normalization - mirrored here as scalar_reciprocal (reproduced),
    sum_update_scope, score_center_scope, kernel_scope."""

    def test_all_four_compose_in_ladder_order_with_no_anchor_collisions(self):
        source = synthetic_kernel_source()

        # 1. scalar_reciprocal
        source = apply_once(source, RECIP_BEFORE, RECIP_BEFORE + RECIP_BODY, label='scalar_reciprocal')

        # 2. sum_update_scope
        after_sum = (f'                if constexpr (!QWEN_DRAFT_EXP_APPROX && '
            f'get_compile_time_arg_val(3) == {sum_update.SKT}) {{\n'
            '#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0\n' + sum_update.BODY + '#endif\n'
            '                } else {\n' + sum_update.BEFORE + '\n                }')
        source = apply_once(source, sum_update.BEFORE, after_sum, label='sum_update_scope')

        # 3. score_center_scope (its own four substitutions, in its own internal order)
        source = apply_once(source, '#include <cstdint>',
            '#include <cstdint>\n#include "api/debug/dprint.h"', label='score_center.cstdint')
        source = apply_once(source, score_center.HELPER_ANCHOR,
            score_center.HELPER + score_center.HELPER_ANCHOR, label='score_center.helper')
        source = apply_once(source, score_center.MASK,
            score_center._mask_after(score_center.SKT), label='score_center.mask')
        source = apply_once(source, score_center.INIT,
            score_center._init_after(score_center.SKT), label='score_center.init')
        source = apply_once(source, score_center.SUBTRACT,
            score_center._subtract_after(score_center.SKT), label='score_center.subtract')

        # 4. kernel_scope (its own three substitutions)
        source = apply_once(source, '#include "api/compute/bcast.h"',
            '#include "api/compute/bcast.h"\n#include "api/compute/sfpu_binary_bcast.h"',
            label='kernel_scope.bcast_include')
        source = apply_once(source, 'enum SDPAType {', scratch.KERNEL_HELPER + '\nenum SDPAType {',
            label='kernel_scope.enum')
        after_final = (f'            if constexpr (!QWEN_DRAFT_EXP_APPROX && '
            f'get_compile_time_arg_val(3) == {scratch.SKT}) {{\n'
            '                static_assert(Sq_chunk_t == 1);\n'
            '                qwen_normalize_scratch<vDHt>(alias_mm2_prev_out, alias_prev_sum, '
            'get_compile_time_arg_val(42), cb_out);\n'
            '            } else {\n' + scratch.KERNEL_FINAL_CALL + '\n            }')
        source = apply_once(source, scratch.KERNEL_FINAL_CALL, after_final, label='kernel_scope.final_call')

        # Every insertion landed in the fully-composed text. Most markers are
        # truly one-shot (a helper definition or an #include, inserted once);
        # qwen_scalar_score_transform is the score_center helper's own
        # function NAME, which legitimately appears once as its definition
        # and again at each of its two call sites (MASK and INIT substitute
        # a call to it; SUBTRACT does not) - present, not exactly-once, is
        # the correct check there.
        for marker, label, exact in (
                (RECIP_BODY.strip(), 'reciprocal body', True),
                ('current[offset] = current[offset] + previous[offset]', 'sum_update body', True),
                ('qwen_scalar_score_transform', 'score_center helper (definition + call sites)', False),
                ('qwen_normalize_scratch<vDHt>', 'kernel_scope call site', True),
                ('#include "api/debug/dprint.h"', 'score_center debug include', True),
                ('#include "api/compute/sfpu_binary_bcast.h"', 'kernel_scope sfpu include', True)):
            with self.subTest(marker=label):
                if exact:
                    self.assertEqual(source.count(marker), 1, f'{label} missing or duplicated in composed text')
                else:
                    self.assertGreaterEqual(source.count(marker), 1, f'{label} missing from composed text')

    def test_reordering_score_center_before_sum_update_still_composes(self):
        """Sanity check that the ORDER matters for correctness claims but not
        for mere anchor survival here - these four patches happen to target
        disjoint call sites (confirmed by the main test), so swapping
        sum_update and score_center's relative order doesn't itself break
        anchor uniqueness. This does NOT mean the order is interchangeable
        for the kernel's actual numerical behaviour - only that this test
        file's anchor-collision check isn't accidentally order-dependent in
        a way that would hide a real collision."""
        source = synthetic_kernel_source()
        source = apply_once(source, RECIP_BEFORE, RECIP_BEFORE + RECIP_BODY, label='scalar_reciprocal')
        source = apply_once(source, '#include <cstdint>',
            '#include <cstdint>\n#include "api/debug/dprint.h"', label='score_center.cstdint')
        source = apply_once(source, score_center.HELPER_ANCHOR,
            score_center.HELPER + score_center.HELPER_ANCHOR, label='score_center.helper')
        source = apply_once(source, score_center.MASK,
            score_center._mask_after(score_center.SKT), label='score_center.mask')
        source = apply_once(source, score_center.INIT,
            score_center._init_after(score_center.SKT), label='score_center.init')
        source = apply_once(source, score_center.SUBTRACT,
            score_center._subtract_after(score_center.SKT), label='score_center.subtract')
        after_sum = (f'                if constexpr (!QWEN_DRAFT_EXP_APPROX && '
            f'get_compile_time_arg_val(3) == {sum_update.SKT}) {{\n'
            '#if defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0\n' + sum_update.BODY + '#endif\n'
            '                } else {\n' + sum_update.BEFORE + '\n                }')
        source = apply_once(source, sum_update.BEFORE, after_sum, label='sum_update_scope')
        self.assertIn('qwen_scalar_score_transform', source)
        self.assertIn('current[offset] = current[offset] + previous[offset]', source)


if __name__ == '__main__':
    unittest.main()
