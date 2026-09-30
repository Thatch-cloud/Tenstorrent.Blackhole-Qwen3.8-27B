"""llk_zones: the exact, reversible zone transforms (CPU only; no kernel is compiled here).

Every transform must give back the original bytes under remove(), refuse what it cannot do exactly, and put
each sync call in the slot its thread's meaning needs. Real kernels of this repo are used as fixtures beside
synthetic ones, so an anchor or boundary rule that stops holding for them fails here, not on hardware.
"""
import ast
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import llk_zones as zones  # noqa: E402

REPO_KERNELS = (
    'scripts/ci/gdn_seq_block_compute.cpp', 'scripts/ci/gdn_seq_block_reader.cpp', 'scripts/ci/gdn_seq_block_writer.cpp',
    'optimisation/ttnn-op/k64j/kernels/compute/sdpa_flash_decode_qwen.cpp',
    'optimisation/ttnn-op/k64j/kernels/dataflow/reader_decode_qwen.cpp',
    'optimisation/ttnn-op/k64j/kernels/dataflow/reader_decode_qwen_slice.cpp',
    'optimisation/ttnn-op/k64j/kernels/dataflow/writer_decode_qwen_slice.cpp',
    'optimisation/ttnn-op/kernels-batch64/attn_prep/device/kernels/compute/attn_prep.cpp',
    'optimisation/ttnn-op/kernels-batch64/attn_prep/device/kernels/dataflow/reader_attn_prep.cpp',
    'docker/qwen-c2-graft/graft/gdn/gdn_prefill_conv_exact_compute.cpp',
)

SYNTHETIC = '''// SPDX header
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#if defined(FOO)
#include "foo.h"
#endif

// A comment that names cb_wait_front(cb, 1); and must stay untouched.
/* block comment: tile_regs_acquire(); */
#define WAIT_ONE(cb) cb_wait_front(cb, 1);

namespace {
inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
void helper(uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    copy_tile_to_dst_init_short(o);
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    const char* label = "cb_wait_front(x, 1);";
    // ---- first stage
    cb_wait_front(0, n);
    helper(16);
    if (n > 1) cb_wait_front(1, n);
    else cb_wait_front(2, n);
    for (uint32_t i = 0; i < n; ++i) {
        reconfig_data_format(0, 1);
        helper(17);
    }
    // ---- second stage
    x = y; cb_wait_front(3,
                         n);
    noc.async_read_barrier();
    out->reserve_back(4);
    ckernel::cb_wait_front(5, n);
    WAIT(6, n);
    // ---- end
}
'''

STAGES = (('FIRST', '    // ---- first stage', '    // ---- second stage', 1),
          ('SECOND', '    // ---- second stage', '    // ---- end', 1))


def read(path):
    with open(os.path.join(ROOT, path), 'rb') as handle:
        return zones.decode(handle.read())


class RoundTripTests(unittest.TestCase):
    def test_every_repo_kernel_round_trips_at_both_levels(self):
        for path in REPO_KERNELS:
            text = read(path)
            for level in zones.LEVELS:
                with self.subTest(path=path, level=level):
                    result, record = zones.instrument(text, 'X', level)
                    self.assertNotEqual(result, text)
                    self.assertEqual(zones.remove(result), text)
                    self.assertEqual(record['source_sha256'], zones.sha256(text))
                    self.assertEqual(record['instrumented_sha256'], zones.sha256(result))
                    self.assertEqual(result.count('DeviceZoneScopedN("QWEN_LLK_X")'), 1)
                    self.assertEqual(result.count(zones.HEADER_LINE), 1)

    def test_synthetic_round_trip_with_stages(self):
        result, record = zones.instrument(SYNTHETIC, 'SYN', 'stages', regions=STAGES)
        self.assertEqual(zones.remove(result), SYNTHETIC)
        self.assertEqual([zone['name'] for zone in record['zones']],
                         ['QWEN_LLK_SYN', 'QWEN_LLK_SYN_FIRST', 'QWEN_LLK_SYN_SECOND'])
        self.assertIn('    { DeviceZoneScopedN("QWEN_LLK_SYN_FIRST");  // qwen-llk-zone\n    // ---- first stage', result)
        self.assertIn('    }  // qwen-llk-zone QWEN_LLK_SYN_FIRST\n    { DeviceZoneScopedN("QWEN_LLK_SYN_SECOND")', result)

    def test_bytes_that_are_not_utf8_survive(self):
        data = b'// \xa9 2024 \xff\n#include <cstdint>\nvoid kernel_main() {\n    cb_wait_front(0, 1);\n}\n'
        text = zones.decode(data)
        result, record = zones.instrument(text, 'B', 'stages')
        self.assertEqual(zones.encode(zones.remove(result)), data)
        self.assertEqual(record['source_sha256'], __import__('hashlib').sha256(data).hexdigest())

    def test_already_instrumented_and_crlf_are_refused(self):
        result, _ = zones.instrument(SYNTHETIC, 'SYN', 'tag')
        with self.assertRaisesRegex(zones.ZoneError, 'already instrumented'):
            zones.instrument(result, 'SYN', 'tag')
        with self.assertRaisesRegex(zones.ZoneError, 'LF-only'):
            zones.instrument(SYNTHETIC.replace('\n', '\r\n'), 'SYN', 'tag')


class SyncTests(unittest.TestCase):
    def test_slots_follow_the_thread_meaning(self):
        result, counts, skipped = zones.add_sync(SYNTHETIC)
        wrapped_in = re.findall(r'\{ DeviceZoneScopedSumN1\("QWEN_LLK_WAIT_IN"\); ([^{}]*?;) \}', result)
        wrapped_out = re.findall(r'\{ DeviceZoneScopedSumN2\("QWEN_LLK_WAIT_OUT"\); ([^{}]*?;) \}', result)
        self.assertIn('CircularBuffer(cb).wait_front(n);', wrapped_in)
        self.assertIn('tile_regs_wait();', wrapped_in)
        self.assertIn('cb_wait_front(0, n);', wrapped_in)
        self.assertIn('cb_wait_front(1, n);', wrapped_in)
        self.assertIn('cb_wait_front(2, n);', wrapped_in)
        self.assertIn('cb_wait_front(3,\n                         n);', wrapped_in)
        self.assertIn('noc.async_read_barrier();', wrapped_in)
        self.assertIn('cb_reserve_back(o, 1);', wrapped_out)
        self.assertIn('tile_regs_acquire();', wrapped_out)
        self.assertIn('out->reserve_back(4);', wrapped_out)
        self.assertEqual(counts, {1: len(wrapped_in), 2: len(wrapped_out)})

    def test_comments_strings_and_macros_are_left_alone(self):
        result, _, _ = zones.add_sync(SYNTHETIC)
        self.assertIn('// A comment that names cb_wait_front(cb, 1); and must stay untouched.', result)
        self.assertIn('/* block comment: tile_regs_acquire(); */', result)
        self.assertIn('#define WAIT_ONE(cb) cb_wait_front(cb, 1);', result)
        self.assertIn('const char* label = "cb_wait_front(x, 1);";', result)

    def test_ambiguous_sites_are_listed_not_wrapped(self):
        result, _, skipped = zones.add_sync(SYNTHETIC)
        self.assertIn('ckernel::cb_wait_front(5, n);', result)
        reasons = dict((call, reason) for _, call, reason in skipped)
        self.assertEqual(reasons.get('cb_wait_front(5, n);'), 'no unambiguous statement boundary before it')
        text = 'void kernel_main() {\n    cb_wait_front(zone, 1);\n}\n'
        _, _, skipped = zones.add_sync(text)
        self.assertEqual([reason for _, _, reason in skipped], ['names hash or zone'])

    def test_after_assignment_on_the_same_line_is_a_boundary(self):
        result, _, _ = zones.add_sync(SYNTHETIC)
        self.assertIn('x = y; { DeviceZoneScopedSumN1("QWEN_LLK_WAIT_IN"); cb_wait_front(3,', result)

    def test_sums_off_leaves_calls_bare(self):
        result, record = zones.instrument(SYNTHETIC, 'SYN', 'stages', sums_supported=False)
        self.assertNotIn('DeviceZoneScopedSum', result)
        self.assertFalse(record['sync']['enabled'])
        result, record = zones.instrument(SYNTHETIC, 'SYN', 'tag')
        self.assertNotIn('DeviceZoneScopedSum', result)
        self.assertEqual([zone['kind'] for zone in record['zones']], ['envelope'])


class EnvelopeTests(unittest.TestCase):
    def test_the_whole_body_is_one_block(self):
        result = zones.add_envelope(SYNTHETIC, 'QWEN_LLK_SYN')
        self.assertIn('void kernel_main() {\n    { DeviceZoneScopedN("QWEN_LLK_SYN");  // qwen-llk-zone envelope\n',
                      result)
        self.assertTrue(result.endswith('    // ---- end\n}  // qwen-llk-zone envelope QWEN_LLK_SYN\n}\n'))

    def test_main_style_entry_point(self):
        text = 'namespace NAMESPACE {\nvoid MAIN {\n    cb_wait_front(0, 1);\n}\n}  // namespace NAMESPACE\n'
        result, _ = zones.instrument(text, 'OLD', 'stages')
        self.assertIn('void MAIN {\n    { DeviceZoneScopedN("QWEN_LLK_OLD");', result)
        self.assertEqual(zones.remove(result), text)

    def test_refusals(self):
        with self.assertRaisesRegex(zones.ZoneError, 'exactly one kernel entry point'):
            zones.add_envelope('void helper() {}\n', 'QWEN_LLK_X')
        with self.assertRaisesRegex(zones.ZoneError, 'exactly one kernel entry point'):
            zones.add_envelope('void kernel_main() {}\nvoid kernel_main() {}\n', 'QWEN_LLK_X')
        with self.assertRaisesRegex(zones.ZoneError, 'names hash'):
            zones.add_envelope('void kernel_main() {\n    uint32_t hash = 1;\n}\n', 'QWEN_LLK_X')
        # A commented-out entry point does not count.
        self.assertIn('QWEN_LLK_X', zones.add_envelope('// void kernel_main() {\nvoid kernel_main() {\n}\n', 'QWEN_LLK_X'))


class RegionTests(unittest.TestCase):
    def region(self, body, begin='    // begin', end='    // end'):
        text = 'void kernel_main() {\n    int keep = 0;\n%s\n%s\n%s\n}\n' % (begin, body, end)
        return zones.add_region(text, 'QWEN_LLK_R', begin, end)

    def test_refusals_name_the_reason(self):
        cases = (('    uint32_t x = 1;', 'declares at its own depth'),
                 ('    const bool more = true;', 'declares at its own depth'),
                 ('    Foo bar;', 'declares at its own depth'),
                 ('    if (keep) {', 'leaves 1 braces open'),
                 ('    }', 'closes a brace it did not open'),
                 ('#if FOO', '#if/#endif unbalanced'),
                 ('    case 1: keep = 2;', 'carries a label'),
                 ('    keep = zone;', 'names zone'))
        for body, reason in cases:
            with self.subTest(body=body):
                with self.assertRaisesRegex(zones.ZoneError, reason):
                    self.region(body)

    def test_declarations_inside_nested_blocks_are_fine(self):
        result = self.region('    for (uint32_t i = 0; i < 4; ++i) {\n        uint32_t y = i;\n    }\n#if FOO\n    keep = 1;\n#endif')
        self.assertIn('{ DeviceZoneScopedN("QWEN_LLK_R");', result)

    def test_anchor_rules(self):
        text = 'void kernel_main() {\n    // a\n    x();\n    // a\n}\n'
        with self.assertRaisesRegex(zones.ZoneError, 'exactly once'):
            zones.add_region(text, 'QWEN_LLK_R', '    // a', '}')
        text = 'void kernel_main() {\n    // b\n    x(); // c\n}\n'
        with self.assertRaisesRegex(zones.ZoneError, 'must start a line'):
            zones.add_region(text, 'QWEN_LLK_R', '    // b', '// c')
        with self.assertRaisesRegex(zones.ZoneError, 'precedes'):
            zones.add_region('void kernel_main() {\n    // e\n    // b\n}\n', 'QWEN_LLK_R', '    // b', '    // e')

    def test_end_of_entry(self):
        text = 'void kernel_main() {\n    // epi\n    x();\n}\n'
        result = zones.add_region(text, 'QWEN_LLK_E', '    // epi', None)
        self.assertEqual(result, 'void kernel_main() {\n    { DeviceZoneScopedN("QWEN_LLK_E");  // qwen-llk-zone\n'
                                 '    // epi\n    x();\n    }  // qwen-llk-zone QWEN_LLK_E\n}\n')
        self.assertEqual(zones.remove(result), text)


class HeaderTests(unittest.TestCase):
    def test_after_the_first_unconditional_include(self):
        result = zones.add_header(SYNTHETIC)
        self.assertIn('#include <cstdint>\n' + zones.HEADER_LINE, result)
        text = '/*\n#include "in_comment.h"\n*/\n#if X\n#include "a.h"\n#endif\n#include "b.h"\nint x;\n'
        self.assertIn('#include "b.h"\n' + zones.HEADER_LINE, zones.add_header(text))
        self.assertTrue(zones.add_header('int x;\n').startswith(zones.HEADER_LINE))
        self.assertIn('#pragma once\n' + zones.HEADER_LINE, zones.add_header('#pragma once\nint x;\n'))


class BudgetTests(unittest.TestCase):
    def test_marker_count(self):
        self.assertEqual(zones.marker_count([('A', 16), ('B', 1)]), 2 + 32 + 2 + 2)
        self.assertEqual(zones.marker_count([], envelope=True, sums=False), 2)

    def test_over_budget_is_refused(self):
        heavy = (('FIRST', '    // ---- first stage', '    // ---- second stage', 200),)
        with self.assertRaisesRegex(zones.ZoneError, 'past the 218 the lint allows'):
            zones.instrument(SYNTHETIC, 'SYN', 'stages', regions=heavy)
        # The same regions are fine at level tag, which keeps only the envelope.
        zones.instrument(SYNTHETIC, 'SYN', 'tag', regions=heavy)

    def test_zone_names(self):
        self.assertEqual(zones.zone_name('K5A', 'T1'), 'QWEN_LLK_K5A_T1')
        for bad in (('k5a',), ('K5A', 'T-1'), ('X' * 60,)):
            with self.assertRaises(zones.ZoneError):
                zones.zone_name(*bad)


class ReconfigTests(unittest.TestCase):
    def test_static_counts_follow_same_file_helpers(self):
        # helper: pack_reconfig_data_format + copy_tile_to_dst_init_short = 2 per call.
        body_start = SYNTHETIC.index('    // ---- first stage')
        body_end = SYNTHETIC.index('    // ---- second stage')
        # first stage: helper (2) + reconfig_data_format (1) + helper in the loop (2) = 5.
        self.assertEqual(zones.reconfig_calls(SYNTHETIC, body_start, body_end), 5)
        _, record = zones.instrument(SYNTHETIC, 'SYN', 'stages', regions=STAGES)
        self.assertEqual([zone['reconfig_static'] for zone in record['zones']], [5, 5, 0])


class SyntaxTests(unittest.TestCase):
    def test_module_parses_as_python_37(self):
        with open(os.path.join(HERE, 'llk_zones.py'), encoding='utf-8') as handle:
            ast.parse(handle.read(), 'llk_zones.py', feature_version=(3, 7))


if __name__ == '__main__':
    unittest.main()
