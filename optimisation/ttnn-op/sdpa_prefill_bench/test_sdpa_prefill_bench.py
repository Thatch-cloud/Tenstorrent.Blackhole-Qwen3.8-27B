"""CPU tests for the M1 chunked-SDPA bench helpers: shapes, the arm table, the slope fit and the
verdict reading rules. No ttnn needed."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sdpa_prefill_bench as bench  # noqa: E402


def results_for(slopes_ms_per_1k, intercept=5.0):
    """{arm: {start: ms}} lying exactly on the given per-call slopes."""
    return {name: {str(start): intercept + slope * start / 1000.0 for start in bench.STARTS}
            for name, slope in slopes_ms_per_1k.items()}


class ArmTableTests(unittest.TestCase):
    def test_the_seven_arms_move_one_knob_each_from_the_served_baseline(self):
        self.assertEqual(bench.ARM_NAMES, ('baseline', 'bf16_kv', 'bf16_qkv', 'exp_approx', 'fp32_off',
                                           'q256_2048', 'q256_4096'))
        base = bench.arm_by_name('baseline')
        self.assertEqual(base, dict(name='baseline', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8',
                                    kv_dtype='bf8', exp_approx=False, fp32_dest=True))
        expected_changes = dict(bf16_kv={'kv_dtype'}, bf16_qkv={'q_dtype', 'kv_dtype'}, exp_approx={'exp_approx'},
                                fp32_off={'fp32_dest'}, q256_2048={'q_chunk'}, q256_4096={'q_chunk', 'rows'})
        for name, keys in expected_changes.items():
            with self.subTest(arm=name):
                arm = bench.arm_by_name(name)
                changed = {key for key in base if key != 'name' and arm[key] != base[key]}
                self.assertEqual(changed, keys)

    def test_unknown_arms_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'unknown arm'):
            bench.arm_by_name('bf4')

    def test_arm_by_name_returns_a_copy(self):
        bench.arm_by_name('baseline')['rows'] = 1
        self.assertEqual(bench.arm_by_name('baseline')['rows'], 2048)


class GeometryTests(unittest.TestCase):
    def test_shapes_match_the_served_tp2_call(self):
        self.assertEqual(bench.q_shape(bench.arm_by_name('baseline')), (1, 12, 2048, 256))
        self.assertEqual(bench.q_shape(bench.arm_by_name('q256_4096')), (1, 12, 4096, 256))
        self.assertEqual(bench.kv_shape(2080), (2080, 2, 64, 256))
        self.assertEqual(bench.GRID, (11, 10))

    def test_sweep_starts_are_0_32k_64k_126k(self):
        self.assertEqual(bench.STARTS, (0, 32 * 1024, 64 * 1024, 126 * 1024))

    def test_the_pool_covers_the_largest_start_plus_rows_padded_to_32_blocks(self):
        arms = [bench.arm_by_name(name) for name in bench.ARM_NAMES]
        self.assertEqual(bench.pool_blocks(arms, bench.STARTS), 2080)          # 129024 + 4096 -> 2080
        self.assertEqual(bench.pool_blocks(arms[:1], bench.STARTS), 2048)     # 129024 + 2048 = 131072
        self.assertEqual(bench.blocks_for(131328), 2080)                       # the served 131k page table

    def test_busy_cores_follow_the_pair_distribution(self):
        self.assertEqual(bench.busy_cores(bench.arm_by_name('baseline')), 96)
        self.assertEqual(bench.busy_cores(bench.arm_by_name('q256_2048')), 48)
        self.assertEqual(bench.busy_cores(bench.arm_by_name('q256_4096')), 96)
        odd = dict(bench.arm_by_name('baseline'), rows=384)   # 3 q chunks: no pairing
        self.assertEqual(bench.busy_cores(odd), 36)

    def test_bf16_reads_1_88x_the_bytes_and_q256_4096_half_per_token(self):
        base = bench.arm_by_name('baseline')
        start = 129024
        ratio = bench.kv_bytes(bench.arm_by_name('bf16_kv'), start) / bench.kv_bytes(base, start)
        self.assertAlmostEqual(ratio, 2 * 1024 / 1088)
        self.assertAlmostEqual(ratio, bench.EXPECTED_BF16_RATIO)
        prefix_bytes = lambda arm: bench.kv_bytes(arm, start) - bench.kv_bytes(arm, 0)
        per_token = (prefix_bytes(bench.arm_by_name('q256_4096')) / 4096) / (prefix_bytes(base) / 2048)
        self.assertAlmostEqual(per_token, 0.5)

    def test_validate_enforces_the_flexible_path_preconditions(self):
        for name in bench.ARM_NAMES:
            bench.validate(bench.arm_by_name(name), bench.STARTS)
        with self.assertRaises(ValueError):
            bench.validate(bench.arm_by_name('q256_2048'), [128])       # not a multiple of q_chunk 256
        with self.assertRaises(ValueError):
            bench.validate(bench.arm_by_name('baseline'), [-128])
        with self.assertRaises(ValueError):
            bench.validate(dict(bench.arm_by_name('baseline'), rows=2000), [0])

    def test_schedule_interleaves_and_rotates_the_arm_order(self):
        arms = [bench.arm_by_name(name) for name in ('baseline', 'bf16_kv', 'exp_approx')]
        order = bench.schedule(arms, [0, 32768], 3)
        self.assertEqual(len(order), 3 * 3 * 2)
        self.assertEqual([order[0][0], order[6][0], order[12][0]], ['baseline', 'bf16_kv', 'exp_approx'])
        for key in {(a['name'], s) for a in arms for s in (0, 32768)}:
            self.assertEqual(order.count(key), 3)


class SlopeTests(unittest.TestCase):
    def test_fit_recovers_an_exact_line(self):
        line = bench.fit([(0, 5.0), (32768, 5.0 + 0.7 * 32.768), (129024, 5.0 + 0.7 * 129.024)])
        self.assertAlmostEqual(line['slope_ms_per_1k_keys'], 0.7)
        self.assertAlmostEqual(line['intercept_ms'], 5.0)
        self.assertEqual(line['points'], 3)

    def test_fit_needs_two_distinct_starts(self):
        self.assertIsNone(bench.fit([(0, 1.0)]))
        self.assertIsNone(bench.fit([(0, 1.0), (0, 2.0)]))
        self.assertIsNone(bench.fit([(0, 1.0), (32768, None)]))

    def test_the_4096_row_arm_is_also_given_per_2048_rows(self):
        table = bench.slopes(results_for(dict(baseline=0.4, q256_4096=0.4)))
        self.assertAlmostEqual(table['q256_4096']['slope_ms_per_1k_keys'], 0.4)
        self.assertAlmostEqual(table['q256_4096']['slope_ms_per_1k_keys_per_2048_rows'], 0.2)
        self.assertAlmostEqual(table['baseline']['slope_ms_per_1k_keys_per_2048_rows'], 0.4)

    def test_an_arm_with_failed_starts_has_no_slope(self):
        results = results_for(dict(baseline=0.4))
        results['bf16_kv'] = {str(s): None for s in bench.STARTS}
        self.assertIsNone(bench.slopes(results)['bf16_kv'])


class VerdictTests(unittest.TestCase):
    def verdict(self, **slopes):
        return bench.verdict(bench.slopes(results_for(slopes)))

    def test_bytes_bound(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.4 * 1.85, bf16_qkv=0.4 * 1.9, exp_approx=0.39,
                              fp32_off=0.4, q256_2048=0.45, q256_4096=0.4 * 1.05)
        self.assertEqual(result['verdict'], 'bytes-bound')
        self.assertIn('lever #1', result['action'])
        self.assertAlmostEqual(result['ratios']['q256_4096_per_token'], 0.525)
        self.assertEqual(result['q256_2048_preregistration'], 'confirmed')

    def test_compute_bound(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.42, exp_approx=0.3, fp32_off=0.39, q256_2048=0.5,
                              q256_4096=0.8)
        self.assertEqual(result['verdict'], 'compute-bound')
        self.assertIn('#1b', result['action'])

    def test_bytes_insensitive_without_a_compute_knob_moving_is_mixed(self):
        self.assertEqual(self.verdict(baseline=0.4, bf16_kv=0.41, exp_approx=0.4, fp32_off=0.4)['verdict'], 'mixed')

    def test_bf16_sensitive_but_4096_not_halving_is_mixed(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.7, q256_4096=0.7)
        self.assertEqual(result['verdict'], 'mixed')

    def test_bf16_qkv_stands_in_when_bf16_kv_failed(self):
        result = self.verdict(baseline=0.4, bf16_qkv=0.75, q256_4096=0.44)
        self.assertEqual(result['verdict'], 'bytes-bound')
        self.assertEqual(result['ratios']['bf16_source'], 'bf16_qkv')

    def test_no_baseline_is_incomplete(self):
        self.assertEqual(self.verdict(bf16_kv=0.7)['verdict'], 'incomplete')

    def test_the_q256_2048_preregistration_can_be_refuted(self):
        result = self.verdict(baseline=0.4, bf16_kv=0.75, q256_2048=0.3, q256_4096=0.44)
        self.assertEqual(result['q256_2048_preregistration'], 'refuted')

    def test_the_verdict_line_is_one_parseable_line(self):
        line = bench.verdict_line(self.verdict(baseline=0.4, bf16_kv=0.75, q256_4096=0.44))
        self.assertTrue(line.startswith('M1 VERDICT: bytes-bound | bf16/bf8=1.875 (bf16_kv)'))
        self.assertNotIn(chr(10), line)
        self.assertIn('exp_approx=n/a', line)


class MainTests(unittest.TestCase):
    def test_without_ttnn_main_writes_an_error_report_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'm1.json'
            from contextlib import redirect_stdout
            import io
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = bench.main(['--out', str(out), '--arms', 'baseline', '--rounds', '1'])
            self.assertEqual(status, 1)
            report = json.loads(out.read_text(encoding='utf-8'))
        self.assertFalse(report['passed'])
        self.assertIn('Error', report['error'])
        self.assertIn('M1 VERDICT: incomplete', buffer.getvalue())

    def test_bad_arm_names_are_refused_at_parse_time(self):
        with self.assertRaises(ValueError):
            bench.main(['--out', 'unused.json', '--arms', 'nope'])

    def test_the_table_prints_every_timed_arm(self):
        results = results_for(dict(baseline=0.4, bf16_kv=0.75))
        results['bf16_kv'][str(bench.STARTS[-1])] = None
        text = bench.format_table(results, bench.slopes(results), list(bench.STARTS))
        self.assertIn('baseline', text)
        self.assertIn('ERR', text)
        self.assertEqual(len(text.splitlines()), 3)


class RunnerTests(unittest.TestCase):
    """run_m1.sh: card M only, never a shared card, a fresh kernel cache, the 131k serving image."""

    def setUp(self):
        self.text = (Path(__file__).parent / 'run_m1.sh').read_text(encoding='utf-8')

    def test_card_m_only_and_refuses_a_held_device(self):
        self.assertIn('CARD_M=/dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D', self.text)
        self.assertIn('--device "$node"', self.text)
        self.assertEqual(self.text.count('--device '), 1)
        self.assertIn('holds a Tenstorrent device', self.text)
        self.assertLess(self.text.index('holds a Tenstorrent device'), self.text.index('### M1 $stamp'))

    def test_fresh_kernel_cache_and_the_pinned_image(self):
        self.assertIn('kcache-m1-$stamp,dst=/kcache', self.text)
        self.assertIn('-e TT_METAL_CACHE=/kcache', self.text)
        self.assertIn('sha256:0648ca9ad663acc72e7d8ea59d9cde0f9218b583ad74a60d58ff2f31bddd6fae', self.text)

    def test_the_bench_is_mounted_and_the_verdict_surfaced(self):
        self.assertIn('src=$S/sdpa_prefill_bench.py,dst=/bench/sdpa_prefill_bench.py,readonly', self.text)
        self.assertIn('/bench/sdpa_prefill_bench.py --out', self.text)
        self.assertIn("grep -F 'M1 VERDICT:'", self.text)
        self.assertNotIn(chr(13), self.text)


if __name__ == '__main__':
    unittest.main()
