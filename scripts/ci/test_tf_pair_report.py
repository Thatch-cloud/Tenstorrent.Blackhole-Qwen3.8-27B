"""tf_pair_report on golden synthetic data with a known R: the paired bootstrap, the guardrails, the verdict boundaries, the
censored-kill exception, the private / public split."""
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tau_lab_report as rep  # noqa: E402
import tf_pair_report as pr  # noqa: E402
import tf_pair_walk as walk  # noqa: E402

PASS_ALL = dict((gate, 'PASS') for gate in pr.VALIDITY_GATES)
SMALL = dict(resamples=300, w8_resamples=40, w8_draws=200)


def fake_round(offset, committed, region_think=10):
    return dict(offset=offset, start=1000 + offset, rows=16, cap=16, accepted=committed - 1, available=15,
                committed=committed, uncapped=committed, region='reasoning' if offset < region_think else 'content')


def rounds_of(committed_values, first_offset=1):
    out, offset = [], first_offset
    for value in committed_values:
        out.append(fake_round(offset, value))
        offset += value
    return out


def make_data(ratio=1.2, clusters_per_set=(12, 8), turns_per_cluster=3, noise=0.0, seed=3, difficulty=True, rounds=12,
              long_every=0, own_ratio=None):
    """meta, S arm, D arm. A cluster has a difficulty (a tau level common to both arms); S = ratio * D."""
    rng = random.Random(seed)
    meta, arm_s, arm_d, k = {}, {}, {}, 0
    for set_name, count in zip(pr.SETS, clusters_per_set):
        factor = ratio if (own_ratio is None or set_name == 'swe') else own_ratio
        for cluster in range(count):
            level = rng.uniform(3.0, 8.0) if difficulty else 5.0
            for turn in range(turns_per_cluster):
                bucket = '80k' if long_every and k % long_every == 0 else '8k'
                meta[k] = dict(k=k, set=set_name, cluster=cluster, bucket=bucket, weight=1.0, think_tokens=10, tool_at=None)
                d_values = [max(1, int(round(level + rng.uniform(-noise, noise)))) for _ in range(rounds)]
                s_values = [max(1, int(round(value * factor))) for value in d_values]
                arm_d[k], arm_s[k] = rounds_of(d_values), rounds_of(s_values)
                k += 1
    return meta, {pr.ARM_S: arm_s, pr.ARM_D: arm_d}


class StatisticTests(unittest.TestCase):
    def test_known_ratio_is_recovered_exactly_when_the_data_are_exact(self):
        meta, arms = make_data(ratio=1.2, difficulty=False)
        private, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertAlmostEqual(public['r']['point'], 1.2, places=3)
        self.assertAlmostEqual(public['r']['low'], 1.2, places=3)
        self.assertAlmostEqual(public['r']['high'], 1.2, places=3)

    def test_interval_covers_a_known_ratio_with_noise(self):
        meta, arms = make_data(ratio=1.2, noise=1.5)
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertLessEqual(public['r']['low'], public['r']['point'])
        self.assertGreaterEqual(public['r']['high'], public['r']['point'])
        self.assertLess(abs(public['r']['point'] - 1.2), 0.08)

    def test_pairing_is_preserved_and_unpaired_resampling_widens_the_interval(self):
        meta, arms = make_data(ratio=1.2, noise=0.0, difficulty=True)
        keys = pr.paired_keys(meta, arms[pr.ARM_S], arms[pr.ARM_D])
        units = pr.units_by_set(meta, keys, arms[pr.ARM_S], arms[pr.ARM_D])
        paired = pr.cluster_bootstrap(units, pr.ratio_statistic(), 400, seed=1)
        unpaired = pr.cluster_bootstrap(units, pr.ratio_statistic(), 400, seed=1, paired=False)
        paired_width = paired['high'] - paired['low']
        unpaired_width = unpaired['high'] - unpaired['low']
        self.assertGreater(unpaired_width, 3 * paired_width)

    def test_weights_enter_the_tau(self):
        meta, arms = make_data(ratio=1.0, difficulty=False, clusters_per_set=(4, 4), turns_per_cluster=1)
        # one swe cluster gets triple weight and a much better S: the weighted R moves
        for k, info in meta.items():
            if info['set'] == 'swe' and info['cluster'] == 0:
                info['weight'] = 30.0
                arms[pr.ARM_S][k] = rounds_of([10] * 12)
        _, weighted = pr.build(meta, arms, PASS_ALL, **SMALL)
        for info in meta.values():
            info['weight'] = 1.0
        _, flat = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertGreater(weighted['r']['point'], flat['r']['point'] + 0.01)

    def test_sets_count_equally(self):
        meta, arms = make_data(ratio=1.0, difficulty=False, clusters_per_set=(30, 2), turns_per_cluster=1, own_ratio=2.0)
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        # equal set weight: R is near (1.0 + 2.0) / 2 in tau terms, not near 1.0 as 30:2 clusters would give
        self.assertGreater(public['r']['point'], 1.3)

    def test_uncapped_beside_capped(self):
        meta, arms = make_data(ratio=1.2, difficulty=False)
        for turns in arms.values():
            for rounds in turns.values():
                for entry in rounds:
                    entry['cap'] = 2
                    entry['committed'] = min(entry['committed'], 2)
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertAlmostEqual(public['r']['point'], 1.0, places=3)
        self.assertGreater(public['r']['uncapped_point'], 1.1)


class GuardrailTests(unittest.TestCase):
    def test_g_long_looks_only_at_the_long_buckets(self):
        meta, arms = make_data(ratio=1.2, difficulty=False, long_every=4, clusters_per_set=(10, 10), turns_per_cluster=2)
        for k, info in meta.items():          # S loses on the long turns only
            if info['bucket'] == '80k':
                arms[pr.ARM_S][k] = rounds_of([3] * 12)
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertEqual(public['g_long']['status'], 'FAIL')
        self.assertLess(public['g_long']['point'], 1.0)
        self.assertGreater(public['r']['point'], 1.0)

    def test_p10_needs_three_counted_rounds_in_both_arms(self):
        meta, arms = make_data(ratio=1.2, difficulty=False, clusters_per_set=(5, 5), turns_per_cluster=2)
        keys = sorted(meta)
        short = keys[0]
        arms[pr.ARM_S][short] = rounds_of([1, 1])
        units = pr.turn_units(meta, keys, arms[pr.ARM_S], arms[pr.ARM_D])
        count = sum(len(cluster) for cells in units.values() for cluster in cells)
        self.assertEqual(count, len(keys) - 1)

    def test_p10_and_w8_pass_for_a_uniform_win_and_fail_for_a_tail_loss(self):
        meta, arms = make_data(ratio=1.3, difficulty=True, noise=1.0)
        _, win = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertEqual(win['g_p10']['status'], 'PASS')
        self.assertEqual(win['g_w8']['status'], 'PASS')
        # the worst turns collapse under S: the tails fail while the mean may still look fine
        ranked = sorted(meta, key=lambda k: sum(e['committed'] for e in arms[pr.ARM_D][k]))
        for k in ranked[:len(ranked) // 4]:
            arms[pr.ARM_S][k] = rounds_of([1] * 12)
        _, lose = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertEqual(lose['g_p10']['status'], 'FAIL')
        self.assertEqual(lose['g_w8']['status'], 'FAIL')

    def test_worst_of_k_is_paired_on_the_same_turns(self):
        units = {'swe': [[(5.0, 5.0, 1.0, 10), (2.0, 2.5, 1.0, 10)]]}
        value = pr.worst_of_k_ratio(draws=400)(units, random.Random(2))
        self.assertAlmostEqual(value, 0.8, places=3)    # the minimum of 8 draws is the 2.0 / 2.5 turn in both arms


class VerdictTests(unittest.TestCase):
    def entry(self, point, low, high, status='PASS'):
        return dict(point=point, low=low, high=high, status=status)

    def decide(self, r, long_status='PASS', p10='PASS', w8='PASS', own=1.2, bands=None, validity=None):
        return pr.decide(r, self.entry(1.1, 1.0, 1.2, long_status), self.entry(1.0, 1.0, 1.0, p10),
                         self.entry(1.0, 1.0, 1.0, w8), self.entry(own, own, own),
                         bands or {'0-511': 1.1, '512-2047': 1.1}, PASS_ALL if validity is None else validity)

    def test_go(self):
        self.assertEqual(self.decide(self.entry(1.12, 1.06, 1.18))['verdict'], 'GO')

    def test_go_boundary_at_the_point_estimate(self):
        self.assertEqual(self.decide(self.entry(1.10, 1.06, 1.18))['verdict'], 'GO')
        out = self.decide(self.entry(1.0999, 1.06, 1.18))
        self.assertEqual((out['verdict'], out['reasons']), ('HOLD', ['r_below_1_10']))

    def test_kill_when_the_upper_bound_is_under_1_05(self):
        self.assertEqual(self.decide(self.entry(1.02, 0.98, 1.0499))['verdict'], 'KILL')
        self.assertEqual(self.decide(self.entry(1.02, 0.98, 1.05))['verdict'], 'HOLD')

    def test_censored_kill_exception_asks_for_arm_e(self):
        out = self.decide(self.entry(1.0, 0.95, 1.04), bands={'0-511': 0.98, '512-2047': 1.03})
        self.assertEqual((out['verdict'], out['run_arm_e']), ('HOLD', True))
        out = self.decide(self.entry(1.0, 0.95, 1.04), bands={'0-511': 0.98, '512-2047': 1.029})
        self.assertEqual((out['verdict'], out['run_arm_e']), ('KILL', False))

    def test_each_guardrail_alone_holds(self):
        r = self.entry(1.2, 1.1, 1.3)
        for kwargs, reason in (({'long_status': 'FAIL'}, 'g_long_failed'), ({'p10': 'FAIL'}, 'g_p10_failed'),
                               ({'w8': 'FAIL'}, 'g_w8_failed'), ({'own': 0.99}, 'r_own_below_1')):
            out = self.decide(r, **kwargs)
            self.assertEqual((out['verdict'], out['reasons']), ('HOLD', [reason]), kwargs)

    def test_a_failed_or_missing_validity_gate_is_not_established_never_a_kill(self):
        low = self.entry(0.9, 0.8, 0.95)       # would be a kill with valid gates
        for validity in (dict(PASS_ALL, V3='FAIL'), dict((g, s) for g, s in PASS_ALL.items() if g != 'V4')):
            out = self.decide(low, validity=validity)
            self.assertEqual(out['verdict'], 'NOT_ESTABLISHED')
        self.assertEqual(self.decide(low)['verdict'], 'KILL')

    def test_v2_is_report_only(self):
        validity = dict(PASS_ALL, V2='FAIL')
        self.assertEqual(self.decide(self.entry(1.2, 1.1, 1.3), validity=validity)['verdict'], 'GO')


class WindowAndSecondaryTests(unittest.TestCase):
    def test_window_retention_steers_phase_b(self):
        meta, arms = make_data(ratio=1.3, difficulty=False)
        arms['dspark-t16-w8192'] = dict((k, rounds_of([int(round(5 * 1.28))] * 12)) for k in meta)   # keeps ~93% of R - 1
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertEqual(public['window_steer'], 'bounded_window')
        arms['dspark-t16-w8192'] = dict((k, rounds_of([5] * 12)) for k in meta)                      # keeps none of it
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertEqual(public['window_steer'], 'full_window')

    def test_bands_regions_and_curves(self):
        meta, arms = make_data(ratio=1.2, difficulty=False, rounds=200)
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        secondary = public['secondary']
        self.assertEqual(set(secondary['r_by_band']), set(['0-511', '512-2047']))
        self.assertEqual(len(secondary['acceptance_dspark']), 15)
        self.assertGreater(secondary['win_share'], 0.99)
        self.assertGreater(secondary['miss_removed'], 0)

    def test_unpaired_turns_are_left_out_and_counted(self):
        meta, arms = make_data(ratio=1.2, difficulty=False, clusters_per_set=(4, 4), turns_per_cluster=1)
        del arms[pr.ARM_D][0]
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        self.assertEqual((public['turns_paired'], public['turns_unpaired']), (7, 1))


class PrivacyAndCliTests(unittest.TestCase):
    def test_public_summary_passes_assert_public_and_private_keeps_the_per_turn_rows(self):
        meta, arms = make_data()
        private, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        pr.assert_public(public)
        self.assertNotIn('per_turn', public)
        self.assertIn('per_turn', private)
        text = json.dumps(public)
        self.assertNotIn('"k"', text)

    def test_a_leaked_string_is_refused(self):
        meta, arms = make_data()
        _, public = pr.build(meta, arms, PASS_ALL, **SMALL)
        public['verdict'] = 'SENTINELTEXT'
        with self.assertRaises(rep.PrivacyError):
            pr.assert_public(public)
        public['verdict'] = 'GO'
        public['extra'] = [[1, 2]]
        with self.assertRaises(rep.PrivacyError):
            pr.assert_public(public)

    def test_cli_round_trip_and_exit_codes(self):
        meta, arms = make_data(clusters_per_set=(6, 4))
        root = tempfile.mkdtemp()
        try:
            results, out = os.path.join(root, 'results'), os.path.join(root, 'out')
            os.makedirs(results)
            with open(os.path.join(results, 'meta.jsonl'), 'w') as handle:
                for info in meta.values():
                    handle.write(json.dumps(info) + '\n')
            for name, turns in arms.items():
                with open(os.path.join(results, 'arm-%s.jsonl' % name), 'w') as handle:
                    for k, rounds in turns.items():
                        handle.write(json.dumps(dict(k=k, status='ok', rounds=[walk.to_row(entry) for entry in rounds])) + '\n')
            lines = []
            code = pr.main(['--results', results, '--out', out, '--resamples', '100', '--w8-resamples', '10',
                            '--w8-draws', '100'], say=lines.append)
            self.assertEqual(code, 2)                         # no validity.json: NOT_ESTABLISHED
            with open(os.path.join(results, 'validity.json'), 'w') as handle:
                json.dump(PASS_ALL, handle)
            code = pr.main(['--results', results, '--out', out, '--resamples', '100', '--w8-resamples', '10',
                            '--w8-draws', '100'], say=lines.append)
            self.assertEqual(code, 0)
            with open(os.path.join(out, 'a0-summary.json')) as handle:
                summary = json.load(handle)
            self.assertIn(summary['verdict'], ('GO', 'HOLD', 'KILL'))
            self.assertTrue(os.path.exists(os.path.join(results, 'report.private.json')))
            self.assertFalse(os.path.exists(os.path.join(out, 'report.private.json')))
        finally:
            shutil.rmtree(root)

    def test_failed_turns_in_an_arm_file_are_skipped(self):
        meta, arms = make_data(clusters_per_set=(3, 3), turns_per_cluster=1)
        root = tempfile.mkdtemp()
        try:
            with open(os.path.join(root, 'meta.jsonl'), 'w') as handle:
                for info in meta.values():
                    handle.write(json.dumps(info) + '\n')
            with open(os.path.join(root, 'arm-%s.jsonl' % pr.ARM_S), 'w') as handle:
                for k, rounds in arms[pr.ARM_S].items():
                    status = 'ok' if k else 'failed'
                    handle.write(json.dumps(dict(k=k, status=status, rounds=[walk.to_row(e) for e in rounds])) + '\n')
            loaded = pr.load(root)[1]
            self.assertNotIn(0, loaded[pr.ARM_S])
            self.assertEqual(len(loaded[pr.ARM_S]), len(meta) - 1)
        finally:
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
