"""tp4/drafter-bf16: QWEN_FAST_DRAFTER_BF16 keeps the DFlash2 drafter's projections bfloat16, default off.

The decision point is draft_mlp_branch.draft_projection_dtype (every one of the 36 projection uploads asks it). Off, the dtype is QWEN_FAST_DRAFT_BF8's, exactly
as before; on, bfloat16 whatever DRAFT_BF8 says, with one engaged marker per process. The profiles c2-packed-tp4-best-dbf16 and c2-packed-tp4-best-gate-dbf16 are
their bases plus that flag alone; the job pack parses; the cost arithmetic and the paired report are checked on synthetic input."""

import contextlib
import io
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import draft_mlp_branch  # noqa: E402
import c2_smoke_check  # noqa: E402
import drafter_bf16  # noqa: E402
import lever_n_m3native_gate  # noqa: E402
from draft_attention_branch import prepare_attention_branch  # noqa: E402
from draft_mlp_branch import DRAFT_BF8_FLAG, DRAFTER_BF16_FLAG, draft_projection_dtype, prepare_mlp_branch  # noqa: E402
from test_draft_projection_bf8 import FakeOperations, attention_weights, convolution_weights, mlp_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-dbf16-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
STRACE, BEST_DBF16 = 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-dbf16'
GATE, GATE_DBF16 = 'c2-packed-tp4-best-gate', 'c2-packed-tp4-best-gate-dbf16'
EIGHT, EIGHT_DBF16 = 'c2-packed-tp4-8-best-quad', 'c2-packed-tp4-8-best-quad-dbf16'
IMAGE = 'tp4-dbf16-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
EXPECTED = {
    'X0-status-rescan': ('status rescan', 'c2-packed-tp4'), 'B0-build': ('build', 'c2-packed-tp4'),
    'S1-dbf16-audited-smoke': ('rescan reset smoke', GATE_DBF16),
    'D1-timed-A-best-strace': ('reset smoke', STRACE), 'D2-timed-B-best-dbf16': ('reset smoke', BEST_DBF16),
    'D3-timed-A-best-strace': ('reset smoke', STRACE), 'D4-timed-B-best-dbf16': ('reset smoke', BEST_DBF16),
    'E1-eight-seat-A-best-quad': ('reset smoke', EIGHT), 'E2-eight-seat-B-best-quad-dbf16': ('reset smoke', EIGHT_DBF16),
    'E3-eight-seat-A-best-quad': ('reset smoke', EIGHT), 'E4-eight-seat-B-best-quad-dbf16': ('reset smoke', EIGHT_DBF16),
    'Z-reset': ('status reset', 'c2-packed-tp4'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)


class FlagTests(unittest.TestCase):
    def setUp(self):
        draft_mlp_branch._announced.clear()

    def test_off_the_dtype_is_draft_bf8s_exactly(self):
        operations = FakeOperations()
        for bf8, other in ((None, 'bf16'), ('0', 'bf16'), ('1', 'bf8')):
            for flag in (None, '0', 'true', ''):
                with self.subTest(bf8=bf8, flag=flag):
                    environ = {k: v for k, v in ((DRAFT_BF8_FLAG, bf8), (DRAFTER_BF16_FLAG, flag)) if v is not None}
                    self.assertEqual(draft_projection_dtype(operations, environ), other)

    def test_only_the_value_one_engages_and_it_overrides_bf8(self):
        operations = FakeOperations()
        for bf8 in (None, '0', '1'):
            environ = {DRAFTER_BF16_FLAG: '1'}
            if bf8 is not None:
                environ[DRAFT_BF8_FLAG] = bf8
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(draft_projection_dtype(operations, environ), 'bf16', bf8)

    def test_the_marker_is_printed_once_and_only_when_engaged(self):
        operations = FakeOperations()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in range(36):
                draft_projection_dtype(operations, {DRAFT_BF8_FLAG: '1'})
        self.assertEqual(out.getvalue(), '')
        with contextlib.redirect_stdout(out):
            for _ in range(36):
                draft_projection_dtype(operations, {DRAFTER_BF16_FLAG: '1', DRAFT_BF8_FLAG: '1'})
        self.assertEqual(out.getvalue().count(draft_mlp_branch.ENGAGED_MARKER), 1)
        self.assertTrue(draft_mlp_branch.ENGAGED_MARKER.startswith('[DRAFTER_BF16] engaged'))

    def test_the_flag_reaches_every_projection_upload_under_a_baked_bf8(self):
        old = {name: os.environ.get(name) for name in (DRAFT_BF8_FLAG, DRAFTER_BF16_FLAG)}
        os.environ[DRAFT_BF8_FLAG], os.environ[DRAFTER_BF16_FLAG] = '1', '1'
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mlp = prepare_mlp_branch(FakeOperations(), 'mesh', mlp_weights(), convolution_weights(), lambda value: value)
                attention = prepare_attention_branch(FakeOperations(), 'mesh', attention_weights(), convolution_weights(),
                                                     lambda value: value, native_head_layout=True, block_rows=16)
            self.assertEqual([t.dtype for t in mlp['device_projections']], ['bf16'] * 3)
            self.assertEqual([t.dtype for t in attention['projections'].values()] + [attention['output_projection'].dtype], ['bf16'] * 4)
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class ProfileTests(unittest.TestCase):
    def test_the_file_default_is_still_production(self):
        self.assertEqual(profiles()['default'], 'c2-packed-tp4')

    def test_each_profile_is_its_base_plus_exactly_the_flag(self):
        found = profiles()['profiles']
        for name, base in ((BEST_DBF16, STRACE), (GATE_DBF16, GATE), (EIGHT_DBF16, EIGHT)):
            with self.subTest(profile=name):
                self.assertEqual(found[name]['env'], dict(found[base]['env'], **{DRAFTER_BF16_FLAG: '1'}))
                for key in set(found[name]) | set(found[base]):
                    if key not in ('env', 'description'):
                        self.assertEqual(found[name].get(key), found[base].get(key), key)
                self.assertTrue(found[name]['description'].startswith('GATE ONLY'))
                self.assertIs(found[name]['gate_only'], True)
                self.assertIsNone(BANNED.search(found[name]['description']))

    def test_the_flag_sits_only_in_those_two_profiles(self):
        found = profiles()['profiles']
        self.assertEqual(sorted(n for n, body in found.items() if DRAFTER_BF16_FLAG in body['env']), sorted([BEST_DBF16, GATE_DBF16, EIGHT_DBF16]))
        self.assertNotIn(DRAFTER_BF16_FLAG, found['c2-packed-tp4']['env'])

    def test_the_timed_pair_keeps_the_audits_off_and_the_audited_one_keeps_them(self):
        found = profiles()['profiles']
        self.assertEqual((found[BEST_DBF16]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], found[BEST_DBF16]['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
        self.assertEqual(found[BEST_DBF16]['env']['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
        self.assertEqual(found[GATE_DBF16]['env']['QWEN_FAST_DRAFT_SINGLES_AUDIT'], 'all')


class CostTests(unittest.TestCase):
    def test_the_projection_elements_are_the_drafters_36_tensors(self):
        per_layer = 4096 * 5120 + 2 * 1024 * 5120 + 5120 * 4096 + 3 * 17408 * 5120
        self.assertEqual(drafter_bf16.projection_elements(), 5 * per_layer + 5120 * 25600)
        self.assertEqual(drafter_bf16.projection_elements(), 1730150400)

    def test_tp4_bytes_and_the_extra_read(self):
        four = drafter_bf16.cost(4, passes_per_round=1, round_ms=100.0)
        self.assertEqual((four['per_chip_bf16_mb'], four['per_chip_bf8_mb'], four['extra_mb_per_chip']), (865.1, 459.6, 405.5))
        self.assertAlmostEqual(four['extra_ms_per_pass'], 0.965, places=3)
        self.assertAlmostEqual(four['extra_ms_per_pass_at_peak'], 0.792, places=3)
        self.assertAlmostEqual(four['break_even_tau_gain_pct'], 0.97, places=2)
        eight = drafter_bf16.cost(4, passes_per_round=2, round_ms=299.0)
        self.assertAlmostEqual(eight['extra_ms_per_round'], 1.931, places=3)
        self.assertAlmostEqual(eight['break_even_tau_gain_pct'], 0.65, places=2)

    def test_the_pair_costs_twice_the_per_chip_bytes(self):
        self.assertAlmostEqual(drafter_bf16.cost(2)['extra_mb_per_chip'], 2 * drafter_bf16.cost(4)['extra_mb_per_chip'], places=0)


def round_log(counts_per_round, seconds, start=0.0, users=4, gap_after=None):
    """A server log: per round one '[PHASE] execute ... new=0 cached=<users>' stamp and the users' '[PACKED] request=' lines. gap_after=(round index, seconds)
    inserts an idle gap after that round, which ends the episode."""
    lines, clock = [], start

    def stamp(at):
        total_ms = int(round(at * 1000))
        return '2026-10-03 10:%02d:%02d.%03d | INFO | [PHASE] execute total=%d new=0 cached=%d spec=60' % (
            total_ms // 60000, total_ms // 1000 % 60, total_ms % 1000, users, users)

    for index, counts in enumerate(counts_per_round):
        lines.append(stamp(clock))
        for user, emitted in enumerate(counts):
            lines.append('[PACKED] request=r%d segment=0 position=100 prefix=0 emitted=%d' % (user, emitted))
        clock += seconds
        if gap_after and gap_after[0] == index:
            lines.append('2026-10-03 10:%02d:%02d.%03d | INFO | [PHASE] execute total=1 new=500 cached=0 spec=60' % (
                int(clock * 1000) // 60000, int(clock * 1000) // 1000 % 60, int(clock * 1000) % 1000))
            clock += gap_after[1]
    lines.append(stamp(clock))
    return '\n'.join(lines) + '\n'


class CompareTests(unittest.TestCase):
    def test_tau_round_and_rate_per_arm_and_the_ratios(self):
        a = round_log([(6, 6, 6, 6)] * 10, 0.100)
        b = round_log([(7, 6, 7, 6)] * 10, 0.101)
        result = drafter_bf16.compare(a, b, 4)
        self.assertEqual((result['a']['tau'], result['b']['tau']), (6.0, 6.5))
        self.assertEqual((result['a']['mean_round_ms'], result['b']['mean_round_ms']), (100.0, 101.0))
        self.assertEqual((result['a']['per_user_tok_s'], result['b']['per_user_tok_s']), (60.0, 64.36))
        self.assertAlmostEqual(result['tau_ratio_b_over_a'], 1.0833, places=4)
        self.assertAlmostEqual(result['round_time_ratio_b_over_a'], 1.01, places=4)
        self.assertTrue(result['bf16_wins'])
        self.assertEqual((result['paired_rounds'], result['episodes']), (10, 1))
        self.assertEqual(result['rounds_b_faster'], 10)
        self.assertGreater(result['paired_rate_delta_mean_tok_s'], 0)
        self.assertEqual(result['per_round_committed']['b'][0], [7, 6, 7, 6])

    def test_a_tau_gain_below_the_extra_round_time_does_not_win(self):
        result = drafter_bf16.compare(round_log([(6, 6, 6, 6)] * 6, 0.100), round_log([(6, 6, 6, 6)] * 6, 0.101), 4)
        self.assertFalse(result['bf16_wins'])
        self.assertEqual(result['rounds_b_faster'], 0)

    def test_an_arm_without_timed_rounds_is_refused(self):
        with self.assertRaises(ValueError):
            drafter_bf16.compare('nothing\n', round_log([(6, 6, 6, 6)] * 3, 0.1), 4)

    def test_only_the_common_prefix_of_each_episode_is_paired(self):
        # A runs 6 rounds in its first test and 4 in its second; B runs 3 and 4 (the last round before a prefill has no timing). Rounds beyond B's first episode are not summarised.
        a = round_log([(5, 5, 5, 5)] * 3 + [(9, 9, 9, 9)] * 3 + [(6, 6, 6, 6)] * 4, 0.1, gap_after=(5, 5.0))
        b = round_log([(5, 5, 5, 5)] * 3 + [(6, 6, 6, 6)] * 4, 0.1, gap_after=(2, 5.0))
        result = drafter_bf16.compare(a, b, 4)
        self.assertEqual((result['episodes'], result['paired_rounds']), (2, 6))
        self.assertEqual(result['a']['tau'], round((5 * 2 + 6 * 4) / 6.0, 3))
        self.assertEqual(result['a']['tau'], result['b']['tau'])
        self.assertNotIn([9, 9, 9, 9], result['per_round_committed']['a'])

    def test_eight_seat_rounds_read_with_live_8(self):
        a = round_log([(4,) * 8] * 5, 0.3, users=8)
        b = round_log([(5,) * 8] * 5, 0.302, users=8)
        result = drafter_bf16.compare(a, b, 8)
        self.assertEqual((result['a']['tau'], result['b']['tau'], result['paired_rounds']), (4.0, 5.0, 5))
        with self.assertRaises(ValueError):
            drafter_bf16.compare(a, b, 4)


class SmokeRuleTests(unittest.TestCase):
    ON, OFF = {DRAFTER_BF16_FLAG: '1'}, {'QWEN_FAST_DRAFT_BF8': '1'}
    ENGAGED = draft_mlp_branch.ENGAGED_MARKER
    LEND_BF16 = '[PINDIAG] draft weights lent to quad (borrowers=1 tensors=36)'
    LEND_BF8 = '[PINDIAG] draft weights lent to quad (borrowers=1 tensors=36) projections dtype=bf8 x36'

    def test_on_needs_the_engaged_line_and_no_bf8_lend_line(self):
        self.assertEqual(c2_smoke_check.drafter_bf16_problems(self.ON, self.ENGAGED + '\n' + self.LEND_BF16 + '\n'), [])
        missing = c2_smoke_check.drafter_bf16_problems(self.ON, self.LEND_BF16 + '\n')
        self.assertEqual(len(missing), 1)
        self.assertIn('never took the flag', missing[0])
        bad = c2_smoke_check.drafter_bf16_problems(self.ON, self.ENGAGED + '\n' + self.LEND_BF8 + '\n')
        self.assertEqual(len(bad), 1)
        self.assertIn('still reports bf8', bad[0])

    def test_off_allows_a_bf8_lend_line_and_refuses_any_bf16_line(self):
        self.assertEqual(c2_smoke_check.drafter_bf16_problems(self.OFF, self.LEND_BF8 + '\n'), [])
        self.assertEqual(c2_smoke_check.drafter_bf16_problems({}, 'nothing\n'), [])
        self.assertEqual(len(c2_smoke_check.drafter_bf16_problems(self.OFF, self.ENGAGED + '\n')), 1)
        self.assertEqual(len(c2_smoke_check.drafter_bf16_problems({DRAFTER_BF16_FLAG: '0'}, self.ENGAGED + '\n')), 1)

    def test_the_real_engaged_marker_matches_the_rule(self):
        self.assertTrue(self.ENGAGED.startswith(c2_smoke_check.DRAFTER_BF16_ENGAGED))
        self.assertEqual(c2_smoke_check.DRAFTER_BF16_ENGAGED, lever_n_m3native_gate.DRAFTER_BF16_MARKER)

    def test_the_gate_plan_promises_the_engaged_marker_not_the_baked_bf8_one(self):
        baked = {'QWEN_FAST_DRAFT_BF8': '1'}
        self.assertEqual(lever_n_m3native_gate.required_flag_markers(baked, 4), {'QWEN_FAST_DRAFT_BF8': [lever_n_m3native_gate.DRAFT_BF8_MARKER]})
        both = lever_n_m3native_gate.required_flag_markers(dict(baked, **{DRAFTER_BF16_FLAG: '1'}), 4)
        self.assertEqual(both, {DRAFTER_BF16_FLAG: [lever_n_m3native_gate.DRAFTER_BF16_MARKER]})
        self.assertEqual(lever_n_m3native_gate.required_flag_markers({DRAFTER_BF16_FLAG: '0'}, 4), {})

    def test_every_dbf16_profile_resolves_to_the_engaged_marker_through_the_gate(self):
        found = profiles()['profiles']
        for name in (BEST_DBF16, GATE_DBF16, EIGHT_DBF16):
            with self.subTest(profile=name):
                environ = dict(found[name]['env'], QWEN_FAST_DRAFT_BF8='1')   # the image bakes it
                markers = lever_n_m3native_gate.required_flag_markers(environ, 4)
                self.assertIn(DRAFTER_BF16_FLAG, markers)
                self.assertNotIn('QWEN_FAST_DRAFT_BF8', markers)


class JobPackTests(unittest.TestCase):
    def parsed(self, name):
        with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
            return job.read_job(job.parse_env(handle.read()), sorted(profiles()['profiles']), root=ROOT)

    def test_order_matches_the_templates(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            lines = [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]
        self.assertEqual([line[0] for line in lines], list(EXPECTED))
        self.assertEqual(sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env')), sorted(EXPECTED))
        for name, mode, image, minutes in lines:
            self.assertEqual((mode in ('stop', 'soft'), image, minutes.isdigit()), (True, IMAGE, True), name)
        self.assertEqual(lines[0][1], 'stop')

    def test_every_template_parses_and_never_touches_the_agent_or_production(self):
        for name, (actions, profile) in EXPECTED.items():
            with self.subTest(name):
                result = self.parsed(name)
                self.assertEqual((result['actions'], result['profile'], result['cards'], result['tag']), (actions, profile, 'quad', IMAGE))
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')

    def test_the_timed_arms_alternate_abab_with_one_test_list(self):
        names = ['D1-timed-A-best-strace', 'D2-timed-B-best-dbf16', 'D3-timed-A-best-strace', 'D4-timed-B-best-dbf16']
        self.assertEqual([self.parsed(n)['profile'] for n in names], [STRACE, BEST_DBF16, STRACE, BEST_DBF16])
        self.assertEqual(len({self.parsed(n)['tests'] for n in names}), 1)
        self.assertIn('coding', self.parsed(names[0])['tests'])

    def needs(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            rows = [re.match(r'# NEEDS (.+?) <- (.+)$', line) for line in handle.read().splitlines()]
        return [(m.group(1).split(), m.group(2).split()) for m in rows if m]

    def test_a_failed_smoke_skips_every_timed_arm(self):
        needs = self.needs()
        self.assertEqual(needs, [(['D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4'], ['S1'])])
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            modes = {line.split()[0].split('-')[0]: line.split()[1] for line in handle.read().splitlines() if line.strip() and not line.startswith('#')}
        self.assertEqual((modes['X0'], modes['B0'], modes['S1']), ('stop', 'stop', 'soft'))
        for name in ('D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4'):
            self.assertEqual(modes[name], 'soft')

    def test_rescan_precedes_the_first_reset_and_s1_rescans(self):
        names = list(EXPECTED)
        self.assertEqual(names[0], 'X0-status-rescan')
        self.assertEqual(self.parsed('X0-status-rescan')['actions'].split(), ['status', 'rescan'])
        actions = self.parsed('S1-dbf16-audited-smoke')['actions'].split()
        self.assertEqual(actions, ['rescan', 'reset', 'smoke'])
        first_reset = next(n for n in names if 'reset' in EXPECTED[n][0].split())
        self.assertLess(names.index('X0-status-rescan'), names.index(first_reset))

    def test_the_eight_seat_pair_alternates_with_one_test_list(self):
        names = ['E1-eight-seat-A-best-quad', 'E2-eight-seat-B-best-quad-dbf16', 'E3-eight-seat-A-best-quad', 'E4-eight-seat-B-best-quad-dbf16']
        self.assertEqual([self.parsed(n)['profile'] for n in names], [EIGHT, EIGHT_DBF16, EIGHT, EIGHT_DBF16])
        self.assertEqual(len({self.parsed(n)['tests'] for n in names}), 1)

    def test_the_262k_twin_budget_arithmetic_in_the_pack_and_doc(self):
        blocks = 16416 + int((4.0155 - 0.9605 - 0.4054 - 1.07) * 1e9 // 557056)
        self.assertEqual(blocks, 19251)
        self.assertEqual((19200 - 8) * 64, 1228288)
        self.assertLessEqual(19200, blocks)
        for relative in ('docs/tp4-drafter-bf16.md', 'scripts/ci/references/tp4-dbf16-jobs/ORDER.txt'):
            with open(os.path.join(ROOT, relative), encoding='utf-8') as handle:
                text = handle.read()
            self.assertIn('19,200', text, relative)
            self.assertIn('1,228,288', text, relative)

    def test_no_hostname_address_registry_or_digest_and_lf(self):
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), encoding='utf-8', newline='') as handle:
                text = handle.read()
            self.assertIsNone(BANNED.search(text), name)
            self.assertNotIn('\r', text, name)
        for relative in ('scripts/ci/drafter_bf16.py', 'scripts/ci/test_tp4_dbf16.py', 'scripts/ci/draft_mlp_branch.py', 'scripts/ci/qwen_c2_profiles.json'):
            with open(os.path.join(ROOT, relative), 'rb') as handle:
                self.assertNotIn(b'\r', handle.read(), relative)


if __name__ == '__main__':
    unittest.main()
