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
import drafter_bf16  # noqa: E402
from draft_attention_branch import prepare_attention_branch  # noqa: E402
from draft_mlp_branch import DRAFT_BF8_FLAG, DRAFTER_BF16_FLAG, draft_projection_dtype, prepare_mlp_branch  # noqa: E402
from test_draft_projection_bf8 import FakeOperations, attention_weights, convolution_weights, mlp_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-dbf16-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
STRACE, BEST_DBF16 = 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-dbf16'
GATE, GATE_DBF16 = 'c2-packed-tp4-best-gate', 'c2-packed-tp4-best-gate-dbf16'
IMAGE = 'tp4-dbf16-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
EXPECTED = {
    'B0-build': ('build', 'c2-packed-tp4'), 'S1-dbf16-audited-smoke': ('reset smoke', GATE_DBF16),
    'D1-timed-A-best-strace': ('reset smoke', STRACE), 'D2-timed-B-best-dbf16': ('reset smoke', BEST_DBF16),
    'D3-timed-A-best-strace': ('reset smoke', STRACE), 'D4-timed-B-best-dbf16': ('reset smoke', BEST_DBF16),
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
        for name, base in ((BEST_DBF16, STRACE), (GATE_DBF16, GATE)):
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
        self.assertEqual(sorted(n for n, body in found.items() if DRAFTER_BF16_FLAG in body['env']), sorted([BEST_DBF16, GATE_DBF16]))
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


def round_log(counts_per_round, seconds, start=0.0):
    """A server log: per round one '[PHASE] execute ... new=0 cached=4 spec=...' stamp and four '[PACKED] request=' lines."""
    lines, clock = [], start
    for counts in counts_per_round:
        total_ms = int(clock * 1000)
        lines.append('2026-10-03 10:%02d:%02d.%03d | INFO | [PHASE] execute total=4 new=0 cached=4 spec=60' % (total_ms // 60000, total_ms // 1000 % 60, total_ms % 1000))
        for user, emitted in enumerate(counts):
            lines.append('[PACKED] request=r%d segment=0 position=100 prefix=0 emitted=%d' % (user, emitted))
        clock += seconds
    total_ms = int(clock * 1000)
    lines.append('2026-10-03 10:%02d:%02d.%03d | INFO | [PHASE] execute total=4 new=0 cached=4 spec=60' % (total_ms // 60000, total_ms // 1000 % 60, total_ms % 1000))
    return '\n'.join(lines) + '\n'


class CompareTests(unittest.TestCase):
    def test_tau_round_and_rate_per_arm_and_the_ratios(self):
        a = round_log([(6, 6, 6, 6)] * 10, 0.100)
        b = round_log([(7, 6, 7, 6)] * 10, 0.101)
        result = drafter_bf16.compare(a, b, 4)
        self.assertEqual((result['a']['tau'], result['b']['tau']), (6.0, 6.5))
        self.assertEqual((result['a']['median_round_ms'], result['b']['median_round_ms']), (100.0, 101.0))
        self.assertEqual((result['a']['per_user_tok_s'], result['b']['per_user_tok_s']), (60.0, 64.36))
        self.assertAlmostEqual(result['tau_ratio_b_over_a'], 1.0833, places=4)
        self.assertAlmostEqual(result['round_time_ratio_b_over_a'], 1.01, places=4)
        self.assertTrue(result['bf16_wins'])
        self.assertEqual(result['paired_rounds'], 10)
        self.assertEqual(result['per_round_committed']['b'][0], [7, 6, 7, 6])

    def test_a_tau_gain_below_the_extra_round_time_does_not_win(self):
        result = drafter_bf16.compare(round_log([(6, 6, 6, 6)] * 6, 0.100), round_log([(6, 6, 6, 6)] * 6, 0.101), 4)
        self.assertFalse(result['bf16_wins'])

    def test_an_arm_without_timed_rounds_is_refused(self):
        with self.assertRaises(ValueError):
            drafter_bf16.compare('nothing\n', round_log([(6, 6, 6, 6)] * 3, 0.1), 4)


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
