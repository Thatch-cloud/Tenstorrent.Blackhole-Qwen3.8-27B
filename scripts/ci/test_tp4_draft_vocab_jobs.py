"""The tp4/draft-vocab window (docs/tp4-draft-vocab.md): the timing twins, the tau lab's dvocab arm, the paired timing read and the job pack (scripts/ci/references/tp4-draft-vocab-jobs).

Holds: the generator's twins are their parent plus the pool given back plus exactly the flag, gate only, exempted from the profile-enumerating tests; the lab's dvocab arm adds only the flag to the
production profile, is held to its attach markers before anything is sent and to the engaged line by the canary, boots through the contract, and is judged by the noninferiority bar; the paired
timing read (draft_vocab_report) reads the early draft and the paired rounds and says GO, NO-GO, MAYBE or UNREAD by its pre-registered rule; ORDER.txt lists exactly the pack's templates, each once, with
one image tag, a total that matches its header and NEEDS lines that name earlier jobs; every template passes the job reader (the tau arm once its control run id replaces the placeholder); and the pack
names no host, address, registry or home path (the repo is public)."""

import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402
import c2_tau_lab as lab  # noqa: E402
import draft_vocab_report as timing  # noqa: E402
import draft_vocab_tp  # noqa: E402
import drafter_pair_report as pair_report  # noqa: E402
import make_dvocab_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402
import tau_lab_report as report  # noqa: E402
import test_tau_lab as harness  # noqa: E402
from test_tp4_dbf16 import round_log  # noqa: E402

FOLDER = HERE / 'references' / 'tp4-draft-vocab-jobs'
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
PROFILES = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
BASE = 'c2-packed-tp4'
IMAGE = 'tp4-dvocab-1'
SHIP = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'
TESTS = 'warmup,concurrent8_steady,concurrent8_code_equal,concurrent8_code_32k'
LIST = 'coding-40960'

EXPECTED = (
    ('B0-build', 'stop', 40), ('X0-status-rescan-reset', 'stop', 20), ('S1-dvocab-engage-smoke', 'soft', 30),
    ('D-V1-control', 'soft', 90), ('D-V2-dvocab', 'soft', 85),
    ('TV1-timed-A-control', 'soft', 55), ('TV2-timed-B-dvocab', 'soft', 55), ('TV3-timed-A-control', 'soft', 55), ('TV4-timed-B-dvocab', 'soft', 55),
    ('Z-reset', 'soft', 10),
)
TOTAL = 495


def order_text():
    return (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')


def order_lines():
    return [line.split() for line in order_text().splitlines() if line.strip() and not line.startswith('#')]


def values(name):
    return job.parse_env((FOLDER / (name + '.env')).read_text(encoding='utf-8'))


def short(name):
    """The job's short name: D-V1, S1, TV3, X0, Z."""
    return re.match(r'(D-V\d+|[A-Z]+\d*)', name).group(1)


class ProfileTwinTests(unittest.TestCase):
    def found(self):
        return json.loads(PROFILES_PATH.read_text(encoding='utf-8'))['profiles']

    def test_the_checked_in_twins_are_what_the_parent_generates(self):
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_control_is_the_parent_with_the_pool_given_back_and_nothing_else(self):
        found = self.found()
        parent, control = found[twins.PARENT], found[twins.CONTROL]
        self.assertEqual(control['engine']['num-gpu-blocks-override'], 19712)
        self.assertEqual(parent['engine']['num-gpu-blocks-override'], 19968)
        self.assertEqual({key: value for key, value in control['engine'].items() if key != 'num-gpu-blocks-override'},
                         {key: value for key, value in parent['engine'].items() if key != 'num-gpu-blocks-override'})
        self.assertEqual({key: value for key, value in control['env'].items() if parent['env'].get(key) != value},
                         {'QWEN36_MAX_TOKENS_ALL_USERS': str((19712 - 8) * 64)})
        self.assertEqual(set(control['env']), set(parent['env']))
        for key in parent:
            if key not in ('env', 'engine', 'description'):
                self.assertEqual(control[key], parent[key], key)
        self.assertTrue(control['gate_only'])
        self.assertNotIn(draft_vocab_tp.FLAG, control['env'])

    def test_the_arm_is_the_control_plus_the_flag_and_nothing_else(self):
        found = self.found()
        control, arm = found[twins.CONTROL], found[twins.TWIN]
        self.assertEqual(sorted(set(arm['env']) - set(control['env'])), [draft_vocab_tp.FLAG])
        self.assertEqual(arm['env'][draft_vocab_tp.FLAG], LIST)
        self.assertEqual({key: value for key, value in arm['env'].items() if key in control['env']}, control['env'])
        self.assertEqual(arm['engine'], control['engine'])
        self.assertTrue(arm['gate_only'])
        self.assertEqual(draft_vocab_tp.profile_problems(dict(arm, name=twins.TWIN)), [])

    def test_the_pool_arithmetic_gives_back_more_than_the_head_needs(self):
        given_back = (19968 - 19712) * 557056
        self.assertAlmostEqual(given_back / 1e6, twins.given_back_mb(), places=3)
        # the head's worst case per chip: a bfloat16 copy of the list's columns (bfloat8_b is about half)
        worst = 5120 * 10240 * 2
        self.assertGreater(given_back, worst)
        self.assertGreater(given_back - worst, 30e6, 'a margin beyond the head itself')

    def test_the_twins_are_exempted_from_the_profile_enumerating_tests(self):
        self.assertEqual(set(twins.twin_names()), {twins.CONTROL, twins.TWIN})
        self.assertTrue(set(twins.twin_names()) <= set(profile_twins.twin_names()))

    def test_only_the_arm_names_the_flag_and_the_traffic_profiles_never_do(self):
        named = sorted(name for name, profile in self.found().items() if any(key.startswith(draft_vocab_tp.PREFIX) for key in profile['env']))
        self.assertEqual(named, [twins.TWIN])
        for name, profile in self.found().items():
            if not profile.get('gate_only'):
                self.assertFalse(any(key.startswith(draft_vocab_tp.PREFIX) for key in profile['env']), name)

    def test_the_contract_accepts_the_arm_and_refuses_it_as_traffic(self):
        import serving_c2_contract as contract

        arm = dict(self.found()[twins.TWIN], name=twins.TWIN)
        self.assertEqual(contract.draft_vocab_problems(arm), [])
        traffic = dict(arm, gate_only=False)
        traffic.pop('gate_only')
        self.assertTrue(contract.draft_vocab_problems(dict(traffic, name='c2-packed-tp4-x')))

    def test_a_parent_that_moved_is_refused(self):
        found = self.found()
        for mutate in (lambda p: p['env'].__setitem__('QWEN_FAST_QUAD_DRAFT_BLOCKS', '1'), lambda p: p['env'].__setitem__(draft_vocab_tp.FLAG, LIST),
                       lambda p: p['engine'].__setitem__('num-gpu-blocks-override', 19000), lambda p: p.__setitem__('gate_only', False),
                       lambda p: p['env'].__setitem__('QWEN36_MAX_TOKENS_ALL_USERS', '1')):
            parent = copy.deepcopy(found[twins.PARENT])
            mutate(parent)
            self.assertTrue(twins.parent_problems(parent))
        self.assertEqual(twins.parent_problems(found[twins.PARENT]), [])
        with self.assertRaises(ValueError):
            twins.generate({'profiles': {}})


class LabArmTests(unittest.TestCase):
    def added(self):
        name, document = lab.derive_profile(PROFILES, BASE, 'dvocab')
        derived = document['profiles'][name]
        base = PROFILES['profiles'][BASE]
        return name, document, {key: value for key, value in derived['env'].items() if base['env'].get(key) != value}

    def test_the_arm_adds_the_list_to_the_serving_profile_and_nothing_else(self):
        name, document, added = self.added()
        self.assertEqual(name, BASE + '+taulab+dvocab')
        self.assertEqual(added, dict(QWEN_FAST_PACKED_AUDIT='1', QWEN_FAST_PHASE_TIMING='1', QWEN_FAST_DRAFT_VOCAB=LIST))
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document, 'dvocab'), [])
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document), ['QWEN_FAST_DRAFT_VOCAB'])
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document, 'lookup'), ['QWEN_FAST_DRAFT_VOCAB'])
        self.assertEqual(lab.DRAFTER_ARMS['dvocab']['manifest'], lab.DRAFTER_ARMS['control']['manifest'])
        self.assertNotIn('QWEN_FAST_DRAFT_VOCAB', PROFILES['profiles'][BASE]['env'], 'the control arm carries no list')

    def test_the_names_are_the_modules(self):
        self.assertEqual(lab.VOCAB_FLAG, draft_vocab_tp.FLAG)
        self.assertEqual(lab.VOCAB_LIST, LIST)
        self.assertIn(lab.VOCAB_LIST, draft_vocab_tp.NAMED)
        self.assertEqual((lab.VOCAB_ADMITTED, lab.VOCAB_BUILT, lab.VOCAB_ENGAGED), (draft_vocab_tp.ADMITTED, draft_vocab_tp.BUILT, draft_vocab_tp.ENGAGED))
        self.assertEqual(job.TAULAB_DRAFTER_ARMS[-1], 'dvocab')
        self.assertIn('dvocab', pair_report.CANDIDATE_ARMS)
        self.assertIn('dvocab', pair_report.SAME_IMAGE_ARMS)
        self.assertIn('dvocab', report.PUBLIC_WORDS)
        self.assertEqual(sorted(pair_report.CANDIDATE_ARMS + ('control',)), sorted(lab.DRAFTER_ARMS))

    def test_the_contract_boots_the_derived_profile_and_exports_the_list(self):
        import serving_c2_contract as contract

        name, document = lab.derive_profile(PROFILES, BASE, 'dvocab')
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'profiles.json')
        with open(path, 'w') as handle:
            json.dump(document, handle)
        profile = contract.load_profile(path, name)
        self.assertEqual(contract.apply_environment(profile, {})['QWEN_FAST_DRAFT_VOCAB'], LIST)
        for check in (contract.draft_vocab_problems, contract.parked_problems, contract.drafter_problems, contract.levern_problems, contract.mesh_problems,
                      contract.sticky_problems, contract.prefix_reuse_problems, contract.multi_problems, contract.traffic_waiver_problems):
            self.assertEqual(check(profile), [], check.__name__)
        # the same profile without the lab's name is a traffic profile naming the flag: refused
        stray = dict(profile, name=BASE)
        self.assertTrue(contract.draft_vocab_problems(stray))

    def test_the_attach_markers_are_held_at_launch_and_no_other_arm_may_show_them(self):
        derived = BASE + '+taulab+dvocab'
        good = ['[QWEN-C2] profile %s' % derived, '%s: rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6' % draft_vocab_tp.ADMITTED,
                '%s: rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 dtype=bf8' % draft_vocab_tp.BUILT]
        self.assertEqual(lab.launched_problems(good, derived, 'dvocab'), [])
        self.assertEqual(len(lab.launched_problems(good[:1], derived, 'dvocab')), 2, 'neither marker shown')
        self.assertEqual(len(lab.launched_problems(good[:2], derived, 'dvocab')), 1)
        for arm in ('control', 'b16-bf8', 'dedf-bf16', 'lookup'):
            asked = BASE + '+taulab' if arm == 'control' else BASE + '+taulab+' + arm
            self.assertTrue(lab.launched_problems(['[QWEN-C2] profile %s' % asked] + good[1:], asked, arm), arm)
        # the arms that were never asked about a list are untouched
        control = BASE + '+taulab'
        self.assertEqual(lab.launched_problems(['[QWEN-C2] profile %s' % control], control, 'control'), [])

    def test_the_canary_holds_the_engaged_line_to_the_dvocab_arm_only(self):
        engaged = '%s rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 dtype=bf8 chunks=1 (was 2)' % draft_vocab_tp.ENGAGED
        self.assertIsNone(lab.vocab_problem('noise\n' + engaged, 'dvocab'))
        self.assertIn('never shows it engaged', lab.vocab_problem('noise', 'dvocab'))
        for arm in ('control', 'b16-bf8', 'b32-bf8', 'dedf-bf16', 'lookup'):
            self.assertIsNone(lab.vocab_problem('noise', arm), arm)
            self.assertIn('does not ask for a draft vocabulary', lab.vocab_problem(engaged, arm), arm)

    def test_the_vocab_canary_waits_for_the_line_the_arm_needs_and_not_for_the_ones_it_forbids(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'server.log')
        now, slept = [0.0], []

        def sleep(seconds):
            slept.append(seconds)
            now[0] += seconds
            if len(slept) == 2:
                with open(path, 'a') as handle:
                    handle.write('%s rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6\n' % draft_vocab_tp.ENGAGED)

        self.assertIsNone(lab.with_vocab_check(lambda: None, path, 'dvocab', clock=lambda: now[0], sleep=sleep, wait=60)())
        self.assertEqual(len(slept), 2)
        os.remove(path)
        now[0] = 0.0
        self.assertIn('never shows it engaged', lab.with_vocab_check(lambda: None, path, 'dvocab', clock=lambda: now[0],
                                                                    sleep=lambda seconds: now.__setitem__(0, now[0] + seconds), wait=20)())
        slept[:] = []
        self.assertIsNone(lab.with_vocab_check(lambda: None, path, 'control', clock=lambda: now[0], sleep=sleep, wait=60)())
        with open(path, 'w') as handle:
            handle.write('%s rows=1\n' % draft_vocab_tp.ENGAGED)
        self.assertIn('does not ask for a draft vocabulary', lab.with_vocab_check(lambda: None, path, 'control', clock=lambda: now[0], sleep=sleep, wait=60)())
        self.assertEqual(slept, [])
        self.assertEqual(lab.with_vocab_check(lambda: 'no rounds', path, 'dvocab', clock=lambda: now[0], sleep=sleep, wait=60)(), 'no rounds')
        # the lookup canary is still the lookup's, through the shared check
        self.assertIsNone(lab.with_lookup_check(lambda: None, path, 'control', clock=lambda: now[0], sleep=sleep, wait=60)())

    def test_the_report_names_the_arm_and_stays_public(self):
        report.assert_public(dict(drafter=dict(arm='dvocab', bf16=False), label='non-production', verdict='GO'))


class LabRunTests(unittest.TestCase):
    """main() against the lab's fakes: the dvocab arm runs when its markers show, and the canary stops it when the flag never crossed into the container."""

    STARTUP = ('%s: rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 (GATE ONLY, UNQUALIFIED)' % draft_vocab_tp.ADMITTED,
               '%s: rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 dtype=bf8 bytes_per_chip=55705600' % draft_vocab_tp.BUILT)
    PER_REQUEST = '%s rows=40960 per_chip=10240 chips=4 topk_width=32768 sha256=94fbd2a832f6 dtype=bf8 chunks=1 (was 2) rows_per_launch=16' % draft_vocab_tp.ENGAGED

    class Engine(harness.FakeEngine):
        def stream(self, path, body, timeout, keep):
            result = harness.FakeEngine.stream(self, path, body, timeout, keep)
            self.emit([LabRunTests.PER_REQUEST])
            return result

    def run_arm(self, arm, log_for, arguments=(), **options):
        def start(follower, since=None):
            follower.engine.path = follower.path
            follower.engine.emit(log_for)
        with mock.patch.object(harness.FakeLog, 'start', start):
            return harness.run_lab(self, arguments=['--drafter-arm', arm] + list(arguments), **options)

    def derived(self, arm):
        return lab.derive_profile(PROFILES, BASE, arm)[0]

    def test_a_dvocab_arm_whose_attach_lines_show_runs_and_the_run_records_the_arm(self):
        log = ['[QWEN-C2] profile %s' % self.derived('dvocab')] + list(self.STARTUP)
        code, out, results, public, engine, _ = self.run_arm('dvocab', log, ['--arms', 'A1'], engine=self.Engine())
        self.assertNotEqual(code, 2, out)
        self.assertTrue(engine.requests)
        with open(os.path.join(results, 'run.json'), encoding='utf-8') as handle:
            info = json.load(handle)
        self.assertEqual((info['drafter_arm'], info['launched_problems']), ('dvocab', []))
        self.assertFalse(info.get('tripped'), info)
        with open(os.path.join(results, 'profiles.json'), encoding='utf-8') as handle:
            derived = json.load(handle)
        self.assertEqual(derived['profiles'][self.derived('dvocab')]['env']['QWEN_FAST_DRAFT_VOCAB'], LIST)
        self.assertTrue(any('canary ok' in line for line in out), out)

    def test_a_dvocab_arm_whose_flag_never_reached_the_container_is_refused_before_anything_is_sent(self):
        code, out, results, public, engine, _ = self.run_arm('dvocab', ['[QWEN-C2] profile %s' % self.derived('dvocab')], ['--arms', 'A1'])
        self.assertEqual((code, engine.requests), (2, []))

    def test_a_dvocab_arm_that_never_engaged_is_stopped_by_the_canary(self):
        log = ['[QWEN-C2] profile %s' % self.derived('dvocab')] + list(self.STARTUP)
        with mock.patch.object(lab, 'CANARY_WAIT_SECONDS', 0.05):
            code, out, results, public, engine, _ = self.run_arm('dvocab', log, ['--arms', 'A1'])
        self.assertEqual(code, 1)
        with open(os.path.join(results, 'run.json'), encoding='utf-8') as handle:
            info = json.load(handle)
        self.assertTrue(info['tripped'], info)
        self.assertTrue(any('canary' in line and 'never shows it engaged' in line for line in out), out)

    def test_a_control_whose_log_shows_the_list_is_stopped(self):
        log = ['[QWEN-C2] profile %s' % self.derived('control')] + list(self.STARTUP)
        code, out, _, _, engine, _ = self.run_arm('control', log, ['--arms', 'A1'], engine=self.Engine())
        self.assertEqual((code, engine.requests), (2, []), 'the attach lines on a control are refused at launch')


class PairReportTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.count = 0

    def run_pair(self, gain, arm, candidate_extra=None, resamples=400):
        """test_drafter_arms' synthetic pair (the candidate's tau at `gain` times the control's) through the pair report for `arm`."""
        import test_drafter_arms as arms

        control, candidate = arms.synthetic(gain=gain)
        self.count += 1
        base = os.path.join(self.root, 'run%d' % self.count)
        arms.write_run(os.path.join(base, 'control'), control, arms.run_info('control', 'tp4-drafter-1'), public=dict(calibration=dict(verdict='PASS')))
        arms.write_run(os.path.join(base, 'candidate'), candidate, arms.run_info(arm, 'tp4-drafter-1', **(candidate_extra or {})))
        controls, control_info, summary = pair_report.read_run(os.path.join(base, 'control'))
        candidates, candidate_info, _ = pair_report.read_run(os.path.join(base, 'candidate'))
        public = pair_report.build(controls, candidates, arm, control_info, candidate_info, summary, seed=1, resamples=resamples)
        report.assert_public(public)
        return public

    def test_a_shortlist_is_judged_by_noninferiority_not_by_gain(self):
        self.assertEqual(pair_report.NONINFERIOR_ARMS, {'dvocab': 0.97})
        # no change in tau passes the noninferiority bar for dvocab and fails the gain bar of every other arm
        flat = self.run_pair(1.0, 'dvocab')
        self.assertEqual((flat['verdict'], flat['rule']['pooled_bar']), ('GO', 0.97), flat['gates'])
        other = self.run_pair(1.0, 'lookup')
        self.assertEqual((other['verdict'], other['rule']['pooled_bar']), ('NO-GO', 1.0))
        # a 1 percent tau cost is inside the bar, a 10 percent one is not
        self.assertEqual(self.run_pair(0.99, 'dvocab')['gates']['pooled'], 'PASS')
        lost = self.run_pair(0.90, 'dvocab')
        self.assertEqual((lost['gates']['pooled'], lost['verdict']), ('FAIL', 'NO-GO'))

    def test_the_other_gates_still_apply_to_the_arm(self):
        self.assertEqual(self.run_pair(1.0, 'dvocab', candidate_extra=dict(image_tag='tp4-drafter-b16-1'))['gates']['matched'], 'FAIL')
        self.assertEqual(self.run_pair(1.0, 'dvocab')['gates']['matched'], 'PASS')


def timed_log(rounds, seconds, drafts, users=8, live=8):
    """A container log of `rounds` eight-seat rounds of `seconds` each (per-user committed counts `committed`) with `drafts` early-draft lines at `live` seats."""
    text = round_log(rounds, seconds, users=users)
    lines = ['[PACKED-EARLY-DRAFT] round=%d path=reuse live=%d draft_ms=%.3f reason=-' % (index, live, ms) for index, ms in enumerate(drafts)]
    return text + '\n'.join(lines) + '\n'


class TimingReportTests(unittest.TestCase):
    def logs(self, control_draft=43.0, arm_draft=38.0, control_round=0.300, arm_round=0.292, control_tokens=4, arm_tokens=4, count=120):
        control = timed_log([(control_tokens,) * 8] * count, control_round, [control_draft] * count)
        arm = timed_log([(arm_tokens,) * 8] * count, arm_round, [arm_draft] * count)
        return control, arm

    def test_the_early_draft_lines_are_read_at_the_live_count_and_the_paths_that_drafted(self):
        text = ('[PACKED-EARLY-DRAFT] round=1 path=reuse live=8 draft_ms=40.5 reason=-\n[PACKED-EARLY-DRAFT] round=2 path=redo live=8 draft_ms=41.5 reason=x\n'
                '[PACKED-EARLY-DRAFT] round=3 path=failed live=8 draft_ms=99.0 reason=x\n[PACKED-EARLY-DRAFT] round=4 path=reuse live=4 draft_ms=12.0 reason=-\n'
                '[PACKED-EARLY-DRAFT] round=5 path=untaken live=8 draft_ms=77.0 reason=-\n')
        self.assertEqual(timing.early_draft_ms(text, 8), [40.5, 41.5])
        self.assertEqual(timing.early_draft_ms(text, 4), [12.0])
        self.assertEqual(timing.early_draft_ms('', 8), [])
        self.assertEqual(timing.draft_summary([1.0, 3.0, 2.0])['median'], 2.0)

    def test_the_real_early_draft_line_parses(self):
        # the producer's own format (early_draft.py MARKER, lever_n_m3native_gate.EARLY_LINE): the report reads what the server writes
        import lever_n_m3native_gate as gate

        line = '[PACKED-EARLY-DRAFT] round=7 path=reuse live=8 draft_ms=43.215 reason=-'
        self.assertTrue(gate.EARLY_LINE.search(line))
        self.assertEqual(timing.early_draft_ms(line, 8), [43.215])

    def test_a_pair_that_saves_draft_time_at_the_same_tau_is_a_go(self):
        control, arm = self.logs()
        result = timing.pair(control, arm, 8)
        self.assertEqual(result['verdict'], 'GO', result['reason'])
        self.assertEqual(result['draft_ms_change'], -5.0)
        self.assertGreater(result['paired']['rate_ratio_b_over_a'], 1.02)
        self.assertEqual(result['paired']['tau_ratio_b_over_a'], 1.0)

    def test_a_tau_loss_larger_than_the_time_gain_is_a_no_go(self):
        control, arm = self.logs(arm_tokens=4, control_tokens=5)
        # tau 4 against 5 (0.8 of the control's) with a faster round: slower per user, and under the loss floor
        self.assertEqual(timing.pair(control, arm, 8)['verdict'], 'NO-GO')
        # a small tau loss that the time more than pays for is still a go at the bar; a loss inside 3 percent and no rate gain is a maybe
        control, arm = self.logs(arm_draft=42.9, arm_round=0.2999)
        self.assertEqual(timing.pair(control, arm, 8)['verdict'], 'MAYBE')

    def test_a_pair_where_the_early_draft_did_not_fall_is_not_a_go(self):
        control, arm = self.logs(arm_draft=44.0)
        self.assertNotEqual(timing.pair(control, arm, 8)['verdict'], 'GO')

    def test_too_few_rounds_or_draft_lines_is_unread(self):
        control, arm = self.logs(count=40)
        self.assertEqual(timing.pair(control, arm, 8)['verdict'], 'UNREAD')
        control, arm = self.logs()
        self.assertEqual(timing.pair(control, arm.replace('live=8', 'live=7'), 8)['verdict'], 'UNREAD')
        self.assertEqual(timing.pair('nothing\n', arm, 8)['verdict'], 'UNREAD')

    def test_the_window_is_a_go_only_when_every_pair_is(self):
        good = self.logs()
        bad = self.logs(arm_draft=44.0, arm_round=0.305)
        self.assertEqual(timing.compare([good[0], good[0]], [good[1], good[1]])['verdict'], 'GO')
        self.assertEqual(timing.compare([good[0], bad[0]], [good[1], bad[1]])['verdict'], 'NO-GO')
        self.assertEqual(timing.compare([good[0], good[0]], [good[1], 'nothing\n'])['verdict'], 'UNREAD')
        self.assertEqual(timing.compare([good[0]], [good[1]])['draft_ms_saved'], [5.0])
        with self.assertRaises(ValueError):
            timing.compare([good[0]], [good[1], good[1]])
        with self.assertRaises(ValueError):
            timing.compare([], [])

    def test_the_command_line_prints_the_verdict_and_exits_by_it(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        control, arm = self.logs()
        paths = []
        for name, text in (('a1', control), ('b1', arm), ('a2', control), ('b2', arm)):
            paths.append(os.path.join(directory, name + '.log'))
            Path(paths[-1]).write_text(text, encoding='utf-8')
        with mock.patch('builtins.print') as shown:
            code = timing.main(['--control', paths[0], paths[2], '--arm', paths[1], paths[3]])
        self.assertEqual(code, 0)
        self.assertTrue(any('DRAFT_VOCAB_VERDICT GO' in str(call) for call in shown.call_args_list))
        with mock.patch('builtins.print'):
            self.assertEqual(timing.main(['--control', paths[0], '--arm', os.path.join(directory, 'absent.log')]), 2)
        self.assertEqual(timing.EXIT, {'GO': 0, 'NO-GO': 1, 'UNREAD': 2, 'MAYBE': 3})


class OrderTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        self.assertEqual([line[0] for line in lines], [name for name, _mode, _minutes in EXPECTED])
        self.assertEqual(sorted(path.stem for path in FOLDER.glob('*.env')), sorted(name for name, _mode, _minutes in EXPECTED))
        for (name, mode, image, minutes), (_expected, expected_mode, expected_minutes) in zip(lines, EXPECTED):
            self.assertEqual((mode, image, int(minutes)), (expected_mode, IMAGE, expected_minutes), name)

    def test_the_order_is_the_priority(self):
        names = [short(line[0]) for line in order_lines()]
        self.assertEqual(names[:3], ['B0', 'X0', 'S1'], 'the smallest safe step on the cards comes before any timing')
        self.assertEqual([line[0] for line in order_lines()][3:5], ['D-V1-control', 'D-V2-dvocab'])
        self.assertEqual(names[5:9], ['TV1', 'TV2', 'TV3', 'TV4'], 'the ABAB: control, arm, control, arm')
        self.assertEqual(names[-1], 'Z')

    def test_the_total_in_the_header_is_the_sum(self):
        total = sum(int(line[3]) for line in order_lines())
        self.assertEqual(total, TOTAL)
        self.assertIn('%d minutes (%d h %d min)' % (total, total // 60, total % 60), order_text())
        self.assertIn('%d of them on cards' % (total - 40), order_text())
        self.assertIn('Without the tau pair (175 minutes of cards) it is\n#     %d minutes' % (total - 175), order_text())

    def test_the_needs_lines_name_known_jobs_that_run_earlier(self):
        position = dict((short(line[0]), index) for index, line in enumerate(order_lines()))
        needs = re.findall(r'^# NEEDS (.+?) <- (.+)$', order_text(), re.M)
        self.assertGreaterEqual(len(needs), 6)
        waiting = set()
        for left, right in needs:
            for name in left.split():
                self.assertIn(name, position, name)
                waiting.add(name)
                for needed in right.split():
                    self.assertIn(needed, position, needed)
                    self.assertLess(position[needed], position[name], '%s needs %s, which runs later' % (name, needed))
        for name in ('X0', 'S1', 'D-V1', 'D-V2', 'TV1', 'TV2', 'TV3', 'TV4'):
            self.assertIn(name, waiting, name)

    def test_only_the_build_and_the_card_state_stop_the_window(self):
        self.assertEqual([line[0] for line in order_lines() if line[1] == 'stop'], ['B0-build', 'X0-status-rescan-reset'])

    def test_the_read_rules_name_tools_that_exist(self):
        text = order_text() + ''.join(path.read_text(encoding='utf-8') for path in FOLDER.glob('*.env'))
        for tool in ('draft_vocab_report.py', 'levern_compare.py', 'drafter_pair_report.py'):
            self.assertIn(tool, text, tool)
            self.assertTrue((HERE / tool).exists(), tool)
        self.assertIn('c2_smoke_check', text)


class JobTests(unittest.TestCase):
    def test_every_template_passes_the_job_reader(self):
        profiles = job.profile_names(str(PROFILES_PATH))
        envs = job.profile_envs(str(PROFILES_PATH))
        for name, _mode, _minutes in EXPECTED:
            with self.subTest(name):
                text = (FOLDER / (name + '.env')).read_text(encoding='utf-8').replace('@CONTROL_RUN@', '123456789')
                outputs = job.read_job(job.parse_env(text), profiles, envs=envs)
                self.assertTrue(outputs['actions'])

    def test_the_tau_arm_is_refused_until_its_control_run_id_is_filled_in(self):
        profiles = job.profile_names(str(PROFILES_PATH))
        envs = job.profile_envs(str(PROFILES_PATH))
        with self.assertRaises(Exception) as raised:
            job.read_job(values('D-V2-dvocab'), profiles, envs=envs)
        self.assertIn('C2_TAULAB_PAIR_CONTROL', str(raised.exception))
        self.assertEqual(values('D-V2-dvocab')['C2_TAULAB_PAIR_CONTROL'], '@CONTROL_RUN@')
        self.assertNotIn('C2_TAULAB_PAIR_CONTROL', values('D-V1-control'))

    def test_b0_builds_one_image_and_bakes_the_profile_production_runs(self):
        b0 = values('B0-build')
        self.assertEqual((b0['C2_ACTIONS'], b0['C2_PROFILE'], b0['C2_IMAGE_TAG']), ('build', BASE, IMAGE))
        self.assertEqual(b0['C2_BAKE_DEFAULT_PROFILE'], SHIP)
        window = job.parse_env((HERE / 'references' / 'tp4-next2-1-jobs' / 'B0-build.env').read_text(encoding='utf-8'))
        self.assertEqual(b0['C2_BAKE_DEFAULT_PROFILE'], window['C2_BAKE_DEFAULT_PROFILE'])
        self.assertEqual([name for name, *_ in EXPECTED if values(name).get('C2_ACTIONS') == 'build'], ['B0-build'])

    def test_every_job_serves_the_one_image(self):
        for name, _mode, _minutes in EXPECTED:
            self.assertEqual(values(name).get('C2_IMAGE_TAG'), IMAGE, name)

    def test_the_tau_pair_is_a_same_image_control_and_arm(self):
        control, arm = values('D-V1-control'), values('D-V2-dvocab')
        self.assertEqual((control['C2_ACTIONS'], arm['C2_ACTIONS']), ('reset taulab', 'reset taulab'))
        self.assertEqual((control['C2_TAULAB_DRAFTER_ARM'], arm['C2_TAULAB_DRAFTER_ARM']), ('control', 'dvocab'))
        self.assertEqual(control['C2_IMAGE_TAG'], arm['C2_IMAGE_TAG'])
        self.assertEqual(control['C2_TAULAB_PROFILE'], arm['C2_TAULAB_PROFILE'])
        self.assertEqual(control['C2_TAULAB_ARMS'], 'A3 A1 A2', 'A3 calibrates the control arm only')
        self.assertEqual(arm['C2_TAULAB_ARMS'], 'A1 A2')
        self.assertEqual(control['C2_TAULAB_IN_FLIGHT'], arm['C2_TAULAB_IN_FLIGHT'])
        self.assertIn('dvocab', job.TAULAB_DRAFTER_ARMS)

    def test_the_timed_pair_is_an_abab_on_the_twins_running_the_same_tests(self):
        found = PROFILES['profiles']
        profile_of = {'TV1-timed-A-control': twins.CONTROL, 'TV2-timed-B-dvocab': twins.TWIN, 'TV3-timed-A-control': twins.CONTROL, 'TV4-timed-B-dvocab': twins.TWIN}
        for name, profile in profile_of.items():
            self.assertEqual(values(name)['C2_PROFILE'], profile, name)
            self.assertEqual(values(name)['C2_SMOKE_TESTS'], TESTS, name)
            self.assertIn(profile, found)
            self.assertEqual(values(name)['C2_ACTIONS'], 'reset smoke')
        self.assertEqual(values('S1-dvocab-engage-smoke')['C2_PROFILE'], twins.TWIN)
        self.assertEqual(values('S1-dvocab-engage-smoke')['C2_SMOKE_TESTS'], 'warmup,concurrent8_steady,coding')
        # every twin the generator makes has its job
        self.assertEqual(set(profile_of.values()), set(twins.twin_names()))
        self.assertTrue(found[twins.TWIN]['gate_only'] and found[twins.CONTROL]['gate_only'])

    def test_no_job_starts_the_node_agent_or_pushes_a_tag(self):
        for name, _mode, _minutes in EXPECTED:
            actions = values(name)['C2_ACTIONS'].split()
            self.assertFalse({'agent', 'start', 'serve', 'deploy', 'agentstart', 'agentstop', 'push'} & set(actions), name)


class DocsAndPublicTests(unittest.TestCase):
    def test_the_pack_names_no_host_address_registry_or_home_path(self):
        pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|/home/|/Users/|[A-Za-z]:[\\/]|\.local\b|\.lan\b|\bssh\b|ghcr\.io|docker\.io|sha256:[0-9a-f]{12}|spark-|thatch@')
        for path in sorted(FOLDER.iterdir()):
            text = path.read_text(encoding='utf-8')
            with self.subTest(path.name):
                self.assertIsNone(pattern.search(text), pattern.search(text) and pattern.search(text).group(0))
                self.assertNotIn('\r', text)

    def test_the_docs_name_the_arm_the_flag_and_the_pack(self):
        arms = (HERE.parent.parent / 'docs' / 'drafter-arms.md').read_text(encoding='utf-8')
        self.assertIn('| `dvocab` |', arms)
        self.assertIn('QWEN_FAST_DRAFT_VOCAB=coding-40960', arms)
        design = (HERE.parent.parent / 'docs' / 'tp4-draft-vocab.md').read_text(encoding='utf-8')
        for word in ('QWEN_FAST_DRAFT_VOCAB', 'tp4-draft-vocab-jobs', 'draft_vocab_report.py', 'ESTIMATE'):
            self.assertIn(word, design, word)

    def test_the_smoke_rule_and_the_pack_agree_on_the_list(self):
        self.assertEqual(c2_smoke_check.DRAFT_VOCAB_NAMED_ROWS[LIST], 40960)
        self.assertIn('rows=40960', order_text())


if __name__ == '__main__':
    unittest.main()
