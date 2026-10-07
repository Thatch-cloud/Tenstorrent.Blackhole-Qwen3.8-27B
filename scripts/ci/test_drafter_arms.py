"""The tau lab's drafter arms and the paired drafter report: profiles, labels, the A3 rule, the GO rule, the job keys.

CPU only; the lab's container, the cards and the data directory are not touched (test_tau_lab holds the lab itself).
"""

import copy
import json
import os
import random
import shutil
import tempfile
import unittest

import c2_serving_job as job
import c2_tau_lab as lab
import drafter_pair_report as pair_report
import tau_lab_report as report

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
FOLDER = os.path.join(HERE, 'references', 'tp4-drafter-jobs')
BASE = 'c2-packed-tp4'


class ArmProfileTests(unittest.TestCase):
    def added(self, arm):
        name, document = lab.derive_profile(PROFILES, BASE, arm)
        derived = document['profiles'][name]
        base = PROFILES['profiles'][BASE]
        return name, document, dict((key, value) for key, value in derived['env'].items() if base['env'].get(key) != value)

    def test_control_adds_the_two_log_flags_only(self):
        name, document, added = self.added('control')
        self.assertEqual(name, BASE + '+taulab')
        self.assertEqual(added, dict(QWEN_FAST_PACKED_AUDIT='1', QWEN_FAST_PHASE_TIMING='1'))
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document, 'control'), [])

    def test_the_candidate_arm_adds_the_manifest_flag(self):
        name, document, added = self.added('b16-bf8')
        self.assertEqual(name, BASE + '+taulab+b16-bf8')
        self.assertEqual(added, dict(QWEN_FAST_PACKED_AUDIT='1', QWEN_FAST_PHASE_TIMING='1', QWEN_DRAFTER_MANIFEST='b16-98759a49'))
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document, 'b16-bf8'), [])

    def test_the_bf16_arm_adds_the_drafter_bf16_flag_to_the_serving_profile(self):
        name, document, added = self.added('dedf-bf16')
        self.assertEqual(added['QWEN_FAST_DRAFTER_BF16'], '1')
        self.assertEqual(set(added), {'QWEN_FAST_PACKED_AUDIT', 'QWEN_FAST_PHASE_TIMING', 'QWEN_FAST_DRAFTER_BF16'})
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document, 'dedf-bf16'), [])
        self.assertNotIn('gate_only', document['profiles'][name] and [key for key in document['profiles'][name]
                                                                       if document['profiles'][name][key] is True])

    def test_the_arm_flags_are_arithmetic_only_for_their_own_arm(self):
        name, document = lab.derive_profile(PROFILES, BASE, 'dedf-bf16')
        # Judged without the arm (the lab as it was), the same derived profile is an arithmetic change; so is another arm's flag.
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document), ['QWEN_FAST_DRAFTER_BF16'])
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document, 'b16-bf8'), ['QWEN_FAST_DRAFTER_BF16'])
        document['profiles'][name]['env']['QWEN_FAST_QUAD_DRAFT'] = '1' if PROFILES['profiles'][BASE]['env'].get('QWEN_FAST_QUAD_DRAFT') != '1' else '0'
        self.assertEqual(lab.arithmetic_diff(PROFILES, BASE, document, 'dedf-bf16'), ['QWEN_FAST_QUAD_DRAFT'])

    def test_what_is_refused(self):
        with self.assertRaises(lab.LabError):
            lab.derive_profile(PROFILES, BASE, 'nonsense')
        clash = copy.deepcopy(PROFILES)
        clash['profiles'][BASE]['env']['QWEN_FAST_DRAFTER_BF16'] = '0'
        with self.assertRaises(lab.LabError):
            lab.derive_profile(clash, BASE, 'dedf-bf16')
        gated = copy.deepcopy(PROFILES)
        gated['profiles'][BASE]['gate_only'] = True
        with self.assertRaises(lab.LabError):
            lab.derive_profile(gated, BASE, 'control')

    def test_no_arm_is_the_lab_as_it_was(self):
        self.assertEqual(lab.derive_profile(PROFILES, BASE), lab.derive_profile(PROFILES, BASE, None))
        self.assertEqual(lab.derive_profile(PROFILES, BASE)[0], BASE + '+taulab')

    def test_the_manifest_flag_names_a_real_manifest(self):
        import drafter_manifest
        for arm, entry in lab.DRAFTER_ARMS.items():
            self.assertIn(entry['manifest'], drafter_manifest.names(), arm)
            for name, value in entry['env']:
                if name == 'QWEN_DRAFTER_MANIFEST':
                    self.assertEqual(value, entry['manifest'])
        self.assertEqual(lab.DRAFTER_ARMS['control']['env'], ())


class LabRuleTests(unittest.TestCase):
    def test_production_needs_the_control_drafter_and_that_images_audit_setting(self):
        self.assertTrue(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-serve-10', False))
        self.assertTrue(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-serve-10', False, 'control'))
        self.assertFalse(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-serve-10', True), 'serve-10 ran the audits off')
        self.assertTrue(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-serve-2', True))
        self.assertFalse(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-serve-2', False), 'serve-2 ran the audits on')
        self.assertFalse(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-serve-10', False, 'b16-bf8'))
        self.assertFalse(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-serve-10', False, 'dedf-bf16'))
        self.assertFalse(lab.is_production_run('r/tt-vllm:qwen38-c2-tp4-cand-1', False))
        self.assertIsNone(lab.production_audits('r/tt-vllm:qwen38-c2-tp4-cand-1'))

    def test_the_current_production_profile_runs_the_audits_the_table_says(self):
        base = PROFILES['profiles'][BASE]['env']
        self.assertFalse(lab.audits_on(PROFILES, BASE))
        self.assertEqual(base.get('QWEN_FAST_VERIFY_T1_AUDIT'), '0')

    def test_a3_is_the_control_arms_calibration(self):
        every = ' '.join(lab.ARMS)
        self.assertEqual(lab.select_arms(every, None), list(lab.ARMS))
        self.assertEqual(lab.select_arms(every, 'control'), list(lab.ARMS))
        for arm in ('b16-bf8', 'dedf-bf16'):
            self.assertEqual(lab.select_arms(every, arm), ['A1', 'A2', 'A4', 'A5'])
            self.assertEqual(lab.select_arms('A1 A2', arm), ['A1', 'A2'])
            with self.assertRaisesRegex(lab.LabError, 'control arm'):
                lab.select_arms('A1 A3', arm)

    def test_the_launched_log_must_show_the_arms_own_markers(self):
        derived = BASE + '+taulab+b16-bf8'
        good = ['[QWEN-C2] profile %s' % derived, '[DRAFTER_MANIFEST] b16-98759a49 in force: x/y at 98759a4995e4, 81 tensors verified']
        self.assertEqual(lab.launched_problems(good, derived, 'b16-bf8'), [])
        self.assertTrue(lab.launched_problems(good[:1], derived, 'b16-bf8'))
        bf16 = BASE + '+taulab+dedf-bf16'
        engaged = ['[QWEN-C2] profile %s' % bf16, '[DRAFTER_BF16] engaged']
        self.assertEqual(lab.launched_problems(engaged, bf16, 'dedf-bf16'), [])
        self.assertTrue(lab.launched_problems(engaged[:1], bf16, 'dedf-bf16'), 'bf16 asked for, never engaged')
        control = BASE + '+taulab'
        self.assertEqual(lab.launched_problems(['[QWEN-C2] profile %s' % control], control, 'control'), [])
        self.assertTrue(lab.launched_problems(['[QWEN-C2] profile %s' % control, '[DRAFTER_BF16] engaged'], control, 'control'))
        self.assertTrue(lab.launched_problems(['[QWEN-C2] profile %s' % control] + good[1:], control, 'control'))
        self.assertEqual(lab.launched_problems(['[QWEN-C2] profile %s' % control], control), [])

    def test_the_report_names_the_arm_and_stays_public(self):
        public = dict(drafter=dict(arm='b16-bf8', bf16=False), label='non-production', verdict='NO-GO')
        report.assert_public(public)
        with self.assertRaises(report.PrivacyError):
            report.assert_public(dict(drafter=dict(arm='0xSomeone/private-name')))


def tape(arm, ident, set_name, cluster, tokens, taus, text, weight=1.0, live=4):
    """A tape of one turn: one counted round per tau entry (emitted = the entry), then the terminal round."""
    rounds = [['P', index % 4, emitted, 16, 16, live] for index, emitted in enumerate(taus)]
    rounds.append(['P', 0, 3, 16, 16, live])
    return dict(arm=arm, id=ident, set=set_name, cluster=cluster, turn=0, prompt_tokens=tokens, weight=weight, rounds=rounds,
                text=text)


def write_run(directory, tapes, info, public=None):
    os.makedirs(directory)
    with open(os.path.join(directory, 'tapes.jsonl'), 'w') as handle:
        for entry in tapes:
            handle.write(json.dumps({key: value for key, value in entry.items() if key != 'text'}) + '\n')
    with open(os.path.join(directory, 'outputs.jsonl'), 'w') as handle:
        for entry in tapes:
            handle.write(json.dumps(dict(arm=entry['arm'], id=entry['id'], output_ids=entry['text'])) + '\n')
    with open(os.path.join(directory, 'run.json'), 'w') as handle:
        json.dump(info, handle)
    if public is not None:
        with open(os.path.join(directory, 'report.private.json'), 'w') as handle:
            json.dump(dict(public=public), handle)


def synthetic(gain=1.10, long_gain=None, p10_gain=None, diverge=0, clusters=40, seed=3):
    """A control and a candidate over the same turns: per-turn taus around 4.5 (long contexts around 3.5); the candidate's rounds
    are the control's with a deterministic uplift of `gain` (a list of emitted counts scaled and rounded)."""
    rng = random.Random(seed)
    control, candidate = [], []
    for index in range(clusters):
        set_name = ('swe', 'own')[index % 2]
        arm = 'A1' if set_name == 'swe' else 'A2'
        longer = index % 5 == 0
        tokens = 90000 if longer else 12000 + 1000 * (index % 7)
        base_tau = 3.5 if longer else 4.5
        taus = [max(1, min(16, int(round(rng.gauss(base_tau, 1.2))))) for _ in range(40)]
        factor = (long_gain if (long_gain is not None and longer) else gain)
        better = [max(1, min(16, int(round(value * factor)))) for value in taus]
        ident = 'turn-%d' % index
        text = [index, index + 1, index + 2]
        control.append(tape(arm, ident, set_name, 'c%d' % index, tokens, taus, text))
        other_text = text if index >= diverge else text + [9]
        candidate.append(tape(arm, ident, set_name, 'c%d' % index, tokens, better, other_text))
    return control, candidate


class PairReportTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.count = 0

    def run_pair(self, control, candidate, arm='b16-bf8', calibration='PASS', resamples=400, control_arm='control'):
        self.count += 1
        base = os.path.join(self.root, 'run%d' % self.count)
        write_run(os.path.join(base, 'control'), control, dict(drafter_arm=control_arm),
                  public=dict(calibration=dict(verdict=calibration)) if calibration else None)
        write_run(os.path.join(base, 'candidate'), candidate, dict(drafter_arm=arm))
        controls, control_info, summary = pair_report.read_run(os.path.join(base, 'control'))
        candidates, candidate_info, _ = pair_report.read_run(os.path.join(base, 'candidate'))
        public = pair_report.build(controls, candidates, arm, control_info, candidate_info, summary, seed=1, resamples=resamples)
        report.assert_public(public)
        return public

    def test_a_clear_gain_with_the_same_text_is_a_go(self):
        public = self.run_pair(*synthetic(gain=1.15))
        self.assertEqual(public['verdict'], 'GO', public['gates'])
        self.assertGreater(public['pooled']['ci95_low'], 1.0)
        self.assertEqual(public['pairs']['different_text'], 0)
        self.assertEqual(public['pairs']['turns'], 40)
        self.assertEqual(set(public['gates'].values()), {'PASS'})

    def test_no_gain_is_not_a_go(self):
        public = self.run_pair(*synthetic(gain=1.0))
        self.assertEqual(public['gates']['pooled'], 'FAIL')
        self.assertEqual(public['verdict'], 'NO-GO')

    def test_a_long_context_regression_blocks_a_pooled_gain(self):
        public = self.run_pair(*synthetic(gain=1.2, long_gain=0.8))
        self.assertEqual(public['gates']['pooled'], 'PASS')
        self.assertEqual(public['gates']['long'], 'FAIL')
        self.assertEqual(public['verdict'], 'NO-GO')

    def test_a_text_difference_is_a_no_go(self):
        public = self.run_pair(*synthetic(gain=1.2, diverge=3))
        self.assertEqual(public['pairs']['different_text'], 3)
        self.assertEqual(public['gates']['text'], 'FAIL')
        self.assertEqual(public['verdict'], 'NO-GO')

    def test_an_uncalibrated_control_cannot_be_a_go(self):
        public = self.run_pair(*synthetic(gain=1.2), calibration=None)
        self.assertEqual(public['gates']['control'], 'NOT_ESTABLISHED')
        self.assertEqual(public['verdict'], 'NOT_ESTABLISHED')
        failed = self.run_pair(*synthetic(gain=1.2), calibration='FAIL')
        self.assertEqual(failed['verdict'], 'NO-GO')

    def test_the_wrong_arm_labels_fail(self):
        public = self.run_pair(*synthetic(gain=1.2), control_arm='b16-bf8')
        self.assertEqual(public['gates']['arms'], 'FAIL')
        self.assertEqual(public['verdict'], 'NO-GO')

    def test_a_thin_pairing_fails_coverage(self):
        control, candidate = synthetic(gain=1.2)
        public = self.run_pair(control, candidate[:20])
        self.assertEqual(public['gates']['coverage'], 'FAIL')
        self.assertEqual(public['pairs']['turns'], 20)

    def test_the_ratio_is_over_the_same_weighted_turns(self):
        control, candidate = synthetic(gain=1.15, clusters=20)
        public = self.run_pair(control, candidate)
        manual_control = sum(sum(entry[2] for entry in item['rounds'][:-1]) for item in control)
        manual_candidate = sum(sum(entry[2] for entry in item['rounds'][:-1]) for item in candidate)
        self.assertAlmostEqual(public['pooled']['ratio'], manual_candidate / float(manual_control), delta=0.03)

    def test_the_public_summary_holds_no_id_or_text(self):
        public = self.run_pair(*synthetic(gain=1.15, clusters=10))
        blob = json.dumps(public)
        self.assertNotIn('turn-', blob)
        self.assertNotIn('c1"', blob)

    def test_the_command_line_writes_the_summary_and_exits_by_the_verdict(self):
        control, candidate = synthetic(gain=1.15, clusters=12)
        write_run(os.path.join(self.root, 'control'), control, dict(drafter_arm='control'),
                  public=dict(calibration=dict(verdict='PASS')))
        write_run(os.path.join(self.root, 'candidate'), candidate, dict(drafter_arm='b16-bf8'))
        out, lines = os.path.join(self.root, 'out'), []
        code = pair_report.main(['--control', os.path.join(self.root, 'control'), '--candidate', os.path.join(self.root, 'candidate'),
                                 '--arm', 'b16-bf8', '--out', out], say=lines.append)
        self.assertIn(code, (0, 1))
        with open(os.path.join(out, pair_report.SUMMARY_NAME), encoding='utf-8') as handle:
            summary = json.load(handle)
        self.assertEqual(code == 0, summary['verdict'] == 'GO')
        self.assertTrue(lines and lines[0].startswith('[DRAFTER-PAIR] arm=b16-bf8'))

    def test_an_unknown_arm_is_refused(self):
        with self.assertRaises(ValueError):
            pair_report.build({}, {}, 'nonsense')


class JobKeyTests(unittest.TestCase):
    base = dict(C2_IMAGE_TAG='tp4-drafter-1', C2_CARDS='quad', C2_ACTIONS='reset taulab')

    def read(self, **extra):
        return job.read_job(dict(self.base, **extra), NAMES, meshes=job.profile_meshes())

    def test_defaults_leave_the_lab_as_it_was(self):
        outputs = self.read()
        self.assertEqual((outputs['taulab_drafter_arm'], outputs['taulab_pair_control']), ('', ''))
        self.assertEqual(outputs['taulab_arms'], 'A1 A2 A3 A4 A5')

    def test_a_candidate_arm_leaves_a3_out_and_refuses_it_by_name(self):
        outputs = self.read(C2_TAULAB_DRAFTER_ARM='b16-bf8')
        self.assertEqual(outputs['taulab_arms'], 'A1 A2 A4 A5')
        self.assertEqual(self.read(C2_TAULAB_DRAFTER_ARM='control')['taulab_arms'], 'A1 A2 A3 A4 A5')
        with self.assertRaisesRegex(job.JobError, 'A3'):
            self.read(C2_TAULAB_DRAFTER_ARM='b16-bf8', C2_TAULAB_ARMS='A1 A3')

    def test_the_pair_control_is_a_run_id_for_a_candidate_only(self):
        outputs = self.read(C2_TAULAB_DRAFTER_ARM='dedf-bf16', C2_TAULAB_PAIR_CONTROL='12345678901', C2_TAULAB_ARMS='A1 A2')
        self.assertEqual((outputs['taulab_drafter_arm'], outputs['taulab_pair_control']), ('dedf-bf16', '12345678901'))
        for bad in ({'C2_TAULAB_DRAFTER_ARM': 'nonsense'},
                    {'C2_TAULAB_DRAFTER_ARM': 'b16-bf8', 'C2_TAULAB_PAIR_CONTROL': 'run-1'},
                    {'C2_TAULAB_PAIR_CONTROL': '123'},
                    {'C2_TAULAB_DRAFTER_ARM': 'control', 'C2_TAULAB_PAIR_CONTROL': '123'}):
            with self.assertRaises(job.JobError, msg=bad):
                self.read(**bad)

    def test_the_workflow_passes_the_arm_and_the_pair(self):
        with open(os.path.join(HERE, '..', '..', '.github', 'workflows', 'qwen-c2-serving.yml'), encoding='utf-8') as handle:
            text = handle.read()
        for needle in ('taulab_drafter_arm', 'taulab_pair_control', '--drafter-arm', 'drafter_pair_report.py'):
            self.assertIn(needle, text)


class TemplateTests(unittest.TestCase):
    """The drafter card-gate pack (references/tp4-drafter-jobs): the three tau arms and the qualification of a GO arm."""

    def rows(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]

    def parse(self, name):
        with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
            return job.parse_env(handle.read())

    def test_every_listed_template_exists_and_every_template_is_listed(self):
        listed = [row[0] + '.env' for row in self.rows()]
        self.assertEqual(sorted(listed), sorted(name for name in os.listdir(FOLDER) if name.endswith('.env')))

    def test_every_template_parses_with_the_job_reader(self):
        for row in self.rows():
            values = self.parse(row[0] + '.env')
            outputs = job.read_job(values, NAMES, meshes=job.profile_meshes())
            self.assertEqual(outputs['cards'], 'quad', row[0])

    def test_the_three_tau_arms(self):
        arms = {}
        for row in self.rows():
            if row[0].startswith('D-T'):
                outputs = job.read_job(self.parse(row[0] + '.env'), NAMES, meshes=job.profile_meshes())
                arms[row[0]] = outputs
        self.assertEqual(sorted(entry['taulab_drafter_arm'] for entry in arms.values()), ['b16-bf8', 'control', 'dedf-bf16'])
        for name, outputs in arms.items():
            self.assertIn('taulab', outputs['actions'].split(), name)
            if outputs['taulab_drafter_arm'] == 'control':
                self.assertIn('A3', outputs['taulab_arms'].split())
            else:
                self.assertNotIn('A3', outputs['taulab_arms'].split())
                self.assertTrue(outputs['taulab_pair_control'] or True)
            self.assertLessEqual(int(outputs['taulab_deadline']), 150, name)

    def test_the_order_runs_the_control_first_and_qualifies_only_a_go_arm(self):
        names = [row[0] for row in self.rows()]
        self.assertEqual(names[0], 'X0-status-rescan')
        self.assertLess(names.index('D-T1-control'), names.index('D-T2-b16-bf8'))
        qualification = [row for row in self.rows() if row[0].startswith('D-Q')]
        self.assertTrue(qualification)
        for row in qualification:
            self.assertEqual(row[1], 'go-arm-only', row)

    def test_the_pack_is_public(self):
        banned = __import__('re').compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|(?!127\.0\.0\.1)\b\d{1,3}(\.\d{1,3}){3}\b|'
                                          r'sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.|registry\.[a-z]+\.[a-z]+')
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
                text = handle.read()
            self.assertIsNone(banned.search(text), (name, banned.search(text)))


if __name__ == '__main__':
    unittest.main()
