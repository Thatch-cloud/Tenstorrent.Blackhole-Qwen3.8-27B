"""The four-card speed window: its job templates (scripts/ci/references/tp4-speed-jobs), their order, and the profiles and smoke
test they use (tp4/speed).

The templates are public, so they name no rig, card, address, registry or digest. Every template parses with c2_serving_job, opens
the cards in ONE step and resets first, uses ONE image tag, the smokes name only tests c2_serving_smoke knows, the smoke and gate
arms serve the speed profiles (each c2-packed-tp4-gate plus its one documented difference), and the exactness arms serve the same
four prompts (the real-text corpus fixes a prompt by user index and length) so E1, E2 and E3 texts can be compared per user."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-speed-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
SMOKES = ('K1-smoke-gate-slide-on', 'K2-smoke-speed', 'K3-smoke-speed-noslide')
MATRICES = ('E1-matrix-slide-on', 'E2-matrix-slide-off', 'E3-g1-tp4-reference')
EXPECTED_PROFILES = {'K0-build': 'c2-packed-tp4-speed', 'K1-smoke-gate-slide-on': 'c2-packed-tp4-gate',
                     'K2-smoke-speed': 'c2-packed-tp4-speed', 'K3-smoke-speed-noslide': 'c2-packed-tp4-speed-noslide',
                     'E1-matrix-slide-on': 'c2-packed-tp4-gate', 'E2-matrix-slide-off': 'c2-packed-tp4-gate-noslide',
                     'E3-g1-tp4-reference': 'general-tp4'}


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [tuple(line.split()) for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    values = job.parse_env(text_of(name))
    return values, job.read_job(values, NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_and_the_order_names_no_other(self):
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [name for name, _ in read_order()]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(len(set(ordered)), len(ordered))

    def test_the_stages_build_then_smokes_then_exactness(self):
        order = read_order()
        self.assertEqual([name.split('-')[0] for name, _ in order], ['K0', 'K1', 'K2', 'K3', 'E1', 'E2', 'E3'])
        self.assertEqual([name for name, mode in order if mode == 'soft'], ['K3-smoke-speed-noslide', 'E3-g1-tp4-reference'])
        self.assertEqual(set(mode for _, mode in order), {'stop', 'soft'})
        self.assertEqual(order[0], ('K0-build', 'stop'))

    def test_every_template_parses_names_no_host_and_uses_the_one_image_tag(self):
        for name, _ in read_order():
            with self.subTest(template=name):
                values, outputs = parsed(name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
                self.assertEqual(outputs['tag'], 'tp4-speed-1')
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(outputs['profile'], EXPECTED_PROFILES[name])
                self.assertEqual(set(re.findall(r'@[A-Z0-9_]+@', text_of(name))), set(), 'no placeholder: no one-card job here')

    def test_the_build_opens_no_card_and_a_run_resets_first_and_opens_the_cards_once(self):
        for name, _ in read_order():
            _, outputs = parsed(name)
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                if name.startswith('K0'):
                    self.assertEqual(actions, ['status', 'reset', 'build'])
                    continue
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card run follows a reset')


class SmokeTests(unittest.TestCase):
    def source(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            return handle.read()

    def test_the_smokes_name_tests_the_smoke_knows_and_run_the_same_four(self):
        smoke = self.source()
        tests = set()
        for name in SMOKES:
            _, outputs = parsed(name)
            self.assertEqual(outputs['actions'].split(), ['reset', 'smoke'], name)
            listed = outputs['tests'].split(',')
            for test in listed:
                self.assertIn("record('%s'" % test, smoke, (name, test))
            tests.add(tuple(listed))
        self.assertEqual(tests, {('warmup', 'warm_lifecycle', 'coding', 'concurrent4')}, 'the A/B smokes run the same tests')

    def test_the_lifecycle_warm_runs_before_coding_and_drives_every_prefill_size_twice(self):
        smoke = self.source()
        self.assertLess(smoke.index("record('warm_lifecycle'"), smoke.index("record('coding'"))
        self.assertIn("for label in ('cold', 'warm'):", smoke)
        self.assertIn('sizes = (600, 1800, 3600, 9000)', smoke)
        compile(smoke, 'c2_serving_smoke.py', 'exec')   # the rig host's python 3.7 takes the same syntax: no walrus, no match


class ProfileTests(unittest.TestCase):
    def test_the_window_serves_the_speed_profiles_and_the_g1_reference(self):
        served = {parsed(name)[1]['profile'] for name, _ in read_order()}
        self.assertEqual(served, {'c2-packed-tp4-speed', 'c2-packed-tp4-gate', 'c2-packed-tp4-speed-noslide',
                                  'c2-packed-tp4-gate-noslide', 'general-tp4'})
        for name in served:
            self.assertEqual(PROFILES['profiles'][name]['mesh_device'], 'P150x4', name)

    def test_the_timed_smokes_have_the_audits_off_and_the_first_attach_has_them_on(self):
        env = lambda job_name: PROFILES['profiles'][parsed(job_name)[1]['profile']]['env']
        self.assertEqual((env('K1-smoke-gate-slide-on')['QWEN_FAST_VERIFY_T1_AUDIT'], env('K1-smoke-gate-slide-on')['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
        for name in ('K2-smoke-speed', 'K3-smoke-speed-noslide'):
            self.assertEqual((env(name)['QWEN_FAST_VERIFY_T1_AUDIT'], env(name)['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), name)
        self.assertEqual([env(name)['QWEN_FAST_TP_KV_SLIDE'] for name in ('K1-smoke-gate-slide-on', 'K2-smoke-speed', 'K3-smoke-speed-noslide')],
                         ['1', '1', '0'])
        self.assertEqual([env(name)['QWEN_FAST_TP_KV_SLIDE'] for name in ('E1-matrix-slide-on', 'E2-matrix-slide-off')], ['1', '0'])


class MatrixTests(unittest.TestCase):
    def test_the_exactness_arms_serve_the_same_four_prompts(self):
        found = {parsed(name)[1]['gate_lengths'] for name in MATRICES}
        self.assertEqual(found, {'512,1536,4096,16384'}, 'user index and length fix the prompt: the texts compare per user')
        self.assertEqual({parsed(name)[1]['gate_max_tokens'] for name in MATRICES}, {'256'})
        for name in MATRICES:
            self.assertEqual(parsed(name)[1]['gate_plan'], 'matrix')
        # two ramp users (prompts under the 2,048-row draft window), two at the steady state: the slide is exercised in both
        lengths = [int(part) for part in parsed('E1-matrix-slide-on')[1]['gate_lengths'].split(',')]
        self.assertEqual([length < 2048 for length in lengths], [True, True, False, False])

    def test_the_fast_path_arms_record_the_kernel_cache_and_keep_the_pair_only_audits_off(self):
        for name in ('E1-matrix-slide-on', 'E2-matrix-slide-off'):
            _, outputs = parsed(name)
            self.assertEqual(outputs['gate_jit'], 'record', name)
            self.assertEqual(outputs['gate_audits'], '', 'the prestage and fused-commit audits are not ported to four cards')

    def test_the_gate_plans_are_accepted_on_their_profiles_and_fit_the_step(self):
        for name in MATRICES:
            _, outputs = parsed(name)
            lengths = [int(part) for part in outputs['gate_lengths'].split(',')]
            arms = gate.plan_arms('matrix', outputs['profile'], PROFILES, lengths, int(outputs['gate_max_tokens']), None, [], {})
            self.assertEqual([arm[0] for arm in arms], ['matrix-concurrent', 'matrix-solo'], name)
            self.assertEqual(len(lengths), 4, 'four users, four seats: the concurrent arm is a packed-4 round')
            worst = 2 * sum(arm[2] + gate.ARM_OVERHEAD_SECONDS for arm in arms)
            self.assertLessEqual(worst, 380 * 60 - 600, name)

    def test_a_gate_only_profile_is_booted_with_the_gate_switch_in_the_dry_run_argv(self):
        for name in MATRICES:
            _, outputs = parsed(name)
            lines = []
            code = gate.main(['--image', 'img', '--profile', outputs['profile'], '--plan', 'matrix', '--cards', 'quad',
                              '--dry-run', '--results', os.path.join(HERE, 'no-results'), '--profiles',
                              os.path.join(HERE, 'qwen_c2_profiles.json'), '--lengths', outputs['gate_lengths'],
                              '--max-tokens', outputs['gate_max_tokens'], '--jit', outputs['gate_jit'] or 'auto'],
                             log=lines.append)
            self.assertEqual(code, 0, name)
            arms = [json.loads(line)['docker'] for line in lines[1:]]
            self.assertTrue(arms, name)
            wanted = PROFILES['profiles'][outputs['profile']].get('gate_only') is True
            for argv in arms:
                self.assertEqual('QWEN_C2_GATE=1' in argv, wanted, name)
                # the container reads its env from the named profile of the image: the launched argv must name it
                self.assertIn('QWEN_C2_PROFILE=%s' % outputs['profile'], argv, name)


if __name__ == '__main__':
    unittest.main()
