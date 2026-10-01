"""The four-card fused-commit window: its job templates (scripts/ci/references/tp4-fcommit-jobs), their order, and the profiles they serve
(tp4/fcommit).

The templates are public, so they name no rig, card, address, registry or digest. Every template parses with c2_serving_job, uses ONE image
tag, opens the cards once (the one-card harness steps need their own tag and run on the pair form), resets first when it opens four cards, and
serves the fused-commit profiles: the audited smokes the audited profiles (audits on, never timed), the timed arms the timed ones (audits
off), the timed arms alternate eager and fused twice per shape, and the fallback is kept out of the order."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-fcommit-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
NAMES = sorted(PROFILES)
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
EXPECTED = {'F0-build': 'c2-packed-tp4-speed-fcommit-quad',
            'F1a-cardm-watcher': None, 'F1b-cardm-full': None,
            'F2-fcommit-smoke': 'c2-packed-tp4-gate-fcommit',
            'F3-fcommit-live-smoke': 'c2-packed-tp4-gate-fcommit-live',
            'F4-fcommit-quad-smoke': 'c2-packed-tp4-gate-fcommit-quad',
            'F5a-quad-eager-timed': 'c2-packed-tp4-speed-quad', 'F5b-quad-fused-timed': 'c2-packed-tp4-speed-fcommit-quad',
            'F5c-quad-eager-timed': 'c2-packed-tp4-speed-quad', 'F5d-quad-fused-timed': 'c2-packed-tp4-speed-fcommit-quad',
            'F5e-pairs-eager-timed': 'c2-packed-tp4-speed-pairs', 'F5f-pairs-fused-timed': 'c2-packed-tp4-speed-fcommit',
            'F5g-pairs-eager-timed': 'c2-packed-tp4-speed-pairs', 'F5h-pairs-fused-timed': 'c2-packed-tp4-speed-fcommit',
            'F5x-oop-timed': 'c2-packed-tp4-speed-fcommit-oop',
            'F6-fcommit-matrix': 'c2-packed-tp4-gate-fcommit-quad'}
FUSED, INPLACE, LIVE, AUDIT = ('QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_FUSED_COMMIT_INPLACE', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS',
                               'QWEN_FAST_FUSED_COMMIT_AUDIT')


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
    def test_every_template_is_in_the_order_once_except_the_fallback(self):
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [name for name, _ in read_order()]
        self.assertEqual(len(set(ordered)), len(ordered))
        self.assertEqual(sorted(ordered + ['F5x-oop-timed']), on_disk)
        self.assertEqual(sorted(on_disk), sorted(EXPECTED))

    def test_build_then_one_card_then_audited_smokes_then_timed_arms_then_the_matrix(self):
        order = read_order()
        self.assertEqual([name.split('-')[0] for name, _ in order],
                         ['F0', 'F1a', 'F1b', 'F2', 'F3', 'F4'] + ['F5%s' % c for c in 'abcdefgh'] + ['F6'])
        self.assertEqual([name for name, mode in order if mode == 'stop'],
                         ['F0-build', 'F1a-cardm-watcher', 'F1b-cardm-full', 'F2-fcommit-smoke', 'F3-fcommit-live-smoke'])
        self.assertEqual(order[-1], ('F6-fcommit-matrix', 'soft'), 'the matrix is soft and last')
        self.assertEqual(dict(order)['F4-fcommit-quad-smoke'], 'soft', 'a quad failure must not halt the pairs arms')

    def test_the_timed_arms_alternate_eager_and_fused_twice_per_shape(self):
        timed = [EXPECTED[name] for name, _ in read_order() if name.startswith('F5')]
        self.assertEqual(timed, ['c2-packed-tp4-speed-quad', 'c2-packed-tp4-speed-fcommit-quad'] * 2
                         + ['c2-packed-tp4-speed-pairs', 'c2-packed-tp4-speed-fcommit'] * 2)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_names_no_host_and_uses_the_one_image_tag(self):
        for name in EXPECTED:
            with self.subTest(template=name):
                values, outputs = parsed(name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
                self.assertEqual(outputs['tag'], 'tp4-fcommit-1')
                self.assertEqual(set(re.findall(r'@[A-Z0-9_]+@', text_of(name))), set())
        self.assertIn('PLACEHOLDER', text_of('F0-build'))
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            self.assertIsNone(BANNED.search(handle.read()))

    def test_four_card_steps_reset_first_and_open_the_cards_once_the_build_opens_none(self):
        for name, profile in EXPECTED.items():
            _, outputs = parsed(name)
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                if name == 'F0-build':
                    self.assertEqual(actions, ['status', 'reset', 'build'])
                elif name.startswith('F1'):
                    self.assertEqual(actions, ['cardm'])
                    self.assertEqual(outputs['cards'], 'pair', 'cardm cannot run with the four cards held')
                else:
                    self.assertEqual(outputs['cards'], 'quad')
                    self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                    self.assertEqual(actions[0], 'reset')

    def test_every_template_states_its_duration_and_its_stop_rule(self):
        modes = dict(read_order())
        modes['F5x-oop-timed'] = 'soft'
        for name, mode in modes.items():
            text = text_of(name)
            with self.subTest(template=name):
                self.assertRegex(text, r'# Duration \(estimate')
                self.assertIn(mode.upper(), text)

    def test_the_one_card_harness_runs_at_two_heads_and_the_full_pass_traces_the_80_worker_layout(self):
        for name in ('F1a-cardm-watcher', 'F1b-cardm-full'):
            values, outputs = parsed(name)
            self.assertEqual(values['C2_CARDM_HARNESS'], 'optimisation/ttnn-op/draft_slide_inplace/run_card_b.sh')
            self.assertIn('--heads 2', values['C2_CARDM_ARGS'])
        self.assertIn('WATCHER=1', parsed('F1a-cardm-watcher')[0]['C2_CARDM_ENV'])
        self.assertNotIn('C2_CARDM_ENV', parsed('F1b-cardm-full')[0])
        self.assertIn('--trace-layouts 1,2,5,10', parsed('F1b-cardm-full')[0]['C2_CARDM_ARGS'])
        self.assertTrue(os.path.isfile(os.path.join(HERE, '..', '..', 'optimisation', 'ttnn-op', 'draft_slide_inplace', 'run_card_b.sh')))


class ProfileTests(unittest.TestCase):
    def env(self, name):
        return PROFILES[EXPECTED[name]]['env']

    def test_the_audited_smokes_carry_the_audit_and_the_timed_arms_never_do(self):
        for name in EXPECTED:
            if EXPECTED[name] is None or name == 'F0-build':
                continue
            env = self.env(name)
            with self.subTest(template=name):
                if name.startswith(('F2', 'F3', 'F4', 'F6')):
                    self.assertEqual(env[AUDIT], '1')
                    self.assertEqual((env[FUSED], env[INPLACE]), ('1', '1'))
                if name.startswith('F5'):
                    self.assertEqual(env.get(AUDIT, '0'), '0')
                    self.assertEqual(env.get('QWEN_FAST_DRAFT_SINGLES_AUDIT', '0'), '0')

    def test_the_fused_arms_of_the_timed_pairs_carry_fused_inplace_and_live_banks_and_the_eager_arms_none(self):
        for name in EXPECTED:
            if not name.startswith('F5') or name.startswith('F5x'):
                continue
            env = self.env(name)
            fused = 'fused' in name
            with self.subTest(template=name):
                self.assertEqual([env.get(flag, '0') for flag in (FUSED, INPLACE, LIVE)], ['1'] * 3 if fused else ['0'] * 3)
                self.assertEqual(env.get('QWEN_FAST_QUAD_DRAFT', '0'), '1' if 'quad' in name else '0')

    def test_the_fallback_is_the_projection_trace_alone(self):
        env = self.env('F5x-oop-timed')
        self.assertEqual([env.get(flag, '0') for flag in (FUSED, INPLACE, LIVE, AUDIT)], ['1', '0', '0', '0'])

    def test_the_quad_arms_hold_the_quad_and_the_live_arms_need_the_inplace_slides(self):
        self.assertEqual(self.env('F4-fcommit-quad-smoke')['QWEN_FAST_QUAD_DRAFT'], '1')
        self.assertEqual(self.env('F3-fcommit-live-smoke').get('QWEN_FAST_QUAD_DRAFT', '0'), '0')
        for name in EXPECTED:
            if EXPECTED[name] and self.env(name).get(LIVE) == '1':
                self.assertEqual(self.env(name)[INPLACE], '1', name)
                self.assertEqual(self.env(name)[FUSED], '1', name)


if __name__ == '__main__':
    unittest.main()
