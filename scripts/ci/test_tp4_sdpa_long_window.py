"""The tp4/sdpa-long window: its job templates (scripts/ci/references/tp4-sdpa-long-jobs), their order, and the sweep harness they run.

One sweep job (Q1, cardm on one card, optional) between the window's two bookends: A0 takes production down, H1 hands the cards back. No
build job: the harness mounts its scripts from the checkout and the image already on the rig carries the served SDPA graft. The templates
are public, so they name no rig, card, address, registry or digest.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_sdpa_long_window` from scripts/ci.
"""

import json
import os
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_tp4_long'))

import c2_serving_job as job  # noqa: E402
import sdpa_long_tp  # noqa: E402
import sdpa_tp4_long  # noqa: E402

FOLDER = HERE / 'references' / 'tp4-sdpa-long-jobs'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
NAMES = sorted(PROFILES['profiles'])
IMAGE = 'tp4-next-3a'
FIRST, SWEEP, HANDBACK = 'A0-agentstop-unserve', 'Q1-sdpa-long-sweep', 'H1-handback-reset-fabric-agentstart'
ORDERED = (FIRST, SWEEP, HANDBACK)
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')


def read_order():
    return [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]


def text_of(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(path.name[:-4] for path in FOLDER.iterdir() if path.name.endswith('.env'))
        self.assertEqual(sorted(row[0] for row in rows), on_disk)
        self.assertEqual([row[0] for row in rows], list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        modes = {}
        for name, mode, image, minutes in read_order():
            modes[name] = mode
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180, minutes)
        self.assertEqual(modes, {FIRST: 'stop', SWEEP: 'optional', HANDBACK: 'stop'})

    def test_no_build_job_and_the_order_says_why(self):
        for name in ORDERED:
            self.assertNotIn('build', parsed(name)['actions'].split(), name)
        text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('NO build job (B0) is needed', 'mounts no graft', 'carries the K64j op graft', 'tp4-next-3a'):
            self.assertIn(word, text)

    def test_the_first_job_takes_production_down_and_nothing_else_does(self):
        self.assertEqual(parsed(FIRST)['actions'], 'status agentstop unserve')
        self.assertEqual(parsed(FIRST)['cards'], 'quad')
        for name in ORDERED:
            if name != FIRST:
                self.assertFalse({'agentstop', 'unserve'} & set(parsed(name)['actions'].split()), name)
        self.assertIn('PRODUCTION IS LIVE ON THE CARDS', (FOLDER / 'ORDER.txt').read_text(encoding='utf-8'))

    def test_the_hand_back_is_last_resets_remeasures_the_fabric_restarts_the_agent_and_never_deploys(self):
        self.assertEqual(read_order()[-1][0], HANDBACK)
        self.assertEqual(parsed(HANDBACK)['actions'].split(), ['status', 'reset', 'fabric', 'agentstart'])
        self.assertEqual(parsed(HANDBACK)['cards'], 'quad')
        for word in ("TODAY'S PRODUCTION RECIPE", 'NEVER place a gate arm', 'AN OWNER /deploy IS STILL NEEDED', 'fabric re-measure',
                     'stops before agentstart', 'c2-packed-tp4-best-sdpa'):
            self.assertIn(word, text_of(HANDBACK))
        for name in ORDERED:
            self.assertNotIn('push', parsed(name)['actions'].split(), 'no :latest retag, no publish')
            self.assertFalse(re.search(r'(?im)^C2_(PLACE|DEPLOY)', text_of(name)), name)
        self.assertIn('runs LAST and even when any earlier job failed or hung', (FOLDER / 'ORDER.txt').read_text(encoding='utf-8'))

    def test_the_order_says_what_comes_after_and_that_nothing_is_applied(self):
        text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('WHAT COMES AFTER', 'c2-packed-tp4-best-sdpa', 'PAIRED per round', 'ABAB', 'tp4-next-3a does not carry it',
                     'it applies nothing'):
            self.assertIn(word, text)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_or_digest(self):
        for name in ORDERED:
            with self.subTest(template=name):
                parsed(name)
                self.assertNotIn(b'\r', (FOLDER / (name + '.env')).read_bytes(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        self.assertNotIn(b'\r', (FOLDER / 'ORDER.txt').read_bytes())
        self.assertIsNone(BANNED.search((FOLDER / 'ORDER.txt').read_text(encoding='utf-8')))

    def test_only_the_sweep_is_a_one_card_job_and_it_is_the_only_device_step(self):
        self.assertEqual([name for name in ORDERED if parsed(name)['cards'] == 'pair'], [SWEEP])
        for name in ORDERED:
            steps = set(parsed(name)['actions'].split()) & set(DEVICE_STEPS)
            self.assertEqual(steps, {'cardm'} if name == SWEEP else (set() if name == FIRST else {'fabric'}), name)

    def test_the_sweep_runs_the_sdpa_harness_with_the_windows_tag_and_arguments_the_sweep_parses(self):
        outputs = parsed(SWEEP)
        self.assertEqual(outputs['actions'], 'cardm')
        self.assertEqual(outputs['cardm_harness'], 'optimisation/ttnn-op/sdpa_tp4_long/run_card_m.sh')
        self.assertEqual(outputs['cardm_env'], 'IMAGE_TAG=%s' % IMAGE)
        self.assertTrue((ROOT / outputs['cardm_harness']).is_file())
        args = sdpa_tp4_long.parse_args(['--out', 'x.json'] + outputs['cardm_args'].split())
        self.assertEqual(args.users, 4)
        self.assertEqual(args.arms, list(sdpa_tp4_long.ARMS), 'every arm runs')
        self.assertEqual(args.extents, [33024, 65792, 131328, 262400])
        for word in ('NOTHING IS APPLIED', 'SERVED SDPA GRAFT', 'OPTIONAL', 'Never cancel', 'differing_rows', 'winners',
                     'c2-packed-tp4-best-sdpa'):
            self.assertIn(word, text_of(SWEEP))

    def test_the_sweep_pins_nothing_it_cannot_say_publicly(self):
        text = text_of(SWEEP)
        self.assertNotIn('EXPECT_TTNNCPP_SHA256=', text)
        self.assertNotIn('KOPGRAFT', text)
        self.assertNotIn('QUAL_', outputs_env(SWEEP))


def outputs_env(name):
    return parsed(name)['cardm_env']


class ProfileTests(unittest.TestCase):
    def test_the_gate_only_twin_exists_and_names_a_servable_configuration(self):
        body = PROFILES['profiles']['c2-packed-tp4-best-sdpa']
        self.assertIs(body['gate_only'], True)
        self.assertIn(body['env'][sdpa_long_tp.FLAG], sdpa_long_tp.servable_names())
        self.assertEqual(sorted(sdpa_tp4_long.ARMS), sorted(sdpa_long_tp.names()))


if __name__ == '__main__':
    unittest.main()
