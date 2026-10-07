"""The window-WY probe's card job templates (scripts/ci/references/tp4-wy-probe-jobs) and their order (docs/gdn-wy-probe.md).

NON-EXACT research probe: the templates run the one-card harness only (the cardm step), never a serving profile, never a deploy. The
templates are public, so they name no rig, card, address, registry or digest.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(ROOT)
FOLDER = os.path.join(HERE, 'references', 'tp4-wy-probe-jobs')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-wy-probe-1'
ORDERED = ('W0-cardm-baseline', 'W1-cardm-watcher', 'W2-cardm-full')
HARNESS = 'optimisation/ttnn-op/wy_probe/run_card_m.sh'
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    NAMES = sorted(json.load(_handle)['profiles'])


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def read_order():
    return [line.split() for line in order_text().splitlines() if line.strip() and not line.startswith('#')]


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


class WindowTests(unittest.TestCase):
    def test_the_order_lists_every_template_once_with_four_columns(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual([row[0] for row in rows], list(ORDERED))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(sorted(ORDERED), on_disk)

    def test_modes_images_and_minutes(self):
        modes = {}
        for name, mode, image, minutes in read_order():
            modes[name] = mode
            self.assertIn(mode, ('stop', 'optional'))
            self.assertEqual(image, IMAGE)
            self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180, minutes)
        self.assertEqual(modes['W0-cardm-baseline'], 'optional')
        self.assertEqual(modes['W1-cardm-watcher'], 'stop')
        self.assertEqual(modes['W2-cardm-full'], 'stop')

    def test_the_order_states_the_kernel_dependency_and_that_nothing_licenses_serving(self):
        text = order_text()
        self.assertIn('# NEEDS W1-cardm-watcher W2-cardm-full <- scripts/ci/gdn_wy_block.py exists', text)
        self.assertIn('# NEEDS W2-cardm-full <- W1-cardm-watcher', text)
        self.assertIn('NON-EXACT RESEARCH ONLY', text)
        self.assertNotIn('agentstart', text.replace('no agent stop or start', ''))

    def test_every_template_is_a_cardm_step_on_the_probe_harness_only(self):
        for name in ORDERED:
            with self.subTest(job=name):
                text = text_of(name)
                self.assertNotIn('\r', text)
                self.assertIsNone(BANNED.search(text))
                self.assertIn('NON-EXACT RESEARCH PROBE', text)
                values = job.parse_env(text)
                self.assertEqual(values['C2_ACTIONS'], 'cardm')
                self.assertEqual(values['C2_CARDM_HARNESS'], HARNESS)
                self.assertEqual(values['C2_IMAGE_TAG'], IMAGE)
                self.assertNotIn('C2_PROFILE', values)   # no serving profile is named: the probe never serves
                parsed(name)

    def test_the_harness_the_templates_name_exists_and_embeds_the_card_library(self):
        path = os.path.join(ROOT, *HARNESS.split('/'))
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        with open(os.path.join(HERE, 'qual_card.sh'), encoding='utf-8') as handle:
            library = handle.read()
        self.assertTrue(job.embeds_qual_card(text, library))

    def test_arguments_match_what_the_harness_accepts(self):
        sys.path[:0] = [os.path.join(ROOT, 'optimisation', 'ttnn-op', 'wy_probe'), os.path.join(ROOT, 'optimisation', 'ttnn-op', 'v5split')]
        import gdn_wy_card_m as probe
        for name in ORDERED:
            values = job.parse_env(text_of(name))
            args = values.get('C2_CARDM_ARGS', '').split()
            arguments = probe.parse(['--out', 'x.json'] + args)
            if name == 'W2-cardm-full':
                self.assertEqual(probe.scope_missing(arguments), [])
            else:
                self.assertTrue(probe.scope_missing(arguments))   # W0 and W1 are reduced scope on purpose

    def test_the_watcher_is_set_only_for_the_watcher_job(self):
        for name in ORDERED:
            env = job.parse_env(text_of(name)).get('C2_CARDM_ENV', '')
            self.assertEqual('WATCHER=1' in env.split(), name == 'W1-cardm-watcher')


if __name__ == '__main__':
    unittest.main()
