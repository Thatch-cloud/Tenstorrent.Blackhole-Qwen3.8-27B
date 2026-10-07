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
ORDERED = ('W1-cardm-watcher', 'W2-cardm-full', 'W0-cardm-baseline')   # W0 is filler: last
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
        self.assertIn('# REQUIRES-FILE W1-cardm-watcher W2-cardm-full W0-cardm-baseline <- scripts/ci/gdn_wy_block.py', text)
        self.assertIn('# NEEDS W2-cardm-full <- W1-cardm-watcher', text)
        self.assertIn('NON-EXACT RESEARCH ONLY', text)
        self.assertNotIn('agentstart', text.replace('no agent stop or start', ''))

    def test_needs_lines_name_jobs_only_and_file_conditions_have_their_own_key(self):
        # the established convention (tp4-262k8-*-jobs/ORDER.txt): '# NEEDS <jobs> <- <jobs>' with job names on both sides
        needs = [line for line in order_text().splitlines() if line.startswith('# NEEDS ')]
        self.assertTrue(needs)
        for line in needs:
            left, right = line[len('# NEEDS '):].split(' <- ')
            for name in left.split() + right.split():
                self.assertIn(name, ORDERED, line)
        files = [line for line in order_text().splitlines() if line.startswith('# REQUIRES-FILE ')]
        self.assertEqual(len(files), 1)
        left, right = files[0][len('# REQUIRES-FILE '):].split(' <- ')
        self.assertEqual(sorted(left.split()), sorted(ORDERED))
        self.assertTrue(right.split()[0].endswith('gdn_wy_block.py'))
        for name in ('W0-cardm-baseline', 'W1-cardm-watcher', 'W2-cardm-full'):
            self.assertIn('REQUIRES-FILE', text_of(name))

    def test_w0_is_ten_minutes_of_filler_that_gates_nothing(self):
        text = text_of('W0-cardm-baseline')
        values = job.parse_env(text)
        self.assertEqual(values['C2_CARDM_ARGS'], '--sections selftest,timing --timing-arms A,A2')
        self.assertIn('FILLER', text)
        self.assertIn('kernel=present', text)    # the READ rule names what the harness prints once the kernel exists
        self.assertNotIn('before W1', text)       # an optional job cannot gate a stop job
        rows = {row[0]: row for row in read_order()}
        self.assertEqual(rows['W0-cardm-baseline'][3], '10')
        self.assertEqual([row[0] for row in read_order()][-1], 'W0-cardm-baseline')
        for name in ('W1-cardm-watcher', 'W2-cardm-full'):
            self.assertNotIn('W0', ' '.join(line for line in order_text().splitlines() if line.startswith('# NEEDS ') and name in line))

    def test_every_estimate_leaves_the_container_timeout_at_least_one_and_a_half_times_over(self):
        with open(os.path.join(ROOT, *HARNESS.split('/')), encoding='utf-8') as handle:
            timeout = int(re.search(r'^timeout_s=(\d+)', handle.read(), flags=re.M).group(1))
        for name, mode, image, minutes in read_order():
            self.assertGreaterEqual(timeout, int(minutes) * 60 * 1.5, name)

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
