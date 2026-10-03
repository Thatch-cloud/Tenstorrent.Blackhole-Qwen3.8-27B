"""tp4/sdpa-multi: the one-launch SDPA window's job pack (references/tp4-sdpa-multi-jobs).

Every template parses with the job parser (`py -3.11 -B scripts/ci/c2_serving_job.py <env>` is what the driver runs), names the one image, never touches the node
agent (no agentstop, agentstart, unserve, platform, cardm), names no production tag, bakes nothing, and the ORDER.txt lines match the files. The first quad job is
`status rescan reset`. The audited attach is the exactness gate (stop mode) and the ABAB pair differs in the SDPA flag alone. The templates are public, so they name
no rig, card, address, registry or digest.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_sdpa_multi_window` from scripts/ci.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import sdpa_long_tp  # noqa: E402
import sdpa_multi_tp  # noqa: E402

FOLDER = HERE / 'references' / 'tp4-sdpa-multi-jobs'
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
NAMES = sorted(PROFILES['profiles'])
IMAGE = 'tp4-sdpa-multi-1'
CONTROL, MULTI, AUDITED = ('c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-sdpamulti',
                           'c2-packed-tp4-8x262k-best-sdpamulti-audit')
EXPECTED = {
    'X0-status-rescan-reset': ('status rescan reset', None, 'stop'),
    'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    'A1-audited-attach-smoke': ('reset smoke', AUDITED, 'stop'),
    'T1-timed-A-control-8x262k-time-gate': ('reset smoke', CONTROL, 'soft'),
    'T2-timed-B-multi-8x262k-time-gate': ('reset smoke', MULTI, 'soft'),
    'T3-timed-A-control-8x262k-time-gate': ('reset smoke', CONTROL, 'soft'),
    'T4-timed-B-multi-8x262k-time-gate': ('reset smoke', MULTI, 'soft'),
    'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')


def text_of(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES, root=ROOT)


def order_lines():
    return [line.split() for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')]


def needs_lines():
    found = {}
    for line in (FOLDER / 'ORDER.txt').read_text(encoding='utf-8').splitlines():
        match = re.match(r'# NEEDS (.+) <- (.+)$', line)
        if match:
            for name in match.group(1).split():
                found.setdefault(name, set()).update(match.group(2).split())
    return found


class PackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        self.assertEqual([line[0] for line in lines], list(EXPECTED))
        self.assertEqual(sorted(path.name[:-4] for path in FOLDER.iterdir() if path.name.endswith('.env')), sorted(EXPECTED))
        for name, mode, image, minutes in lines:
            self.assertEqual((mode, image), (EXPECTED[name][2], IMAGE), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_with_the_expected_actions_and_profile(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            with self.subTest(name):
                result = parsed(name)
                self.assertEqual(result['actions'], actions)
                self.assertEqual(result['cards'], 'quad')
                self.assertEqual(result['tag'], IMAGE)
                if profile:
                    self.assertEqual(result['profile'], profile)

    def test_every_template_passes_the_job_parser_the_driver_runs(self):
        for name in EXPECTED:
            with self.subTest(name):
                completed = subprocess.run([sys.executable, '-B', str(HERE / 'c2_serving_job.py'), str(FOLDER / (name + '.env'))],
                                           capture_output=True, cwd=str(ROOT))
                self.assertEqual(completed.returncode, 0, completed.stderr.decode('utf-8', 'replace')[-400:])

    def test_the_window_never_touches_the_node_agent_or_production(self):
        for name in EXPECTED:
            with self.subTest(name):
                result = parsed(name)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
                self.assertNotRegex(text_of(name), r'(?im)^C2_(PLACE|DEPLOY)')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('NO agentstop, NO agentstart, NO hand-back', 'NO deploy', 'NO :latest move'):
            self.assertIn(word, order)

    def test_the_first_quad_job_is_status_rescan_reset_and_the_window_ends_with_a_reset(self):
        self.assertEqual(order_lines()[0][0], 'X0-status-rescan-reset')
        self.assertEqual(parsed('X0-status-rescan-reset')['actions'], 'status rescan reset')
        self.assertEqual(order_lines()[-1][0], 'Z-reset')
        self.assertEqual(PROFILES['default'], 'c2-packed-tp4')

    def test_the_audited_attach_is_a_stop_gate_and_the_timing_waits_for_it(self):
        self.assertEqual(EXPECTED['A1-audited-attach-smoke'][2], 'stop')
        needs = needs_lines()
        self.assertEqual(needs['A1'], {'B0'})
        for name in ('T1', 'T2', 'T3', 'T4'):
            self.assertEqual(needs[name], {'A1'})
        tests = parsed('A1-audited-attach-smoke')['tests'].split(',')
        for shape in ('concurrent8_code_equal', 'concurrent8_steady', 'concurrent8_code'):
            self.assertIn(shape, tests)

    def test_the_pair_is_abab_at_eight_users_32k_and_128k_and_differs_in_the_profile_alone(self):
        names = [name for name in EXPECTED if name.startswith('T')]
        self.assertEqual([parsed(name)['profile'] for name in names], [CONTROL, MULTI, CONTROL, MULTI])
        reference = parsed(names[0])
        for name in names:
            result = parsed(name)
            self.assertEqual(result['tests'], 'warmup,coding,concurrent8_code_32k,concurrent8_code_128k')
            self.assertEqual({key: value for key, value in result.items() if key != 'profile'},
                             {key: value for key, value in reference.items() if key != 'profile'}, name)

    def test_the_arms_are_the_gate_only_profiles_and_the_audited_one_is_never_timed(self):
        for name in (CONTROL, MULTI, AUDITED):
            self.assertIs(PROFILES['profiles'][name]['gate_only'], True, name)
        for name in EXPECTED:
            if name.startswith('T'):
                self.assertNotEqual(parsed(name)['profile'], AUDITED)
        control, multi = PROFILES['profiles'][CONTROL]['env'], PROFILES['profiles'][MULTI]['env']
        self.assertEqual(set(multi) - set(control), {sdpa_long_tp.FLAG})
        self.assertEqual(multi[sdpa_long_tp.FLAG], 'multi')

    def test_the_order_says_what_it_decides_and_what_fails_it(self):
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('WHAT THIS WINDOW DECIDES', 'byte-identical', 'PAIRED per round', 'ABAB', 'NO-GO', 'any audit mismatch', 'UNQUALIFIED',
                     'qwen-cpu-suite.yml', 'exact=True'):
            self.assertIn(word, order)
        self.assertIn('audit MISMATCH', text_of('A1-audited-attach-smoke'))
        self.assertIn('scratch_slots=4', text_of('A1-audited-attach-smoke'))
        self.assertIn('0x21', text_of('A1-audited-attach-smoke'))

    def test_the_build_job_names_the_multi_files_the_image_must_carry(self):
        text = text_of('B0-build')
        for name in sdpa_multi_tp.RUNTIME_FILES:
            self.assertIn(name, text)

    def test_templates_are_lf_public_safe_and_one_image(self):
        for name in EXPECTED:
            with self.subTest(template=name):
                self.assertNotIn(b'\r', (FOLDER / (name + '.env')).read_bytes())
                self.assertIsNone(BANNED.search(text_of(name)), name)
                self.assertEqual(len(re.findall(r'(?m)^C2_IMAGE_TAG=(.+)$', text_of(name))), 1)
                self.assertEqual(re.findall(r'(?m)^C2_IMAGE_TAG=(.+)$', text_of(name)), [IMAGE])
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertNotIn(b'\r', (FOLDER / 'ORDER.txt').read_bytes())
        self.assertIsNone(BANNED.search(order))

    def test_the_templates_are_not_the_live_job_file(self):
        live = ROOT / '.github' / 'c2-serving-job.env'
        if live.is_file():
            self.assertNotIn(IMAGE, live.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
