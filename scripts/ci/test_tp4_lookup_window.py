"""The tp4/lookup window: its job templates (scripts/ci/references/tp4-lookup-jobs), the profile it adds, the smoke rule, the tau report, the image overlay and the CI allowlist.

The window times prompt-lookup drafting (QWEN_FAST_LOOKUP_DRAFT) against the combined best: B0 builds the image, L1 is the audited smoke, TL1..TL4 the paired timing ABAB, Z the closing reset. The
cards are under development: no job stops or starts the node agent, hands back or deploys. The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402
import lookup_tau_report  # noqa: E402
import prompt_lookup  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FOLDER = HERE / 'references' / 'tp4-lookup-jobs'
with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)['profiles']
NAMES = sorted(PROFILES)
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@|[A-Za-z]:[\\/]Users')
IMAGE = 'tp4-lookup-1'
CONTROL, ARM, PRODUCTION = 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-lookup', 'c2-packed-tp4'
FLAG, POLICY = 'QWEN_FAST_LOOKUP_DRAFT', 'n3m12'
STATUS, BUILD, SMOKE, RESET = 'X0-status-rmi', 'B0-build', 'L1-lookup-audited-smoke', 'Z-reset'
TIMED = ('TL1-timed-A-best-strace', 'TL2-timed-B-best-lookup', 'TL3-timed-A-best-strace', 'TL4-timed-B-best-lookup')
ORDERED = (STATUS, BUILD, SMOKE) + TIMED + (RESET,)
SMOKE_ARM = 'c2-packed-tp4-best-gate-lookup'   # the audited twin: best-gate plus the flag, so the lever audits run beside the lookup
PROFILE_OF = {BUILD: PRODUCTION, SMOKE: SMOKE_ARM, TIMED[0]: CONTROL, TIMED[1]: ARM, TIMED[2]: CONTROL, TIMED[3]: ARM}
FORBIDDEN_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'push', 'replay', 'prefix', 'platform'}


def read_order():
    text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
    return [line.split() for line in text.splitlines() if line.strip() and not line.startswith('#')]


def text_of(name):
    return (FOLDER / (name + '.env')).read_text(encoding='utf-8')


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual([row[0] for row in rows], list(ORDERED))
        self.assertEqual(sorted(row[0] for row in rows), sorted(path.stem for path in FOLDER.glob('*.env')))

    def test_modes_images_minutes_and_the_one_stop_job(self):
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'soft'))
                self.assertEqual(mode, 'stop' if name in (STATUS, BUILD) else 'soft')
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180, minutes)
        total = sum(int(row[3]) for row in read_order())
        self.assertIn('sum to %d min' % total, (FOLDER / 'ORDER.txt').read_text(encoding='utf-8'))

    def test_each_template_parses_through_the_job_parser_on_four_cards(self):
        for name in ORDERED:
            with self.subTest(job=name):
                outputs = parsed(name)
                self.assertEqual(outputs['cards'], 'quad')

    def test_no_job_touches_the_agent_hands_back_or_deploys(self):
        for name in ORDERED:
            actions = set(parsed(name)['actions'].split())
            self.assertFalse(FORBIDDEN_ACTIONS & actions, name)
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('NO agentstop', 'NO agentstart', 'NO hand-back', 'NO deploy'):
            self.assertIn(word, order)

    def test_the_build_is_card_free_and_the_first_quad_job_rescans_before_it_resets(self):
        outputs = parsed(BUILD)
        self.assertEqual((outputs['actions'], outputs['cards'], outputs['tag'], outputs['profile']), ('build', 'quad', IMAGE, PRODUCTION))
        self.assertEqual(parsed(SMOKE)['actions'], 'rescan reset smoke')
        self.assertEqual(parsed(STATUS)['actions'], 'status rescan')
        for name in ORDERED:
            if name not in (BUILD, SMOKE, STATUS):
                self.assertNotIn('rescan', parsed(name)['actions'].split(), name)
            self.assertEqual('build' in parsed(name)['actions'].split(), name == BUILD, name)

    def test_profiles_and_tests_per_job(self):
        for name, profile in PROFILE_OF.items():
            self.assertEqual(parsed(name)['profile'], profile, name)
        for name in TIMED:
            self.assertEqual(parsed(name)['actions'], 'reset smoke')
            self.assertEqual(parsed(name)['tests'], 'warmup,coding,concurrent4_code_equal,concurrent4_code_32k')
        self.assertEqual(parsed(SMOKE)['tests'], 'warmup,coding,long_real_text,concurrent4_solo,concurrent4,concurrent4_code_equal')
        self.assertEqual((parsed(STATUS)['actions'], parsed(STATUS)['cards']), ('status rescan', 'quad'))
        self.assertEqual(parsed(RESET)['actions'], 'status reset')

    def test_the_dependencies_are_written_down_machine_greppable(self):
        text = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        needs = [line[len('# NEEDS '):].split(' <- ') for line in text.splitlines() if line.startswith('# NEEDS ')]
        self.assertEqual({tuple(left.split()): tuple(right.split()) for left, right in needs}, {('TL1', 'TL2', 'TL3', 'TL4'): ('L1',)})

    def test_the_timed_arms_alternate_A_B_A_B_and_the_order_names_the_pairing(self):
        self.assertEqual([parsed(name)['profile'] for name in TIMED], [CONTROL, ARM, CONTROL, ARM])
        order = (FOLDER / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('PAIRED per round', 'ABAB', 'lookup_tau_report.py', 'lookup_sim.py', 'NO-GO', 'rescan before reset', 'SMOKE_JSON', 'FOUR boards'):
            self.assertIn(word, order)

    def test_public_text_names_no_rig_card_address_or_digest(self):
        for path in list(FOLDER.iterdir()) + [HERE / 'prompt_lookup.py', HERE / 'lookup_tau_report.py', ROOT / 'optimisation' / 'lookup' / 'lookup_sim.py',
                                              ROOT / 'optimisation' / 'lookup' / 'build_tapes.py', ROOT / 'docs' / 'tp4-lookup.md']:
            with self.subTest(path=path.name):
                found = BANNED.search(path.read_text(encoding='utf-8'))
                self.assertIsNone(found, found and found.group(0))

    def test_files_are_lf(self):
        for path in list(FOLDER.iterdir()) + [HERE / 'prompt_lookup.py', HERE / 'test_prompt_lookup.py', HERE / 'lookup_tau_report.py', HERE / 'test_tp4_lookup_window.py']:
            self.assertNotIn(b'\r', path.read_bytes(), path.name)


class ProfileTests(unittest.TestCase):
    def test_the_arm_is_the_control_plus_the_one_flag(self):
        control, arm = PROFILES[CONTROL], PROFILES[ARM]
        self.assertEqual(arm['env'], dict(control['env'], **{FLAG: POLICY}))
        for key in set(control) | set(arm):
            if key not in ('env', 'description'):
                self.assertEqual(arm.get(key), control.get(key), key)
        self.assertTrue(arm['gate_only'])
        self.assertIn('GATE ONLY', arm['description'])
        self.assertIn('UNVERIFIED on hardware', arm['description'])

    def test_the_audited_smoke_arm_is_best_gate_plus_the_one_flag_and_carries_the_lever_audits(self):
        gate, arm = PROFILES['c2-packed-tp4-best-gate'], PROFILES[SMOKE_ARM]
        self.assertEqual(arm['env'], dict(gate['env'], **{FLAG: POLICY}))
        for key in set(gate) | set(arm):
            if key not in ('env', 'description'):
                self.assertEqual(arm.get(key), gate.get(key), key)
        for audit in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_FAST_FUSED_COMMIT_AUDIT', 'QWEN_FAST_DRAFT_SINGLES_AUDIT', 'QWEN_FAST_TP4_VGLUE_AUDIT'):
            self.assertNotIn(PROFILES[ARM]['env'].get(audit, '0'), ('1', 'all'), 'the timed arm stays unaudited: ' + audit)
            self.assertIn(arm['env'].get(audit), ('1', 'all'), audit)
        self.assertTrue(arm['gate_only'])
        self.assertIn('GATE ONLY', arm['description'])
        self.assertIn('UNVERIFIED on hardware', arm['description'])

    def test_the_flag_is_off_everywhere_else_and_the_policy_parses(self):
        for name, profile in PROFILES.items():
            self.assertEqual(FLAG in profile['env'], name in (ARM, SMOKE_ARM, 'c2-packed-tp4-8x262k-best-time-gate-lookup'), name)
        self.assertEqual(repr(prompt_lookup.parse_policy(PROFILES[ARM]['env'][FLAG])), POLICY)

    def test_the_default_stays_production(self):
        with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['default'], PRODUCTION)


class SmokeRuleTests(unittest.TestCase):
    ON = {FLAG: POLICY}
    ENGAGED = '[LOOKUP-DRAFT] engaged policy=n3m12 request=r0 history=4100\n'
    ROUND = '[LOOKUP-ROUND] request=r0 position=4100 source=%s match=%d offered=%d proposed=%d committed=%d\n'

    def problems(self, env, text):
        return c2_smoke_check.lookup_problems(env, text)

    def test_off_wants_no_lookup_line_at_all(self):
        self.assertEqual(self.problems({}, 'nothing\n'), [])
        self.assertEqual(self.problems({FLAG: 'off'}, 'nothing\n'), [])
        self.assertTrue(self.problems({}, self.ENGAGED))
        self.assertTrue(self.problems({}, self.ROUND % ('lookup', 12, 15, 15, 4)))

    def test_on_wants_the_engaged_line_naming_the_policy_and_round_lines(self):
        good = self.ENGAGED + self.ROUND % ('lookup', 12, 15, 15, 4) + self.ROUND % ('dflash2', 0, 0, 15, 6)
        self.assertEqual(self.problems(self.ON, good), [])
        self.assertTrue(self.problems(self.ON, self.ROUND % ('lookup', 12, 15, 15, 4)))                    # no engaged line
        self.assertTrue(self.problems(self.ON, self.ENGAGED))                                              # no round line
        self.assertTrue(self.problems(self.ON, self.ENGAGED.replace('n3m12', 'n4m8') + self.ROUND % ('dflash2', 0, 0, 15, 6)))
        self.assertTrue(self.problems(self.ON, self.ENGAGED + '[LOOKUP-ROUND] request=r0 garbage\n'))     # malformed
        self.assertTrue(self.problems(self.ON, self.ENGAGED + self.ROUND % ('lookup', 12, 0, 0, 4)))       # a lookup round proposing nothing
        self.assertTrue(self.problems(self.ON, self.ENGAGED + self.ROUND % ('dflash2', 0, 0, 15, 17)))     # committed past the width

    def test_the_check_wires_the_rule_in_for_a_profile_env(self):
        source = (HERE / 'c2_smoke_check.py').read_text(encoding='utf-8')
        self.assertIn('problems += lookup_problems(env, container_text)', source)


class TauReportTests(unittest.TestCase):
    def write(self, directory, name, lines):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return path

    def packed(self, request, positions_and_emitted, live=4):
        lines = ['[PACKED-PHASE] round=1 users=4 bind_ms=1 input_ms=1 trace_ms=1 sync_ms=1 readback_ms=1 live=%d' % live]
        lines += ['[PACKED] request=%s segment=0 position=%d prefix=0 emitted=%d' % (request, p, e) for p, e in positions_and_emitted]
        return lines

    def test_tau_excludes_each_requests_last_round_and_other_live_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            lines = self.packed('a', [(10, 4), (14, 6), (20, 1)]) + self.packed('b', [(10, 8), (18, 2)]) + self.packed('c', [(10, 16), (26, 1)], live=3)
            self.write(directory, 'x/container.log', lines)
            result = lookup_tau_report.summarize(lookup_tau_report.logs_under(directory), 4)
        self.assertEqual(result['packed_rounds'], 3)               # a: 4, 6; b: 8 (each last round dropped); c is live 3
        self.assertAlmostEqual(result['tau'], 6.0)

    def test_lookup_rounds_are_split_by_source(self):
        row = '[LOOKUP-ROUND] request=a position=%d source=%s match=%d offered=%d proposed=%d committed=%d'
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory, 'container.log', [row % (10, 'lookup', 12, 15, 15, 9), row % (20, 'dflash2', 3, 15, 15, 5), row % (30, 'dflash2', 0, 0, 15, 3)])
            result = lookup_tau_report.summarize(lookup_tau_report.logs_under(directory), 4)
        self.assertEqual(result['lookup']['rounds'], 1)
        self.assertAlmostEqual(result['lookup']['tau'], 9.0)
        self.assertAlmostEqual(result['dflash2']['tau'], 4.0)
        self.assertAlmostEqual(result['lookup']['share'], 1 / 3.0)
        self.assertIn('source lookup', '\n'.join(lookup_tau_report.render('B', result)))

    def test_main_prints_a_pair_and_refuses_an_empty_tree(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second, tempfile.TemporaryDirectory() as empty:
            self.write(first, 'container.log', self.packed('a', [(10, 4), (14, 4), (20, 1)]))
            self.write(second, 'container.log', self.packed('a', [(10, 5), (15, 5), (20, 1)]))
            self.assertEqual(lookup_tau_report.main([first, second]), 0)
            self.assertEqual(lookup_tau_report.main([empty]), 2)


class ImageAndCiTests(unittest.TestCase):
    def test_the_module_is_in_the_image_overlay_with_its_callers(self):
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        lines = [line.strip() for line in overlay.splitlines() if line.strip() and not line.startswith('#')]
        for path in ('scripts/ci/prompt_lookup.py', 'scripts/ci/serving_fast_request.py', 'scripts/ci/serving_request_factory.py'):
            self.assertIn(path, lines)

    def test_the_test_modules_are_on_the_cpu_allowlist(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        for module in ('test_prompt_lookup', 'test_tp4_lookup_window', 'test_lookup_sim'):
            self.assertIn(module, workflow)

    def test_the_design_document_names_the_flag_the_injection_point_and_the_estimate(self):
        text = (ROOT / 'docs' / 'tp4-lookup.md').read_text(encoding='utf-8')
        for word in (FLAG, 'c2-packed-tp4-best-lookup', 'FastRequest.prepare', 'segment_users', 'fixture.tokens', 'lookup_sim.py', 'byte-identical',
                     'tp4-lookup-jobs', 'no device allocation', 'lower bound'):
            self.assertIn(word, text)


if __name__ == '__main__':
    unittest.main()
