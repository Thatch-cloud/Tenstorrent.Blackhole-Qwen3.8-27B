"""tp4/next-5: the four-card development window's job pack (references/tp4-next5-jobs).

Every template parses with the job parser, names the one image, never touches the node agent (no agentstop, agentstart, unserve, platform), names no
production tag, bakes nothing, and the ORDER.txt lines match the files. The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-next5-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
IMAGE = 'tp4-next-5a'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
SHIP, WARM4 = 'c2-packed-tp4-best-ship', 'c2-packed-tp4-best-ship-warm4'
BS_TESTS = 'warmup,coding,concurrent4,concurrent4_steady,steady_resend,replay_concurrent4,concurrent4_code_equal'
TPUB_TESTS = 'warmup,coding,long_real_text,concurrent4,concurrent4_steady,steady_resend,replay_concurrent4,concurrent4_code_equal'
SPEED, SPEED_SLICE = 'c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-strace-pairslice'
TPUB = 'c2-packed-tp4-best-ship-tpub'
EXPECTED = {
    'X0-status-rmi': ('status rmi', None, 'stop'), 'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    'S0-production-smoke': ('reset smoke', 'c2-packed-tp4', 'stop'),
    'BS1-hang-shapes-best-ship': ('reset smoke', SHIP, 'stop'), 'BS2-hang-shapes-best-ship': ('reset smoke', SHIP, 'stop'),
    'BS3-hang-shapes-best-ship': ('reset smoke', SHIP, 'stop'),
    'T1-tpub-audited-smoke': ('reset smoke', 'c2-packed-tp4-best-gate-tpub', 'soft'),
    'T2-tpub-hang-shapes': ('reset smoke', 'c2-packed-tp4-best-strace-tpub', 'soft'),
    'T3-tpub-hang-shapes': ('reset smoke', 'c2-packed-tp4-best-strace-tpub', 'soft'),
    'TT1-timed-A-best-ship': ('reset smoke', SHIP, 'soft'), 'TT2-timed-B-best-ship-tpub': ('reset smoke', TPUB, 'soft'),
    'TT3-timed-A-best-ship': ('reset smoke', SHIP, 'soft'), 'TT4-timed-B-best-ship-tpub': ('reset smoke', TPUB, 'soft'),
    'G0-pairslice-control-audited': ('reset smoke', 'c2-packed-tp4-gate', 'soft'),
    'G1-pairslice-exact-audited': ('reset smoke', 'c2-packed-tp4-gate-pairslice', 'soft'),
    'GT1-timed-A-speed-strace': ('reset smoke', SPEED, 'soft'), 'GT2-timed-B-speed-strace-pairslice': ('reset smoke', SPEED_SLICE, 'soft'),
    'GT3-timed-A-speed-strace': ('reset smoke', SPEED, 'soft'), 'GT4-timed-B-speed-strace-pairslice': ('reset smoke', SPEED_SLICE, 'soft'),
    'E1-eight-seat-best-attach': ('reset smoke', 'c2-packed-tp4-8-best', 'soft'),
    'E2-eight-seat-quad-blocks': ('reset smoke', 'c2-packed-tp4-8-best-quad', 'soft'),
    'E3-eight-seat-quad-audited': ('reset smoke', 'c2-packed-tp4-8-best-quad-gate', 'soft'),
    'W1-hang-shapes-best-ship-warm4': ('reset smoke', WARM4, 'soft'), 'W2-hang-shapes-best-ship-warm4': ('reset smoke', WARM4, 'soft'),
    'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)['profiles']


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), sorted(profiles()), root=ROOT)


def order_lines():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


class JobPackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        names = [line[0] for line in lines]
        self.assertEqual(names, list(EXPECTED))
        files = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(files, sorted(names))
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

    def test_the_window_never_touches_the_node_agent_or_production(self):
        for name in EXPECTED:
            with self.subTest(name):
                result = parsed(name)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertEqual(parsed('X0-status-rmi')['rmi_tags'], 'tp4-next-2b tp4-seats8-rwarm tp4-tau-1')
        for tag in parsed('X0-status-rmi')['rmi_tags'].split():
            self.assertNotIn(tag, job.PROTECTED)
            self.assertFalse(tag.startswith(job.PROTECTED_PREFIXES))

    def test_ends_with_reset_and_no_agentstart_and_the_default_is_unchanged(self):
        self.assertEqual(order_lines()[-1][0], 'Z-reset')
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['default'], 'c2-packed-tp4')

    def test_the_hang_shape_arms_carry_the_asked_tests(self):
        for name in ('BS1-hang-shapes-best-ship', 'BS2-hang-shapes-best-ship', 'BS3-hang-shapes-best-ship',
                     'W1-hang-shapes-best-ship-warm4', 'W2-hang-shapes-best-ship-warm4'):
            self.assertEqual(parsed(name)['tests'].replace(' ', ','), BS_TESTS)
        for name in ('T2-tpub-hang-shapes', 'T3-tpub-hang-shapes'):
            self.assertEqual(parsed(name)['tests'].replace(' ', ','), TPUB_TESTS, 'T2 and T3 carry the same shapes')

    def test_the_timed_pairs_alternate_abab_and_differ_in_the_profile_alone(self):
        for pair, control, other in ((('TT1', 'TT2', 'TT3', 'TT4'), SHIP, TPUB), (('GT1', 'GT2', 'GT3', 'GT4'), SPEED, SPEED_SLICE)):
            names = [name for name in EXPECTED if name.split('-')[0] in pair]
            profile_order = [parsed(name)['profile'] for name in names]
            self.assertEqual(profile_order, [control, other, control, other])
            self.assertEqual(len({parsed(name)['tests'] for name in names}), 1)

    def test_the_glue_arm_is_timed_where_the_flag_is_live(self):
        # With V2 on the pair-slice flag is idle by construction, so the timed pair runs the V2-off speed-strace profiles.
        found = profiles()
        for name in (SPEED, SPEED_SLICE):
            self.assertNotEqual(found[name]['env'].get('QWEN_FAST_TP4_GDN_GLUE'), '1', name)
        self.assertEqual(found[SPEED_SLICE]['env'].get('QWEN_FAST_GDN_PAIR_SLICE'), '1')
        self.assertNotEqual(found[SPEED]['env'].get('QWEN_FAST_GDN_PAIR_SLICE'), '1')
        self.assertEqual(found['c2-packed-tp4-best-ship-glue']['env'].get('QWEN_FAST_TP4_GDN_GLUE'), '1')
        used = [parsed(name)['profile'] for name in EXPECTED if EXPECTED[name][1]]
        self.assertNotIn('c2-packed-tp4-best-ship-glue', used)

    def test_the_audited_arms_have_their_same_window_controls(self):
        found = profiles()
        g0, g1 = found['c2-packed-tp4-gate']['env'], found['c2-packed-tp4-gate-pairslice']['env']
        self.assertEqual({key: g1[key] for key in g1 if g0.get(key) != g1[key]},
                         {'QWEN_FAST_GDN_PAIR_SLICE': '1', 'QWEN_FAST_TP4_VGLUE_AUDIT': '1'})
        self.assertEqual(parsed('G0-pairslice-control-audited')['tests'], parsed('G1-pairslice-exact-audited')['tests'])
        t1 = parsed('T1-tpub-audited-smoke')['tests'].replace(' ', ',').split(',')
        self.assertIn('concurrent4_solo', t1)
        self.assertIn('concurrent4', t1)

    def test_the_eight_seat_arms_run_identical_tests_and_the_gate_twin_audits_both_quads(self):
        tests = {name: parsed(name)['tests'] for name in ('E1-eight-seat-best-attach', 'E2-eight-seat-quad-blocks', 'E3-eight-seat-quad-audited')}
        self.assertEqual(len(set(tests.values())), 1, tests)
        for needed in ('concurrent4_solo', 'concurrent4', 'concurrent8_steady', 'concurrent8_code_equal'):
            self.assertIn(needed, tests['E1-eight-seat-best-attach'].replace(' ', ',').split(','))
        found = profiles()
        quad, gate = found['c2-packed-tp4-8-best-quad'], found['c2-packed-tp4-8-best-quad-gate']
        self.assertEqual(gate['env'], dict(quad['env'], QWEN_FAST_DRAFT_SINGLES_AUDIT='all'))
        self.assertIs(gate['gate_only'], True)
        self.assertNotIn('QWEN_FAST_QUAD_DRAFT_AUDIT', gate['env'])

    def test_the_dependencies_are_written_down_and_name_real_jobs(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            text = handle.read()
        needs = [line[len('# NEEDS '):].split(' <- ') for line in text.splitlines() if line.startswith('# NEEDS ')]
        found = {tuple(left.split()): tuple(right.split()) for left, right in needs}
        self.assertEqual(found, {('T2', 'T3', 'TT1', 'TT2', 'TT3', 'TT4'): ('T1',), ('TT1', 'TT2', 'TT3', 'TT4'): ('T2', 'T3'),
                                 ('G1',): ('G0',), ('GT1', 'GT2', 'GT3', 'GT4'): ('G1',), ('E2', 'E3'): ('E1',), ('E3',): ('E2',),
                                 ('W1', 'W2'): ('BS1', 'BS2', 'BS3')})
        prefixes = [name.split('-')[0] for name in EXPECTED]
        for left, right in found.items():
            for name in left + right:
                self.assertIn(name, prefixes)
            self.assertTrue(all(prefixes.index(r) < prefixes.index(l) for l in left for r in right), (left, right))

    def test_the_header_time_matches_the_lines_and_x0_is_a_stop_job_on_the_rig_precondition(self):
        total = sum(int(line[3]) for line in order_lines())
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('sum to %s min' % format(total, ','), text)
        self.assertEqual(order_lines()[0][:2], ['X0-status-rmi', 'stop'])
        for needle in ('FOUR boards resolved', 'serving containers: none'):
            self.assertIn(needle, text_of('X0-status-rmi'))

    def test_no_hostname_address_registry_or_digest(self):
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
                text = handle.read()
            self.assertIsNone(BANNED.search(text), name)
            self.assertNotIn('\r', text, name)


if __name__ == '__main__':
    unittest.main()
