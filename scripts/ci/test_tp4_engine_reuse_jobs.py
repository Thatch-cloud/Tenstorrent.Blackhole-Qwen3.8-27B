"""tp4/engine-reuse: the card-gate job pack (references/tp4-engine-reuse-jobs).

Every template parses with the job parser, names the one image placeholder, never touches the node agent, bakes nothing, and ORDER.txt matches the files. The
profiles the pack names exist and are gate only; every test a template names is one the smoke defines; the ER5 arms are the control, the bound-drafter arm and
the E1 arm, alternated; the dependencies are the greppable NEEDS lines. The templates are public, so they name no rig, card, address, registry or digest."""

import ast
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-engine-reuse-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
SMOKE = os.path.join(HERE, 'c2_serving_smoke.py')
IMAGE = 'tp4-engine-reuse-1'
# The guard names only SHAPES (an address, a digest, a device path, a home directory). The private names a public repository must never carry
# (a host, a registry, a domain) are not written here: a maintainer's CI supplies them as a regular expression in QWEN_PUBLIC_GUARD_EXTRA, and the
# test applies them when the variable is set.
BANNED = re.compile(r'\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|[A-Za-z]:[/\\]Users[/\\]')
EXTRA = os.environ.get('QWEN_PUBLIC_GUARD_EXTRA')
BANNED_PRIVATE = re.compile(EXTRA) if EXTRA else None
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}
P = 'c2-packed-tp4-8x262k-ship-prefix-levern'


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


def needs_lines():
    found = {}
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        for line in handle.read().splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for name in match.group(1).split():
                    found.setdefault(name, set()).update(match.group(2).split())
    return found


def tests_of(name):
    return [test for test in parsed(name)['tests'].replace(' ', ',').split(',') if test]


def smoke_functions():
    with open(SMOKE, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def names():
    return [line[0] for line in order_lines()]


class JobPackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates(self):
        files = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(files, sorted(names()))
        self.assertEqual(len(names()), len(set(names())))
        self.assertEqual(names()[0], 'X0-status-rescan-reset')
        self.assertEqual(names()[1], 'B0-build')
        self.assertEqual(names()[-1], 'Z-reset')
        for name, mode, image, minutes in order_lines():
            self.assertIn(mode, ('stop', 'soft'), name)
            self.assertEqual(image, IMAGE, name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_on_the_quad_with_the_one_image(self):
        for name in names():
            with self.subTest(name):
                result = parsed(name)
                self.assertEqual(result['cards'], 'quad')
                self.assertEqual(result['tag'], IMAGE)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')

    def test_the_first_job_rescans_before_it_resets_and_the_build_names_the_default_profile(self):
        actions = parsed('X0-status-rescan-reset')['actions'].split()
        self.assertLess(actions.index('rescan'), actions.index('reset'))
        self.assertEqual(parsed('B0-build')['actions'], 'build')
        self.assertEqual(parsed('B0-build')['profile'], 'c2-packed-tp4')
        self.assertEqual(json.load(open(PROFILES_PATH, encoding='utf-8'))['default'], 'c2-packed-tp4')
        self.assertEqual(parsed('Z-reset')['actions'], 'status reset')

    def test_every_profile_a_card_job_names_exists_and_is_gate_only(self):
        found = profiles()
        for name in names():
            if name in ('X0-status-rescan-reset', 'B0-build', 'Z-reset'):
                continue
            with self.subTest(name):
                profile = parsed(name)['profile']
                self.assertTrue(profile.startswith(P), profile)
                self.assertTrue(found[profile].get('gate_only'), profile)
                self.assertEqual(parsed(name)['actions'], 'reset smoke')

    def test_every_test_a_template_names_is_defined_by_the_smoke(self):
        defined = smoke_functions()
        always = {'warmup', 'coding', 'concurrent8_steady', 'concurrent8_drain', 'stall8_cold128k', 'stall8_cold262k'}
        for name in names():
            for test in tests_of(name):
                with self.subTest(name=name, test=test):
                    self.assertTrue(test in always or test in defined, test)

    def test_the_audited_pair_runs_the_same_tests_and_the_parked_arm_needs_the_control(self):
        self.assertEqual(parsed('A0-control-audit-attach')['tests'], parsed('A1-parked-audit-attach')['tests'])
        self.assertEqual(parsed('C0-control-audit-churn')['tests'], parsed('C1-parked-audit-churn')['tests'])
        self.assertEqual(needs_lines()['A1'], {'A0'})
        self.assertIn('parked_compare.py', text_of('A1-parked-audit-attach'))
        found = profiles()
        control, parked = found[P + '-audit-r2'], found[P + '-parked-audit']
        for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual(control['env'][key], parked['env'][key])
        self.assertEqual(parked['env']['QWEN_FAST_PARKED_AUDIT'], '1')

    def test_the_er0_rules_are_numeric_and_er1_starts_at_the_worst_corner(self):
        text = text_of('A1-parked-audit-attach')
        self.assertIn('k=8 of 8', text)
        self.assertIn('3.6 GB', text)
        m1 = text_of('M1-parked-memory-churn')
        self.assertIn('253,920', m1)
        self.assertIn('stall8_cold262k', parsed('M1-parked-memory-churn')['tests'])
        self.assertIn('parked_churn_long', parsed('M1-parked-memory-churn')['tests'])

    def test_the_negative_controls_run_on_their_own_profiles_and_name_their_judges(self):
        expected = {'N1-neg-carry': 'neg-carry', 'N2-neg-pages': 'neg-pages', 'N3-neg-widths': 'neg-widths', 'N4-neg-drafter': 'audit-neg-drafter'}
        found = profiles()
        for name, suffix in expected.items():
            with self.subTest(name):
                profile = parsed(name)['profile']
                self.assertEqual(profile, P + '-parked-' + suffix)
                self.assertTrue(found[profile]['env'].get('QWEN_FAST_PARKED_NEGATIVE'))
                self.assertIn('--mode', text_of(name))
        self.assertEqual(parsed('N0-control-plain')['profile'], P)

    def test_the_fault_jobs_run_on_their_fault_profiles(self):
        found = profiles()
        self.assertEqual(found[parsed('F1-fault-park')['profile']]['env']['QWEN_FAST_PARKED_FAULT'], 'park')
        self.assertEqual(found[parsed('F2-fault-rebind')['profile']]['env']['QWEN_FAST_PARKED_FAULT'], 'rebind')
        self.assertNotIn('parked.off', text_of('F2-fault-rebind'), 'no manual step: the kill switch is the K1 job, latched by the server')
        self.assertEqual(found[parsed('K1-kill-switch')['profile']]['env']['QWEN_FAST_PARKED_OFF_AFTER'], '3')
        self.assertNotIn('QWEN_FAST_PARKED_FAULT', found[parsed('K1-kill-switch')['profile']]['env'])
        self.assertIn('exactly one', text_of('K1-kill-switch'))
        self.assertIn('parked_abort_reuse', tests_of('F1-fault-park'))

    def test_the_ballast_ladder_is_the_four_levels(self):
        levels = [parsed(name)['profile'] for name in names() if name.startswith('M') and 'ballast' in name]
        self.assertEqual([profile.rsplit('-', 1)[1] for profile in levels], ['512', '1024', '1536', '2048'])
        found = profiles()
        for profile in levels:
            self.assertEqual(found[profile]['env']['QWEN_FAST_GATE_DRAM_BALLAST'], str(int(profile.rsplit('-', 1)[1]) * 1000000))

    def test_the_hang_shapes_run_three_times_with_arrivals_after_block_rounds(self):
        hangs = ['H1-hang-shapes-parked', 'H2-hang-shapes-parked', 'H3-hang-shapes-parked']
        self.assertEqual(len({parsed(name)['tests'] for name in hangs}), 1)
        self.assertEqual({parsed(name)['profile'] for name in hangs}, {P + '-parked'})
        for shape in ('levern_arrival_during_prefill', 'parked_abort_reuse', 'concurrent8_steady'):
            self.assertIn(shape, tests_of(hangs[0]))
        self.assertIn('100 block rounds', text_of(hangs[0]))
        self.assertEqual({parsed(name)['profile'] and profiles()[parsed(name)['profile']]['env']['QWEN_FAST_VERIFY_T1_AUDIT'] for name in hangs}, {'0'})

    def test_er5_alternates_control_bound_drafter_and_e1_paired(self):
        turns = [name for name in names() if name.startswith('R') and 'turns' in name]
        self.assertEqual([parsed(name)['profile'] for name in turns], [P, P + '-parked', P, P + '-parked', P + '-parked-e1', P])
        self.assertEqual({parsed(name)['tests'] for name in turns}, {'warmup,parked_turns'})
        self.assertEqual(needs_lines()['R1'], {'A1', 'C1', 'H1', 'H2', 'H3', 'M1'})
        self.assertIn('paired', text_of(turns[0]))

    def test_the_dependencies_name_jobs_that_exist_and_run_earlier(self):
        order = names()
        short = {name.split('-')[0]: name for name in order}
        for waiting, needed in needs_lines().items():
            self.assertIn(waiting, short)
            for other in needed:
                self.assertIn(other, short)
                self.assertLess(order.index(short[other]), order.index(short[waiting]), (waiting, other))

    def test_a_job_judged_against_a_reference_job_needs_it(self):
        needs = needs_lines()

        def closure(name, seen=None):
            seen = set() if seen is None else seen
            for other in needs.get(name, ()):
                if other not in seen:
                    seen.add(other)
                    closure(other, seen)
            return seen

        short = {name.split('-')[0]: name for name in names()}
        checked = 0
        for name in names():
            own = name.split('-')[0]
            for reference in sorted(set(re.findall(r'(?:from|against|equal to) ([A-Z][0-9])(?![0-9A-Za-z])', text_of(name)))):
                if reference == own or reference not in short or own.startswith('R'):    # the R arms pair each other: each is a timing of its own
                    continue
                checked += 1
                with self.subTest(name=name, reference=reference):
                    self.assertIn(reference, closure(own), '%s is judged against %s, so it must depend on it' % (own, reference))
        self.assertGreater(checked, 3)

    def test_the_templates_are_public_safe(self):
        for name in names() + ['ORDER']:
            path = os.path.join(FOLDER, name + ('' if name == 'ORDER' else '') + ('.txt' if name == 'ORDER' else '.env'))
            with open(path, encoding='utf-8') as handle:
                found = BANNED.search(handle.read())
            self.assertIsNone(found, (name, found and found.group(0)))
            if BANNED_PRIVATE is not None:
                with open(path, encoding='utf-8') as handle:
                    self.assertIsNone(BANNED_PRIVATE.search(handle.read()), name)
        with open(os.path.join(FOLDER, 'ORDER.txt'), 'rb') as handle:
            self.assertNotIn(b'\r', handle.read())


if __name__ == '__main__':
    unittest.main()
