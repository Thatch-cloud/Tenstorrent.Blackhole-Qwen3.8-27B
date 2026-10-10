"""tp4/octo-t8: the card-gate job pack (references/tp4-octo-jobs).

Every template parses with the job parser, names the one image placeholder, never touches the node agent, bakes nothing, and ORDER.txt matches the files. The profiles the pack names
exist and are gate only; every test a template names is one the smoke defines; the audited pair is the control and the arm on the same tests; the exactness jobs run the strict gate plans on
the audited octo arm; the hang shapes run three times on the timed arm; the timing jobs pair the alternate boot and set the live boot against the flag-off control; the dependencies are the
greppable NEEDS lines. The templates are public, so they name no rig, card, address, registry or digest."""

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
FOLDER = os.path.join(HERE, 'references', 'tp4-octo-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
SMOKE = os.path.join(HERE, 'c2_serving_smoke.py')
IMAGE = 'tp4-octo-1'
# The guard names only SHAPES (an address, a digest, a device path, a home directory). The private names a public repository must never carry (a host, a registry, a domain) are not
# written here: a maintainer's CI supplies them as a regular expression in QWEN_PUBLIC_GUARD_EXTRA, and the test applies them when the variable is set.
BANNED = re.compile(r'\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|[A-Za-z]:[/\\]Users[/\\]')
EXTRA = os.environ.get('QWEN_PUBLIC_GUARD_EXTRA')
BANNED_PRIVATE = re.compile(EXTRA) if EXTRA else None
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}
P = 'c2-packed-tp4-8x262k-ship-prefix-levern'
CONTROLS = ('A0-control-audit-attach', 'M0-memory-ledger-control', 'P3-timing-control', 'R2-parked-timing-control', 'L0-lone-user-control')


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


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def needs_lines():
    found = {}
    for line in order_text().splitlines():
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


def q_jobs():
    """The one-card qualification jobs (Q1w, Q1a, Q1b, Q2w, Q2): the cardm action on a pair-shaped step, the quad boards untouched."""
    return [name for name in names() if re.match(r'Q\d', name)]


def card_jobs():
    """The quad jobs that serve a profile (everything but the status, build, reset and one-card qualification jobs)."""
    return [name for name in names() if name not in ('X0-status-rescan-reset', 'B0-build', 'Z-reset') and name not in q_jobs()]


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
                self.assertEqual(result['tag'], IMAGE)
                self.assertEqual(result['bake_default_profile'] or '', '')
                if name in q_jobs():
                    # a one-card evidence job: cardm is a pair-shaped step (card M alone, no reset, no agent), the only action that may name 'cardm' here
                    self.assertEqual((result['cards'], result['actions']), ('pair', 'cardm'))
                    self.assertFalse((AGENT_ACTIONS - {'cardm'}) & set(result['actions'].split()))
                    continue
                self.assertEqual(result['cards'], 'quad')
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))

    def test_the_qualification_jobs_are_one_card_evidence_on_the_served_graft_and_run_before_any_octo_arm(self):
        self.assertEqual(q_jobs(), ['Q1w-k64j-g8b1-watcher', 'Q1a-k64j-g8b1-cb1', 'Q1b-k64j-g8b1-cb2a', 'Q2w-gdn-eight-users-watcher', 'Q2-gdn-eight-users'])
        order = names()
        for name in q_jobs():
            self.assertLess(order.index(name), order.index('A0-control-audit-attach'), name)
            result = parsed(name)
            self.assertEqual(result['cardm_harness'], 'optimisation/ttnn-op/k64j/run_card_b.sh')
            env = result['cardm_env']
            self.assertIn('EXPECT_TTNNCPP_SHA256=@K64J_TTNNCPP_SHA256@', env, 'the graft binary is pinned: a graft mounted is not a graft executed')
            self.assertIn('KOPGRAFT64=@K64J_GRAFT_DIR@', env)
            self.assertIn('NO NEW GRAFT', text_of(name))
        for name in ('Q1w-k64j-g8b1-watcher', 'Q2w-gdn-eight-users-watcher'):
            self.assertIn('WATCHER=1', parsed(name)['cardm_env'])
        for name in ('Q1a-k64j-g8b1-cb1', 'Q1b-k64j-g8b1-cb2a', 'Q2-gdn-eight-users'):
            self.assertNotIn('WATCHER', parsed(name)['cardm_env'])
        for name in q_jobs()[:3]:
            self.assertIn('K64J_HARNESS=card', parsed(name)['cardm_env'])
            self.assertIn('--kv-heads 1 --octo', parsed(name)['cardm_args'])
        for name in q_jobs()[3:]:
            self.assertIn('K64J_HARNESS=gdn_tp4', parsed(name)['cardm_env'])
            self.assertIn('--users 8 --rows 8', parsed(name)['cardm_args'])
        needs = needs_lines()
        self.assertEqual((needs['Q1a'], needs['Q1b'], needs['Q2']), ({'Q1w'}, {'Q1a'}, {'Q2w'}))
        self.assertIn('Q1b', needs['A1'])
        self.assertIn('Q2', needs['A1'])

    def test_the_graft_the_order_names_is_the_served_one_and_no_new_graft_is_claimed(self):
        import packed_any_admission

        text = order_text()
        self.assertNotIn(packed_any_admission.K64J_TTNNCPP_SHA256, text, 'a public pack names no digest')
        self.assertIn('packed_any_admission.K64J_TTNNCPP_SHA256', text)
        with open(os.path.join(ROOT, 'docker', 'qwen-c2-serving.Dockerfile'), encoding='utf-8') as handle:
            self.assertIn(packed_any_admission.K64J_TTNNCPP_SHA256, handle.read(), 'the image checks the very digest the order tells the driver to pin')
        self.assertIn('GRAFT (none new)', text)
        self.assertIn('No kernel, factory or library changed for octo-T8', text)

    def test_every_eight_seat_smoke_runs_the_steady_eight_the_quad_blocks_rule_judges(self):
        # c2_smoke_check.blocks_problems: a profile that asks for the eight-seat quad (QWEN_FAST_QUAD_DRAFT_BLOCKS=2) fails its job unless the smoke ran concurrent8_steady
        for name in card_jobs():
            result = parsed(name)
            if result['actions'] != 'reset smoke':
                continue
            if profiles()[result['profile']]['env'].get('QWEN_FAST_QUAD_DRAFT_BLOCKS') == '2':
                with self.subTest(name):
                    self.assertIn('concurrent8_steady', tests_of(name))

    def test_the_first_job_rescans_before_it_resets_and_the_build_names_the_default_profile(self):
        actions = parsed('X0-status-rescan-reset')['actions'].split()
        self.assertLess(actions.index('rescan'), actions.index('reset'))
        self.assertEqual(parsed('B0-build')['actions'], 'build')
        self.assertEqual(parsed('B0-build')['profile'], 'c2-packed-tp4')
        self.assertEqual(json.load(open(PROFILES_PATH, encoding='utf-8'))['default'], 'c2-packed-tp4')
        self.assertEqual(parsed('Z-reset')['actions'], 'status reset')

    def test_every_profile_a_card_job_names_exists_and_is_gate_only_or_a_control_of_one(self):
        found = profiles()
        for name in card_jobs():
            with self.subTest(name):
                profile = parsed(name)['profile']
                self.assertTrue(profile.startswith(P), profile)
                if name in CONTROLS:
                    # a control is a flag-off parent: gate only (the Lever N parents are) and with no octo flag at all
                    self.assertNotIn('QWEN_FAST_OCTO', found[profile]['env'])
                    self.assertNotIn('QWEN_FAST_SOLO_PACKED', found[profile]['env'])
                else:
                    self.assertTrue(found[profile].get('gate_only'), profile)
                    self.assertTrue('QWEN_FAST_OCTO' in found[profile]['env'] or 'QWEN_FAST_SOLO_PACKED' in found[profile]['env'], profile)
                self.assertIn(parsed(name)['actions'], ('reset smoke', 'reset gate'))

    def test_every_test_a_template_names_is_defined_by_the_smoke(self):
        defined = smoke_functions()
        always = {'warmup', 'coding', 'concurrent8_steady', 'concurrent8_drain', 'stall8_cold128k', 'stall8_cold262k'}
        for name in card_jobs():
            for test in tests_of(name) if parsed(name)['actions'] == 'reset smoke' else ():
                with self.subTest(name=name, test=test):
                    self.assertTrue(test in always or test in defined, test)

    def test_the_audited_pair_runs_the_same_tests_and_the_arm_needs_the_control(self):
        self.assertEqual(parsed('A0-control-audit-attach')['tests'], parsed('A1-octo-audit-attach')['tests'])
        self.assertEqual(parsed('A0-control-audit-attach')['profile'], P + '-audit')
        self.assertEqual(parsed('A1-octo-audit-attach')['profile'], P + '-octo-audit')
        self.assertEqual(needs_lines()['A1'], {'A0', 'Q1b', 'Q2'})
        self.assertIn('octo_compare.py', text_of('A1-octo-audit-attach'))
        found = profiles()
        control, arm = found[P + '-audit'], found[P + '-octo-audit']
        for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual(control['env'][key], arm['env'][key], key)
        self.assertIn('concurrent8_drain', tests_of('A1-octo-audit-attach'))
        self.assertIn('concurrent5_split', tests_of('A1-octo-audit-attach'))

    def test_the_attach_job_reads_the_three_things_that_prove_the_shape_ran(self):
        text = text_of('A1-octo-audit-attach')
        self.assertIn("'[OCTO] admitted mode=alternate rows=8 users=8 min_live=6 (gate only)'", text)
        self.assertIn('ZERO programs after block A', text)
        self.assertIn('at least 16 octo rounds at rows=8', text)
        self.assertIn('EACH shape alternating strictly', text)
        self.assertIn('no program compiled on a switch', text)
        self.assertIn('ANY difference is NO-GO', text)

    def test_exactness_is_the_strict_gate_on_the_audited_arm_and_covers_a_changing_live_count(self):
        for name, plan in (('E1-exact-matrix-alternate', 'matrix'), ('E2-exact-staggered-alternate', 'staggered')):
            with self.subTest(name):
                result = parsed(name)
                self.assertEqual((result['profile'], result['actions'], result['gate_plan']), (P + '-octo-audit', 'reset gate', plan))
                self.assertEqual(len(result['gate_lengths'].split(',')), 8, 'eight users')
                self.assertIn('IDENTICAL', text_of(name).upper().replace('identical', 'IDENTICAL'))
        self.assertIn('one rerun', text_of('E1-exact-matrix-alternate'))
        self.assertEqual(needs_lines()['E1'], {'A1'})
        self.assertEqual(needs_lines()['E2'], {'A1'})
        self.assertIn('M3 at 5 and under, octo at 6 to 8', text_of('E2-exact-staggered-alternate'))

    def test_the_hang_shapes_run_three_times_on_the_timed_arm_with_the_audits_off(self):
        hangs = ['H1-hang-shapes-octo', 'H2-hang-shapes-octo', 'H3-hang-shapes-octo']
        self.assertEqual(len({parsed(name)['tests'] for name in hangs}), 1)
        self.assertEqual({parsed(name)['profile'] for name in hangs}, {P + '-octo'})
        for shape in ('levern_arrival_during_prefill', 'concurrent8_steady', 'concurrent8_drain', 'concurrent5_split', 'replay_concurrent8'):
            self.assertIn(shape, tests_of(hangs[0]))
        self.assertIn('THREE CONSECUTIVE COMPLETIONS', text_of(hangs[0]))
        self.assertEqual({profiles()[parsed(name)['profile']]['env']['QWEN_FAST_VERIFY_T1_AUDIT'] for name in hangs}, {'0'})
        self.assertEqual(needs_lines()['H1'], {'A1'})

    def test_the_memory_jobs_set_the_lowered_pool_against_the_parents_ledger(self):
        m1, m0 = parsed('M1-memory-ledger'), parsed('M0-memory-ledger-control')
        self.assertEqual((m1['profile'], m1['gate_plan']), (P + '-octo', 'memory'))
        self.assertEqual((m0['profile'], m0['gate_plan']), (P, 'memory'))
        self.assertEqual(needs_lines()['M1'], {'M0', 'A1'})
        self.assertIn('3 GB per chip', text_of('M1-memory-ledger'))
        self.assertIn('44 MB', text_of('M1-memory-ledger'))
        found = profiles()
        self.assertEqual(found[P + '-octo']['engine']['num-gpu-blocks-override'], 16500)
        self.assertEqual(found[P]['engine']['num-gpu-blocks-override'], 19968)

    def test_the_timing_jobs_pair_one_boot_and_set_the_live_boot_against_the_control(self):
        self.assertEqual(parsed('P1-timing-alternate')['profile'], P + '-octo')
        self.assertEqual(parsed('P4-timing-alternate-repeat')['profile'], P + '-octo')
        self.assertEqual(parsed('P2-timing-live')['profile'], P + '-octo-live')
        self.assertEqual(parsed('P3-timing-control')['profile'], P)
        self.assertEqual(len({parsed(name)['tests'] for name in ('P1-timing-alternate', 'P2-timing-live', 'P3-timing-control', 'P4-timing-alternate-repeat')}), 1)
        for test in ('concurrent8_code_equal', 'concurrent8_code', 'concurrent8_code_32k'):
            self.assertIn(test, tests_of('P1-timing-alternate'))
        verdict = text_of('P1-timing-alternate')
        self.assertIn('octo_judge.py', verdict)
        self.assertIn('--verdict', verdict)
        self.assertIn('+10%', verdict)
        self.assertIn('PAIRS', verdict)
        self.assertIn('octo_compare.py', text_of('P2-timing-live'))
        for name in ('P1-timing-alternate', 'P2-timing-live', 'P4-timing-alternate-repeat'):
            self.assertEqual(profiles()[parsed(name)['profile']]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0', name)
        self.assertEqual(needs_lines()['P1'], {'P3', 'E1', 'E2', 'H1', 'H2', 'H3', 'M1'})

    def test_the_engine_reuse_arms_are_the_parked_parent_with_and_without_the_flag(self):
        self.assertEqual(parsed('R1-parked-timing-alternate')['profile'], P + '-octo-parked')
        self.assertEqual(parsed('R2-parked-timing-control')['profile'], P + '-parked')
        self.assertEqual(parsed('R3-parked-timing-live')['profile'], P + '-octo-parked-live')
        self.assertEqual(needs_lines()['R1'], {'R2', 'P1'})

    def test_the_lone_user_jobs_run_the_drain_and_the_lone_coding_stream_against_a_flag_off_control(self):
        l1, l0 = parsed('L1-lone-user-packed'), parsed('L0-lone-user-control')
        self.assertEqual((l1['profile'], l0['profile']), (P + '-solopacked', P))
        self.assertEqual(l1['tests'], l0['tests'])
        for test in ('coding', 'concurrent8_drain'):
            self.assertIn(test, tests_of('L1-lone-user-packed'))
        self.assertEqual(needs_lines()['L1'], {'L0'})
        self.assertIn('packed padded round live=1', text_of('L1-lone-user-packed'))
        self.assertIn('three idle segments', text_of('L1-lone-user-packed'))

    def test_the_order_says_what_it_decides_and_what_it_needs_before_it_can_run(self):
        text = order_text()
        for phrase in ('NO agentstart, NO agentstop', 'NO :latest move', 'serving_octo.DEVICE_PIECES', 'fail at the attach', 'UNQUALIFIED', 'GO', 'NO-GO',
                       'qwen-cpu-suite.yml', 'every text exact', 'K64j is NOT rebuilt', 'KNOWN COST', 'THE MEMORY RISK', 'FIRST CARD JOB'):
            self.assertIn(phrase, text)

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
            for reference in sorted(set(re.findall(r'(?:from|against|equal to|set against) ([A-Z][0-9])(?![0-9A-Za-z])', text_of(name)))):
                if reference == own or reference not in short:
                    continue
                checked += 1
                with self.subTest(name=name, reference=reference):
                    self.assertIn(reference, closure(own), '%s is judged against %s, so it must depend on it' % (own, reference))
        self.assertGreaterEqual(checked, 3)

    def test_the_stop_and_soft_modes_follow_what_each_job_is(self):
        modes = {line[0]: line[1] for line in order_lines()}
        for name in q_jobs() + ['X0-status-rescan-reset', 'B0-build', 'A0-control-audit-attach', 'A1-octo-audit-attach', 'E1-exact-matrix-alternate', 'E2-exact-staggered-alternate',
                     'H1-hang-shapes-octo', 'H2-hang-shapes-octo', 'H3-hang-shapes-octo', 'M1-memory-ledger', 'Z-reset']:
            self.assertEqual(modes[name], 'stop', name)
        for name in ('P1-timing-alternate', 'P2-timing-live', 'P3-timing-control', 'P4-timing-alternate-repeat', 'R1-parked-timing-alternate', 'L1-lone-user-packed'):
            self.assertEqual(modes[name], 'soft', name)

    def test_the_templates_are_public_safe(self):
        for name in names() + ['ORDER']:
            path = os.path.join(FOLDER, name + ('.txt' if name == 'ORDER' else '.env'))
            with open(path, encoding='utf-8') as handle:
                content = handle.read()
            found = BANNED.search(content)
            self.assertIsNone(found, (name, found and found.group(0)))
            if BANNED_PRIVATE is not None:
                self.assertIsNone(BANNED_PRIVATE.search(content), name)
        with open(os.path.join(FOLDER, 'ORDER.txt'), 'rb') as handle:
            self.assertNotIn(b'\r', handle.read())


if __name__ == '__main__':
    unittest.main()
