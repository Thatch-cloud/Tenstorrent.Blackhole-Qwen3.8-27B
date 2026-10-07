"""The tp4/next-3-cheap window: its three cheap levers' profiles, its job templates (scripts/ci/references/tp4-next3-cheap-jobs) and their order.

Levers (TP4 performance audit, synthesis section 4): V6 the ring fabric (c2-packed-tp4-speed-strace-ring against its control, the audited
exactness arm c2-packed-tp4-gate-ring), D2 the drafter's wide norms (c2-packed-tp4-speed-strace-draftwide, QWEN_FAST_TP4_DRAFT_WIDE), V7 the
TP4 re-sweep of the gate/up 64-row matmul grids (one-card sweep, applies nothing). Every new behaviour is off in production's profile; each
timing twin is its control plus exactly its deltas; the window opens with 'agentstop unserve' and hands back with a reset, a fabric re-measure
and agentstart (never a deploy). The templates are public, so they name no rig, card, address, registry or digest.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402
import draft_wide_tp  # noqa: E402
import mesh_link_policy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-next3-cheap-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-next-3m'
CONTROL, PRODUCTION = 'c2-packed-tp4-speed-strace', 'c2-packed-tp4'
RING, WIDE, AUDITED_RING = 'c2-packed-tp4-speed-strace-ring', 'c2-packed-tp4-speed-strace-draftwide', 'c2-packed-tp4-gate-ring'
BUILD, DRAFT_EXACT = 'B0-build', 'D1-draftwide-exact-audited'
FIRST, SWEEP, EXACT, HANDBACK = 'A0-agentstop-unserve', 'V7-tp4-matmul-sweep', 'E1-ring-exact-audited', 'H1-handback-reset-fabric-agentstart'
RING_PAIR = ('TR1-control-timed', 'TR2-ring-timed', 'TR3-control-timed', 'TR4-ring-timed')
WIDE_PAIR = ('TW1-control-timed', 'TW2-draftwide-timed', 'TW3-control-timed', 'TW4-draftwide-timed')
TIMED = RING_PAIR + WIDE_PAIR
ORDERED = (BUILD, FIRST, SWEEP, EXACT) + RING_PAIR + (DRAFT_EXACT,) + WIDE_PAIR + (HANDBACK,)
AUDITED_WIDE = 'c2-packed-tp4-gate-draftwide'
TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code_equal', 'concurrent4_code_32k']
RING_FLAG, WIDE_FLAG = 'QWEN_FAST_CCL_TOPOLOGY', 'QWEN_FAST_TP4_DRAFT_WIDE'


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


def body(name):
    return PROFILES['profiles'][name]


def env_of(name):
    return body(name)['env']


def differing(left, right):
    return {key for key in set(left) | set(right) if left.get(key) != right.get(key)}


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [row[0] for row in rows]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(ordered, list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        modes = {}
        for name, mode, image, minutes in read_order():
            modes[name] = mode
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 180, minutes)
        for name in (BUILD, FIRST, HANDBACK):
            self.assertEqual(modes[name], 'stop', name)
        for name in (SWEEP, EXACT, DRAFT_EXACT) + TIMED:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_first_job_takes_production_down_and_nothing_else_does(self):
        self.assertEqual([row[0] for row in read_order()[:2]], [BUILD, FIRST], 'the build runs first, production still serving')
        self.assertEqual(parsed(FIRST)['actions'], 'status agentstop unserve')
        actions = parsed(FIRST)['actions'].split()
        self.assertLess(actions.index('agentstop'), actions.index('unserve'))
        for name in ORDERED:
            if name != FIRST:
                self.assertFalse({'agentstop', 'unserve'} & set(parsed(name)['actions'].split()), name)
        for word in ('PRODUCTION IS LIVE ON THE CARDS', 'A0 (agentstop, unserve) is the first job'):
            self.assertIn(word, order_text())

    def test_the_hand_back_is_last_resets_remeasures_the_fabric_restarts_the_agent_and_never_deploys(self):
        self.assertEqual(read_order()[-1][0], HANDBACK)
        actions = parsed(HANDBACK)['actions'].split()
        self.assertEqual(actions, ['status', 'reset', 'fabric', 'agentstart'])
        self.assertEqual(parsed(HANDBACK)['cards'], 'quad')
        for word in ("TODAY'S PRODUCTION RECIPE", 'NEVER place a gate arm', 'AN OWNER /deploy IS STILL NEEDED', 'fabric re-measure',
                     'stops before agentstart', 'c2-packed-tp4', 'audits off', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1'):
            self.assertIn(word, text_of(HANDBACK))
        order = order_text()
        for word in ('runs LAST and even when any earlier job failed or hung', 'An owner /deploy is still needed', 'never includes a deploy'):
            self.assertIn(word, order)
        for name in ORDERED:
            self.assertNotIn('push', parsed(name)['actions'].split(), 'no :latest retag, no publish')
            self.assertFalse(re.search(r'(?im)^C2_(PLACE|DEPLOY)', text_of(name)), name)

    def test_the_order_runs_the_sweep_then_the_audited_arm_then_the_timing_pairs(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[:4], [BUILD, FIRST, SWEEP, EXACT])
        self.assertEqual(names[4:8], list(RING_PAIR))
        self.assertEqual(names[8], DRAFT_EXACT)
        self.assertEqual(names[9:13], list(WIDE_PAIR))
        for word in ('ABAB', 'NOTHING COMBINED HAS RUN ON A CARD', 'if E1 fails, skip TR1-TR4', 'PAIRED per round', 'V7',
                     'applies nothing', 'If D1 fails or hangs, skip TW1-TW4'):
            self.assertIn(word, order_text())


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_digest_or_placeholder(self):
        for name in ORDERED:
            with self.subTest(template=name):
                parsed(name)
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        self.assertNotIn('\r', order_text())
        self.assertIsNone(BANNED.search(order_text()))

    def test_every_four_card_job_resets_first_and_opens_the_cards_in_one_step(self):
        for name in ORDERED:
            outputs = parsed(name)
            if outputs['cards'] != 'quad' or name in (BUILD, FIRST, HANDBACK):
                continue
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card job follows a reset')

    def test_only_the_first_job_builds_the_image_and_it_opens_no_card_and_runs_before_production_is_taken_down(self):
        for name in ORDERED:
            self.assertEqual('build' in parsed(name)['actions'].split(), name == BUILD, name)
        outputs = parsed(BUILD)
        self.assertEqual((outputs['actions'], outputs['cards'], outputs['tag']), ('build', 'quad', IMAGE))
        self.assertFalse(set(outputs['actions'].split()) & set(DEVICE_STEPS + ('reset', 'agentstop', 'unserve', 'agentstart', 'push')))
        for word in ('BEFORE A0', 'FRESH', 'already exists', 'PLACEHOLDER', 'production still serves'):
            self.assertIn(word, text_of(BUILD))
        self.assertIn('FRESH tag', order_text())
        self.assertIn('B0', order_text())

    def test_the_audited_draftwide_arm_is_the_gate_plus_the_flag_and_runs_before_the_draftwide_timing_pair(self):
        outputs = parsed(DRAFT_EXACT)
        self.assertEqual((outputs['actions'], outputs['profile'], outputs['gate_plan']), ('reset gate', AUDITED_WIDE, 'matrix'))
        self.assertEqual((outputs['gate_lengths'], outputs['gate_max_tokens']), ('4096,16384,32768,60000', '512'))
        gate, audited = body('c2-packed-tp4-gate'), body(AUDITED_WIDE)
        self.assertEqual(differing(audited['env'], gate['env']), {WIDE_FLAG})
        self.assertEqual(audited['env'][WIDE_FLAG], '1')
        self.assertEqual((audited['env']['QWEN_FAST_VERIFY_T1_AUDIT'], audited['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
        for key in set(gate) | set(audited):
            if key not in ('env', 'description'):
                self.assertEqual(gate.get(key), audited.get(key), key)
        self.assertIs(audited.get('gate_only'), True)
        self.assertTrue(audited['description'].startswith('GATE ONLY'))
        self.assertNotIn(WIDE_FLAG, body(PRODUCTION)['env'])
        names = [row[0] for row in read_order()]
        for name in WIDE_PAIR:
            self.assertLess(names.index(DRAFT_EXACT), names.index(name))
        for word in ('INSIDE the drafter pair traces', 'SKIP TW1-TW4'):
            self.assertIn(word, text_of(DRAFT_EXACT))

    def test_only_the_sweep_is_a_one_card_job_and_its_harness_is_the_sweeps(self):
        self.assertEqual([name for name in ORDERED if parsed(name)['cards'] == 'pair'], [SWEEP])
        outputs = parsed(SWEEP)
        self.assertEqual(outputs['actions'], 'cardm')
        self.assertEqual(outputs['cardm_harness'], 'optimisation/ttnn-op/matmul_tp4_sweep/run_card_m.sh')
        self.assertIn('--shapes mlp_w1,mlp_w3', outputs['cardm_args'])
        self.assertEqual(outputs['cardm_env'], 'IMAGE_TAG=%s' % IMAGE)
        for word in ('NOTHING IS APPLIED', 'best EXACT config', 'byte for byte'):
            self.assertIn(word, text_of(SWEEP))

    def test_the_audited_arm_is_the_ring_gate_arm_on_the_s3a_matrix(self):
        outputs = parsed(EXACT)
        self.assertEqual((outputs['actions'], outputs['profile']), ('reset gate', AUDITED_RING))
        self.assertEqual((outputs['gate_plan'], outputs['gate_lengths'], outputs['gate_max_tokens']),
                         ('matrix', '4096,16384,32768,60000', '512'))
        audits = env_of(AUDITED_RING)
        self.assertEqual((audits['QWEN_FAST_VERIFY_T1_AUDIT'], audits['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))

    def test_the_four_timing_jobs_of_each_pair_run_the_same_coding_tests_4x4k_and_4x32k(self):
        for name in TIMED:
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['tests'].split(','), TIMING_TESTS)
                self.assertEqual(outputs['actions'], 'reset smoke')
                self.assertEqual(outputs['gate_plan'], 'bringup')

    def test_each_pair_is_a_b_a_b_of_the_control_against_its_twin(self):
        self.assertEqual([parsed(name)['profile'] for name in RING_PAIR], [CONTROL, RING, CONTROL, RING])
        self.assertEqual([parsed(name)['profile'] for name in WIDE_PAIR], [CONTROL, WIDE, CONTROL, WIDE])
        for name in TIMED:
            self.assertIs(body(parsed(name)['profile']).get('gate_only'), True, name)


class ProfileTests(unittest.TestCase):
    def test_the_ring_twin_is_the_control_plus_exactly_the_two_ring_deltas(self):
        ring, control = body(RING), body(CONTROL)
        self.assertEqual(differing(ring['env'], control['env']), {RING_FLAG})
        self.assertEqual(ring['env'][RING_FLAG], 'ring')
        self.assertEqual(ring['engine']['additional-config']['tt']['fabric_config'], 'FABRIC_1D_RING')
        self.assertEqual(control['engine']['additional-config']['tt']['fabric_config'], 'FABRIC_1D')
        mine = json.loads(json.dumps(ring))
        theirs = json.loads(json.dumps(control))
        mine['engine']['additional-config']['tt'].pop('fabric_config')
        theirs['engine']['additional-config']['tt'].pop('fabric_config')
        self.assertEqual(mine['engine'], theirs['engine'], 'the engine differs in the fabric alone')
        for key in set(ring) | set(control):
            if key not in ('description', 'env', 'engine'):
                self.assertEqual(ring.get(key), control.get(key), key)
        self.assertEqual(ring['mesh_graph_descriptor'], control['mesh_graph_descriptor'], 'the same ring descriptor')
        self.assertTrue(ring['description'].startswith('GATE ONLY'))

    def test_the_ring_twin_has_the_audited_arms_ring_deltas_and_the_audited_arm_the_audits(self):
        """The audited exactness arm (c2-packed-tp4-gate-ring) and the timing twin move the same two settings from their own bases."""
        gate, audited = body('c2-packed-tp4-gate'), body(AUDITED_RING)
        self.assertEqual(differing(audited['env'], gate['env']), {RING_FLAG})
        self.assertEqual(audited['engine']['additional-config']['tt']['fabric_config'], 'FABRIC_1D_RING')
        for flag in (RING_FLAG,):
            self.assertEqual(env_of(RING)[flag], audited['env'][flag])
        self.assertEqual(body(RING)['engine']['additional-config']['tt']['fabric_config'],
                         audited['engine']['additional-config']['tt']['fabric_config'])
        # the twin's remaining differences from the audited arm are exactly the audits-off recipe's (audits, caps, in-trace sampler)
        self.assertEqual(differing(env_of(RING), audited['env']), {'QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_FAST_BUDGET_CAP',
                                                                   'QWEN_FAST_SEQ_DEADLINE_S', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE'})

    def test_the_draftwide_twin_is_the_control_plus_exactly_the_flag(self):
        wide, control = body(WIDE), body(CONTROL)
        self.assertEqual(differing(wide['env'], control['env']), {WIDE_FLAG})
        self.assertEqual(wide['env'][WIDE_FLAG], '1')
        for key in set(wide) | set(control):
            if key not in ('description', 'env'):
                self.assertEqual(wide.get(key), control.get(key), key)
        self.assertTrue(wide['description'].startswith('GATE ONLY'))
        self.assertEqual(wide['env']['QWEN_FAST_TP'], '4', 'the flag is refused at the pair')
        self.assertEqual(draft_wide_tp.FLAG, WIDE_FLAG)
        self.assertTrue(draft_wide_tp.enabled(wide['env']))
        self.assertFalse(draft_wide_tp.enabled(control['env']))

    def test_both_twins_carry_the_audits_off_recipe_the_hang_fix_and_the_caps_of_production(self):
        production = env_of(PRODUCTION)
        for name in (RING, WIDE):
            env = env_of(name)
            with self.subTest(profile=name):
                self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
                self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
                for flag in ('QWEN_FAST_BUDGET_CAP', 'QWEN_FAST_SEQ_DEADLINE_S'):
                    self.assertEqual(env[flag], production[flag])
                self.assertEqual(env['QWEN_C2_GATE_PROFILE'], '1')

    def test_production_carries_none_of_the_new_flags_and_stays_the_control_less_the_gate_marker(self):
        production, control = body(PRODUCTION), body(CONTROL)
        self.assertNotIn(WIDE_FLAG, production['env'])
        self.assertNotIn(RING_FLAG, production['env'])
        self.assertEqual(production['engine']['additional-config']['tt']['fabric_config'], 'FABRIC_1D')
        self.assertEqual(differing(production['env'], control['env']), {'QWEN_C2_GATE_PROFILE'})
        self.assertNotIn('gate_only', production)
        self.assertEqual(PROFILES['default'], PRODUCTION)

    def test_no_other_profile_carries_the_new_flags(self):
        for name, profile in PROFILES['profiles'].items():
            with self.subTest(profile=name):
                self.assertEqual(WIDE_FLAG in profile['env'], name in (WIDE, AUDITED_WIDE))
                # tp4/w1 carries D1 (the topology flag alone, the fabric stays FABRIC_1D): test_tp4_w1
                self.assertEqual(profile['env'].get(RING_FLAG) == 'ring', name in (RING, AUDITED_RING) + ('c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-audit-digests', 'c2-packed-tp4-8x262k-ship-prefix-dckdefault', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-dbf16', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-w1'))
                self.assertEqual(profile['engine']['additional-config']['tt'].get('fabric_config') == 'FABRIC_1D_RING',
                                 name in (RING, AUDITED_RING, 'general-tp4-ring-mmrs'))

    def test_the_new_profiles_are_gate_only_and_never_traffic(self):
        for name in (RING, WIDE):
            self.assertIs(body(name).get('gate_only'), True, name)
            self.assertIn('UNVERIFIED on hardware', body(name)['description'])
            self.assertIn('Not for traffic', body(name)['description'])

    def test_the_ring_topology_switch_reads_ring_at_four_cards_and_linear_by_default(self):
        class Ops:
            class Topology:
                Ring, Linear = 'ring', 'linear'

        self.assertEqual(mesh_link_policy.fast_ccl_topology(Ops, env_of(RING)), 'ring')
        self.assertEqual(mesh_link_policy.fast_ccl_topology(Ops, env_of(CONTROL)), 'linear')


class SmokeCheckTests(unittest.TestCase):
    """c2_smoke_check: the draftwide arm's markers fail a run that did not exercise the lever."""

    def wide_problems(self, log, profile=WIDE):
        problems, _ = c2_smoke_check.check('', log, False, env=env_of(profile))
        return [problem for problem in problems if 'wide' in problem]

    def test_an_engaged_line_passes(self):
        self.assertEqual(self.wide_problems('%s site=mlp rows=64 cores=20' % draft_wide_tp.ENGAGED), [])

    def test_no_engaged_line_fails_the_arm_that_asked_for_the_lever(self):
        found = self.wide_problems('')
        self.assertEqual(len(found), 1)
        self.assertIn(WIDE_FLAG, found[0])

    def test_a_fall_back_line_fails_even_beside_an_engaged_one(self):
        log = '%s site=mlp\n%s site=final shape x' % (draft_wide_tp.ENGAGED, draft_wide_tp.FALLBACK)
        found = self.wide_problems(log)
        self.assertEqual(len(found), 1)
        self.assertIn('fell back', found[0])

    def test_the_control_is_not_asked_for_the_marker(self):
        self.assertEqual(self.wide_problems('', CONTROL), [])
        self.assertEqual(self.wide_problems('', RING), [])

    def test_the_markers_are_the_modules(self):
        self.assertEqual(c2_smoke_check.DRAFT_WIDE_ENGAGED, draft_wide_tp.ENGAGED)
        self.assertEqual(c2_smoke_check.DRAFT_WIDE_FELL_BACK, draft_wide_tp.FALLBACK)
        self.assertEqual(c2_smoke_check.DRAFT_WIDE_FLAG, draft_wide_tp.FLAG)


if __name__ == '__main__':
    unittest.main()
