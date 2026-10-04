"""tp4/packed-prefix, stage 1: the sticky-session twins of the eight-seat 262k gate profiles.

Each twin is its no-reuse parent plus exactly the prefix deltas (QWEN_PREFIX_REUSE, QWEN_PREFIX_STORE_GIB,
QWEN_FAST_STICKY_SESSIONS in the environment; enable-prefix-caching, enable-chunked-prefill and the sha256 block-hash chain
in place of the two no- flags in the argv) and nothing else, is gate-only, passes the contract's prefix and sticky checks, and
leaves its parent (the control) byte-for-byte what it was: no reuse. The production default and every traffic profile stay
without the switches. The KV-reservation rule the pool relies on is checked as arithmetic: a request's reservation counts the
blocks it shares, so sharing can only over-reserve."""

import copy
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_kv_reservation as reservation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
PAIRS = (('c2-packed-tp4-8x262k-prefix-gate', 'c2-packed-tp4-8x262k-best'),
         ('c2-packed-tp4-8x262k-prefix-time-gate', 'c2-packed-tp4-8x262k-best-time-gate'))
SWITCHES = {'QWEN_PREFIX_REUSE': '1', 'QWEN_PREFIX_STORE_GIB': '8', 'QWEN_FAST_STICKY_SESSIONS': '1'}
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')


def document():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def profile(name):
    return dict(document()['profiles'][name], name=name)


class TwinTests(unittest.TestCase):
    def test_each_twin_is_its_parent_with_the_prefix_deltas_only(self):
        for name, parent in PAIRS:
            with self.subTest(profile=name):
                mine, theirs = profile(name), profile(parent)
                self.assertEqual(mine['env'], dict(theirs['env'], **SWITCHES))
                engine = dict(theirs['engine'])
                self.assertIs(engine.pop('no-enable-prefix-caching'), True)
                self.assertIs(engine.pop('no-enable-chunked-prefill'), True)
                engine.update({'enable-prefix-caching': True, 'enable-chunked-prefill': True,
                               'prefix-caching-hash-algo': 'sha256'})
                self.assertEqual(mine['engine'], engine)
                for key in set(mine) | set(theirs):
                    if key not in ('env', 'engine', 'description', 'name'):
                        self.assertEqual(mine.get(key), theirs.get(key), key)

    def test_the_twins_are_gate_only_and_carry_the_waiver_and_the_gate_marker_of_their_parents(self):
        for name, parent in PAIRS:
            with self.subTest(profile=name):
                mine = profile(name)
                self.assertIs(mine.get('gate_only'), True)
                self.assertEqual(mine['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertTrue(contract.gate_problems(mine, {}), 'a gate-only profile boots only inside a gate')
                self.assertEqual(contract.gate_problems(mine, {contract.GATE_SWITCH: '1'}), [])
                self.assertNotIn('max_prompt_tokens', mine, 'gate limits: no prompt cap, as the parent')

    def test_the_contract_accepts_each_twin_as_sticky_and_refuses_it_with_a_switch_taken_out(self):
        for name, _ in PAIRS:
            with self.subTest(profile=name):
                mine = profile(name)
                self.assertEqual(contract.prefix_reuse_problems(mine), [])
                self.assertTrue(contract.prefix_reuse(mine) and contract.sticky_sessions(mine))
                self.assertEqual(mine['env']['QWEN_FAST_TP'], '4')
                self.assertEqual(mine['env']['QWEN_FAST_M3_BLOCKS'], '2')
                self.assertEqual(mine['env']['QWEN_FAST_KV_RESERVATION'], '1')
                self.assertGreaterEqual(mine['engine']['max-num-batched-tokens'], mine['engine']['max-model-len'])
                for key in SWITCHES:
                    broken = copy.deepcopy(mine)
                    broken['env'].pop(key)
                    if key == 'QWEN_PREFIX_STORE_GIB':
                        continue            # a size, not a switch: the registry's default stands
                    self.assertTrue(contract.prefix_reuse_problems(broken), key)

    def test_the_argv_turns_prefix_caching_on_beside_dflash_at_eight_seats(self):
        for name, _ in PAIRS:
            with self.subTest(profile=name):
                argv = contract.engine_arguments(profile(name), '/snap')
                for flag in ('--enable-prefix-caching', '--enable-chunked-prefill', '--no-async-scheduling',
                             '--speculative-config'):
                    self.assertIn(flag, argv)
                for flag in ('--no-enable-prefix-caching', '--no-enable-chunked-prefill'):
                    self.assertNotIn(flag, argv)
                self.assertEqual(argv[argv.index('--prefix-caching-hash-algo') + 1], 'sha256')
                self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '8')
                self.assertEqual(argv[argv.index('--max-model-len') + 1], '262144')

    def test_the_environment_keeps_the_switches_for_the_twin_and_drops_them_under_the_control(self):
        for name, parent in PAIRS:
            with self.subTest(profile=name):
                environ = contract.apply_environment(profile(name), {})
                self.assertEqual((environ['QWEN_PREFIX_REUSE'], environ['QWEN_FAST_STICKY_SESSIONS']), ('1', '1'))
                control = contract.apply_environment(profile(parent), {'QWEN_PREFIX_REUSE': '1',
                                                                       'QWEN_FAST_STICKY_SESSIONS': '1'})
                self.assertNotIn('QWEN_PREFIX_REUSE', control)
                self.assertNotIn('QWEN_FAST_STICKY_SESSIONS', control)


class ControlTests(unittest.TestCase):
    def test_the_controls_have_no_reuse_and_the_twins_are_the_only_new_sticky_profiles(self):
        for name, parent in PAIRS:
            with self.subTest(profile=parent):
                control = profile(parent)
                self.assertFalse(contract.prefix_reuse(control))
                self.assertFalse(contract.sticky_sessions(control))
                self.assertIs(control['engine'].get('no-enable-prefix-caching'), True)
                self.assertFalse(prefix_gate.is_prefix_profile(control))
                self.assertTrue(prefix_gate.is_sticky_profile(profile(name)))
        found = document()
        self.assertEqual(found['default'], 'c2-packed-tp4')
        traffic = [name for name, body in found['profiles'].items()
                   if body.get('gate_only') is not True and body.get('mesh_device') == 'P150x4'
                   and contract.sticky_sessions(dict(body, name=name))]
        self.assertEqual(traffic, [], 'no four-card traffic profile turns sticky sessions on in stage 1')

    def test_the_twins_and_their_parents_are_on_the_same_four_card_mesh(self):
        for name, parent in PAIRS:
            self.assertEqual(profile(name)['mesh_device'], profile(parent)['mesh_device'])
            self.assertEqual(profile(name)['mesh_graph_descriptor'], profile(parent)['mesh_graph_descriptor'])
            self.assertEqual(profile(name)['mesh_device'], 'P150x4')

    def test_the_descriptions_are_public_safe_and_say_what_is_unverified(self):
        for name, _ in PAIRS:
            text = profile(name)['description']
            self.assertIsNone(BANNED.search(text), name)
            self.assertIn('UNVERIFIED', text)
            self.assertIn('GATE ONLY', text)
            self.assertIn('byte-identical to a cold prefill', text)


class PlanTests(unittest.TestCase):
    def arms(self, plan, profile_name, baseline):
        return prefix_gate.plan_arms(plan, profile_name, baseline, document())

    def test_the_agent_turn_plan_runs_the_twin_and_its_control_on_eight_seats(self):
        arms = self.arms('agent-turns', PAIRS[1][0], PAIRS[1][1])
        self.assertEqual([arm['arm'] for arm in arms], ['agent-turns-prefix', 'agent-turns-baseline'])
        self.assertEqual([arm['served'] for arm in arms], [PAIRS[1][0], PAIRS[1][1]])
        self.assertEqual([arm['prefix'] for arm in arms], [True, False])
        self.assertEqual([arm['seats'] for arm in arms], [8, 8])
        self.assertEqual([arm['sticky'] for arm in arms], [True, False])
        for arm in arms:
            self.assertEqual(arm['scenario'], 'agent_turns')
            self.assertEqual(arm['env'], (), 'a timed arm carries no audit environment')
            self.assertFalse(prefix_gate.wants_digests(arm), 'no row digests on a timed arm')
            self.assertEqual(arm['context'], 262144)

    def test_each_arm_is_a_plan_of_its_own_for_the_abab_jobs(self):
        for plan, served in (('agent-turns-prefix', PAIRS[1][0]), ('agent-turns-baseline', PAIRS[1][1])):
            arms = self.arms(plan, PAIRS[1][0], PAIRS[1][1])
            self.assertEqual([(arm['arm'], arm['served']) for arm in arms], [(plan, served)])

    def test_the_audited_arms_keep_their_audit_environment_and_the_eager_arm_runs_the_whole_set(self):
        arms = self.arms('exactness-eager', PAIRS[0][0], PAIRS[0][1])
        self.assertEqual(len(arms), 1)
        self.assertTrue(arms[0]['full'] and arms[0]['sticky'] and arms[0]['s2'])
        self.assertTrue(arms[0]['env'], 'the extent audit is on')
        self.assertEqual(arms[0]['seats'], 8)
        shared = self.arms('exactness-shared', PAIRS[0][0], PAIRS[0][1])
        self.assertEqual([arm['kind'] for arm in shared], ['audit'])
        self.assertEqual(shared[0]['seats'], 8)

    def test_the_traced_arm_and_the_tiny_pool_do_not_apply(self):
        skipped = dict(prefix_gate.not_applicable('exactness', PAIRS[0][0], document()))
        self.assertIn('exactness-traced', skipped)
        self.assertNotIn('exactness-shared', skipped)
        self.assertIn('lifecycle-tiny', dict(prefix_gate.not_applicable('lifecycle', PAIRS[0][0], document())))

    def test_a_control_that_turns_reuse_on_is_refused(self):
        with self.assertRaises(prefix_gate.PlanError):
            self.arms('agent-turns', PAIRS[1][0], PAIRS[0][0])


class ReservationArithmeticTests(unittest.TestCase):
    """The KV reservation under prefix caching (design 3.3 item 2, the arithmetic half; the real-scheduler proof is named below):
    a request's reservation r counts
    every block of its life, the blocks it shares with another conversation included, and a shared block is one physical
    block, so the pool's physical use is at most the sum of the running reservations. This class is the arithmetic only; the
    real-scheduler proof with the prefix cache on is test_qwen_prefix_scheduler_vllm.StickyReservationOnRealVllmTests (the
    graft's sticky scheduler, DFlash lookahead 16, hits and siblings, a pool smaller than the traffic, a negative control; run by
    qwen-fast-vllm-cpu.yml) and the cards' half is the lifecycle job L1 of references/tp4-packed-prefix-jobs."""

    POOL = 19967

    def test_sharing_can_only_over_reserve(self):
        agents = [(49000, 128), (48000, 128), (30000, 128), (120000, 128)]
        reserved = [reservation.request_blocks(prompt, answer) for prompt, answer in agents]
        shared_prefix_blocks = 2048 // 64 * 3          # three conversations of one tenant share a 6,144-token prefix
        physical = sum(reserved) - 2 * shared_prefix_blocks
        self.assertLess(physical, sum(reserved))
        self.assertLessEqual(sum(reserved), self.POOL)

    def test_the_rule_is_unchanged_by_the_profile_pair(self):
        for name, parent in PAIRS:
            self.assertEqual(profile(name)['env']['QWEN_FAST_KV_RESERVATION'], profile(parent)['env']['QWEN_FAST_KV_RESERVATION'])
            self.assertEqual(profile(name)['engine']['num-gpu-blocks-override'],
                             profile(parent)['engine']['num-gpu-blocks-override'])
        self.assertEqual(reservation.request_blocks(253920, 8192), -(-(253920 + 8192 + 32) // 64) + 1)


if __name__ == '__main__':
    unittest.main()
