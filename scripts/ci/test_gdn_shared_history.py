"""QWEN_FAST_GDN_SHARED_HISTORY: the pool, the plumbing through the real four-card launch, the ledger arithmetic, the refusal paths
and the profile arithmetic (docs/tp4-gdn-shared-history.md). The schedule and the exactness proofs are in
test_gdn_shared_history_schedule.py.

Nothing here touches a card: tensors are recording fakes whose addresses are their identity."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_shared_history as shared
import gdn_user_batch_tp as quad
import memory_ledger
from memory_ledger import MemoryLedger
import serving_c2_contract as contract
import tp_addresses
import tp_shapes
import test_gdn_user_batch_tp as quad_tests
from test_gdn_user_batch import KERNELS

PROFILES = json.loads((Path(__file__).with_name('qwen_c2_profiles.json')).read_text(encoding='utf-8'))['profiles']
CONTROL = 'c2-packed-tp4-8x262k-ship-prefix-4e-control-audit'
ARM = 'c2-packed-tp4-8x262k-ship-prefix-4e-audit'
PLAIN = 'c2-packed-tp4-8x262k-ship-prefix-4e'
GROW = 'c2-packed-tp4-8x262k-ship-prefix-4e-grow'
SHIP = 'c2-packed-tp4-8x262k-ship-prefix'
SHIP_AUDIT = 'c2-packed-tp4-8x262k-ship-prefix-audit'
MIB = 2 ** 20
FOUR = {'QWEN_FAST_TP': '4', 'QWEN_FAST_M3_BLOCKS': '2'}


def block(users=4, rows=16, block_rows=None):
    return SimpleNamespace(users=users, rows_per_user=rows, block_rows=users * rows if block_rows is None else block_rows,
                           phase='idle', pending_segments=set(), deferred_commits=[], flush_commits=lambda site: 0)


class Clean(unittest.TestCase):
    def setUp(self):
        shared.reset()
        self.addCleanup(shared.reset)
        patcher = patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for flag in (*shared.FLAGS, 'QWEN_FAST_GDN_SEQ_BLOCK', 'QWEN_FAST_TP', 'QWEN_FAST_M3_BLOCKS'):
            os.environ.pop(flag, None)


class FlagTests(Clean):
    def test_the_flags_are_strict_and_default_off(self):
        self.assertEqual((shared.enabled({}), shared.grow_enabled({})), (False, False))
        self.assertEqual((shared.enabled({shared.FLAG: '1'}), shared.grow_enabled({shared.GROW_FLAG: '1'})), (True, True))
        for value in ('true', '', '2', 'yes'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, shared.FLAG):
                    shared.enabled({shared.FLAG: value})
                with self.assertRaisesRegex(ValueError, shared.GROW_FLAG):
                    shared.grow_enabled({shared.GROW_FLAG: value})

    def test_requested_is_the_cheap_test_the_verifier_makes_before_importing_anything(self):
        self.assertFalse(shared.requested({}))
        self.assertFalse(shared.requested({shared.FLAG: '0', shared.GROW_FLAG: '0'}))
        self.assertTrue(shared.requested({shared.FLAG: '1'}))
        self.assertTrue(shared.requested({shared.GROW_FLAG: '1'}))
        self.assertTrue(shared.requested({shared.FLAG: 'garbage'}), 'a malformed value is not off: it reaches the strict parser')

    def test_the_refusals_name_the_flag_and_the_missing_condition(self):
        on = {shared.FLAG: '1'}
        self.assertIsNone(shared.refusal({}))
        self.assertIsNone(shared.refusal({**on, **FOUR}))
        self.assertIn('QWEN_FAST_TP=4', shared.refusal({**on, 'QWEN_FAST_M3_BLOCKS': '2'}))
        self.assertIn('QWEN_FAST_TP=4', shared.refusal({**on, 'QWEN_FAST_TP': '2', 'QWEN_FAST_M3_BLOCKS': '2'}))
        self.assertIn('QWEN_FAST_M3_BLOCKS=2', shared.refusal({**on, 'QWEN_FAST_TP': '4'}))
        self.assertIn('QWEN_FAST_M3_BLOCKS=2', shared.refusal({**on, 'QWEN_FAST_TP': '4', 'QWEN_FAST_M3_BLOCKS': '1'}))
        # K5-A is the served launch (the image bakes it): the lever rides on it, so it is not a refusal
        self.assertIsNone(shared.refusal({**on, **FOUR, 'QWEN_FAST_GDN_SEQ_BLOCK': '1'}))
        self.assertIsNone(shared.refusal({**on, **FOUR, 'QWEN_FAST_GDN_SEQ_BLOCK': '0'}))
        self.assertIn('QWEN_FAST_GDN_SPLIT_V', shared.refusal({**on, **FOUR, 'QWEN_FAST_GDN_SPLIT_V': '2'}))
        self.assertIn('needs %s=1' % shared.FLAG, shared.refusal({shared.GROW_FLAG: '1', **FOUR}))
        with self.assertRaises(ValueError):
            shared.refusal({shared.FLAG: 'maybe'})


class JoinTests(Clean):
    def test_with_both_flags_off_a_block_gets_no_pool(self):
        lines = []
        self.assertIsNone(shared.join(block(), log=lines.append, environ={}))
        self.assertEqual((lines, shared.current()), ([], None))

    def test_an_environment_that_cannot_honour_the_flag_is_refused_and_logged_before_anything_attaches(self):
        lines = []
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_M3_BLOCKS=2'):
            shared.join(block(), log=lines.append, environ={shared.FLAG: '1', 'QWEN_FAST_TP': '4'})
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(shared.REFUSED_MARKER + ' reason='))
        self.assertIsNone(shared.current())

    def test_the_growth_marker_alone_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'needs %s=1' % shared.FLAG):
            shared.join(block(), environ={shared.GROW_FLAG: '1', **FOUR})

    def test_two_blocks_attach_a_third_and_a_misshapen_one_do_not(self):
        env = {shared.FLAG: '1', **FOUR}
        first, second = block(), block()
        pool = shared.join(first, environ=env)
        self.assertIs(shared.join(second, environ=env), pool)
        self.assertEqual(len(pool.blocks), 2)
        with self.assertRaisesRegex(ValueError, 'third block'):
            shared.join(block(), environ=env)
        with self.assertRaisesRegex(ValueError, 'already attached'):
            pool.attach(first)
        shared.reset()
        for odd in (block(users=2, rows=16), block(users=4, rows=8), block(users=4, rows=16, block_rows=32)):
            with self.subTest(users=odd.users, rows=odd.rows_per_user, block_rows=odd.block_rows):
                with self.assertRaisesRegex(ValueError, 'built for 4-user'):
                    shared.join(odd, environ=env)

    def test_the_solo_lane_block_never_shares(self):
        env = {shared.FLAG: '1', **FOUR}
        self.assertIsNone(shared.join(block(users=1, rows=16), environ=env))
        self.assertIsNone(shared.current())


class AccountingTests(unittest.TestCase):
    """The research's numbers, from the geometry, not copied."""

    def test_one_user_is_288_mib_a_block_1_125_gib_and_two_blocks_free_one_of_them(self):
        self.assertEqual(shared.history_bytes_per_user(), 288 * MIB)
        self.assertEqual(shared.history_bytes_per_block(), 1152 * MIB)
        self.assertEqual(shared.freed_bytes(2), 1152 * MIB)
        self.assertEqual(shared.freed_bytes(2), 1207959552)
        self.assertEqual(shared.freed_bytes(1), 0)
        self.assertEqual(2 * shared.history_bytes_per_block(), 2304 * MIB)       # the 2.25 GiB both blocks hold today
        self.assertEqual(shared.TENSORS, 192)
        self.assertEqual(shared.TENSORS * 16 * 12 * 128 * 128 * 2, shared.history_bytes_per_block())

    def test_the_freed_bytes_hold_2168_kv_blocks_and_the_pool_edge_moves_to_22144(self):
        self.assertEqual(shared.kv_blocks_gained(), 2168)
        self.assertEqual(shared.PRODUCTION_POOL_EDGE + 2168, 22147)
        self.assertEqual(shared.grown_pool_edge(), 22144)
        self.assertEqual(shared.pool_tokens_for(22144), 1416704)
        self.assertEqual(-(-1416704 // 64) + 8, 22144)

    def test_four_full_windows_today_and_five_with_the_freed_bytes(self):
        self.assertEqual(shared.FULL_WINDOW_BLOCKS, 4097)
        self.assertEqual(shared.windows_that_fit(shared.PRODUCTION_POOL_BLOCKS), 4)
        self.assertEqual(shared.windows_that_fit(22144), 5)
        self.assertEqual(shared.windows_that_fit(20485), 4, 'five reservations are 20,485 and the null block is one more')
        self.assertEqual(shared.windows_that_fit(20486), 5)
        self.assertEqual(shared.windows_that_fit(22144 + 4097), 6)

    def test_the_reservation_admission_agrees_with_the_window_block_count(self):
        from serving_kv_reservation import request_blocks

        # the admission's worst case for the traffic profile's longest request (253,920 prompt tokens and 8,192 answer tokens) is
        # within a full window's blocks plus the lookahead and the spare, so five of them are what the grown pool is sized for
        self.assertLessEqual(request_blocks(253920, 8192), shared.FULL_WINDOW_BLOCKS + 2)


class ProfileTests(unittest.TestCase):
    def test_no_production_profile_names_either_flag(self):
        named = [name for name, profile in PROFILES.items() if any(flag in profile.get('env', {}) for flag in shared.FLAGS)]
        self.assertEqual(sorted(named), sorted([ARM, PLAIN, GROW]))
        self.assertNotIn(CONTROL, named)
        for name in named:
            self.assertTrue(PROFILES[name].get('gate_only'), name)

    def test_every_profile_passes_the_contract_check_and_the_traffic_profiles_are_untouched(self):
        for name, profile in PROFILES.items():
            with self.subTest(profile=name):
                self.assertIsNone(contract.shared_history_problem(dict(profile, name=name)))
        for name in (SHIP, SHIP_AUDIT):
            self.assertFalse(any(flag in PROFILES[name]['env'] for flag in shared.FLAGS))
            self.assertEqual(PROFILES[name]['engine']['num-gpu-blocks-override'], 19968)
            self.assertEqual(PROFILES[name]['env']['QWEN36_MAX_TOKENS_ALL_USERS'], '1277440')

    def test_the_arm_differs_from_its_control_by_the_flag_alone_and_the_control_from_the_audited_ship_by_the_ledger_line_alone(self):
        arm, control, ship = PROFILES[ARM], PROFILES[CONTROL], PROFILES[SHIP_AUDIT]
        self.assertEqual({key for key in arm['env'] if arm['env'].get(key) != control['env'].get(key)} |
                         {key for key in control['env'] if arm['env'].get(key) != control['env'].get(key)}, {shared.FLAG})
        self.assertEqual({key for key in control['env'] if control['env'].get(key) != ship['env'].get(key)} |
                         {key for key in ship['env'] if control['env'].get(key) != ship['env'].get(key)},
                         {'QWEN_FAST_MEMORY_LEDGER'})
        self.assertEqual(arm['engine'], control['engine'])
        self.assertEqual(control['engine'], ship['engine'])
        for key in set(arm) - {'env', 'description'}:
            self.assertEqual(arm[key], control[key], key)
            self.assertEqual(control[key], ship[key], key)

    def test_the_grown_profile_is_the_production_one_plus_exactly_the_flags_the_ledger_and_the_pool(self):
        grow, ship = PROFILES[GROW], PROFILES[SHIP]
        changed = {key for key in {*grow['env'], *ship['env']} if grow['env'].get(key) != ship['env'].get(key)}
        self.assertEqual(changed, {shared.FLAG, shared.GROW_FLAG, 'QWEN_FAST_MEMORY_LEDGER', 'QWEN36_MAX_TOKENS_ALL_USERS'})
        engine = {key for key in {*grow['engine'], *ship['engine']} if grow['engine'].get(key) != ship['engine'].get(key)}
        self.assertEqual(engine, {'num-gpu-blocks-override'})
        self.assertEqual((grow['engine']['num-gpu-blocks-override'], grow['env']['QWEN36_MAX_TOKENS_ALL_USERS']), (22144, '1416704'))
        self.assertTrue(grow['gate_only'])
        self.assertIsNone(contract.kv_pool_problem(grow))
        self.assertEqual(contract.kv_pool_blocks(grow), 22143)
        self.assertEqual(contract.request_limits(grow)['kv_pool_blocks'], 22143)
        self.assertEqual(shared.windows_that_fit(contract.real_pool_blocks(grow)), 5)
        self.assertEqual(shared.windows_that_fit(contract.real_pool_blocks(ship)), 4)

    def test_the_growth_marker_is_refused_when_it_does_not_hold_up(self):
        base = json.loads(json.dumps(PROFILES[GROW]))
        base['name'] = GROW

        def mutated(**changes):
            profile = json.loads(json.dumps(base))
            for path, value in changes.items():
                target, key = profile, path
                if path.startswith('env.'):
                    target, key = profile['env'], path[4:]
                elif path.startswith('engine.'):
                    target, key = profile['engine'], path[7:]
                if value is None:
                    target.pop(key, None)
                else:
                    target[key] = value
            return profile

        self.assertIsNone(shared.pool_problem(base))
        cases = {
            'a traffic profile': (dict(gate_only=None), 'not gate_only'),
            'growth without the sharing': ({'env.' + shared.FLAG: None}, 'needs %s=1' % shared.FLAG),
            'a pool over the freed bytes': ({'engine.num-gpu-blocks-override': 22208,
                                             'env.QWEN36_MAX_TOKENS_ALL_USERS': str((22208 - 8) * 64)}, 'allow 22144 at most'),
            'a pool that did not grow': ({'engine.num-gpu-blocks-override': 19968,
                                          'env.QWEN36_MAX_TOKENS_ALL_USERS': '1277440'}, 'nothing was grown'),
            'a pool off the 64 grid': ({'engine.num-gpu-blocks-override': 22100,
                                        'env.QWEN36_MAX_TOKENS_ALL_USERS': str((22100 - 8) * 64)}, 'multiple of 64'),
            'tokens that disagree': ({'env.QWEN36_MAX_TOKENS_ALL_USERS': '1277440'}, 'must be 1416704'),
            'four seats': ({'engine.max-num-seqs': 4}, 'eight-seat'),
            'a malformed flag': ({'env.' + shared.GROW_FLAG: 'yes'}, 'must be 0 or 1'),
        }
        for name, (changes, message) in cases.items():
            with self.subTest(case=name):
                problem = shared.pool_problem(mutated(**changes))
                self.assertIsNotNone(problem)
                self.assertIn(message, problem)
        # the sharing alone never grows anything, and is a gate arm
        arm = json.loads(json.dumps(PROFILES[ARM]))
        self.assertIsNone(shared.pool_problem(arm))
        arm.pop('gate_only')
        self.assertIn('not gate_only', shared.pool_problem(arm))
        # the boot check reads it
        with self.assertRaisesRegex(ValueError, 'allow 22144 at most'):
            contract.request_limits(mutated(**{'engine.num-gpu-blocks-override': 22208,
                                               'env.QWEN36_MAX_TOKENS_ALL_USERS': str((22208 - 8) * 64)}))


class PoolPlumbingTests(Clean):
    """The real four-card launch (gdn_user_batch_tp.execute) over the recording fake, inside and outside a capture."""

    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = quad_tests.FourChipTTNN()
        self.env = {shared.FLAG: '1', **FOUR}
        self.first, self.second = block(), block()
        self.pool = shared.join(self.first, environ=self.env)
        shared.join(self.second, environ=self.env)

    def launch(self, users=4, base=1000):
        groups = [quad_tests.user_inputs(index, base=base) for index in range(users)]
        with patch.dict(sys.modules, {'ttnn': self.fake}), patch('gdn_multitoken.validate_handoff_runtime'):
            return quad.execute(quad_tests.four_mesh(), groups, KERNELS, self.fake, output_memory='dram')

    def capture_layers(self, block_, layers=48, users=4):
        results = []
        with self.pool.capture(block_, self.fake):
            for layer in range(layers):
                results.append(self.launch(users=users, base=1000 + 10 * layer))
        return results

    def test_outside_a_capture_every_launch_allocates_privately_exactly_as_before(self):
        self.assertIsNone(shared.active())
        produced = self.launch(users=2)
        self.assertEqual(len(self.fake.allocated), 4)
        self.assertEqual(self.pool.tensors, [])
        self.assertFalse(any(shared.holds(states) for output, states in produced))

    def test_the_first_capture_builds_192_tensors_and_the_second_is_handed_the_same_ones(self):
        first = self.capture_layers(self.first)
        self.assertEqual(len(self.pool.tensors), 192)
        self.assertEqual(len(self.fake.allocated), 192 * 2, 'one private output and one pooled history per user per layer')
        first_states = [states for layer in first for output, states in layer]
        self.assertTrue(all(shared.holds(states) for states in first_states))
        self.assertEqual({tuple(states.shape) for states in first_states}, {(16, 12, 128, 128)})
        self.assertEqual(len({id(states) for states in first_states}), 192)
        allocated = len(self.fake.allocated)
        second = self.capture_layers(self.second)
        second_states = [states for layer in second for output, states in layer]
        self.assertEqual(len(self.fake.allocated) - allocated, 192, 'the second capture allocates its outputs and no history')
        self.assertTrue(all(a is b for a, b in zip(first_states, second_states, strict=True)))
        self.assertEqual(self.pool.counts['reused'], 192)
        self.assertEqual(self.pool.expected, 192)
        self.assertIsNone(shared.active(), 'the capture closed the pool')
        # an output is never shared
        first_outputs = {id(output) for layer in first for output, states in layer}
        second_outputs = {id(output) for layer in second for output, states in layer}
        self.assertFalse(first_outputs & second_outputs)

    def test_a_capture_that_takes_the_wrong_number_of_tensors_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'took 188 history tensors; the shared set is 192'):
            self.capture_layers(self.first, layers=47)
        shared.reset()
        self.pool = shared.join(self.first, environ=self.env)
        shared.join(self.second, environ=self.env)
        self.capture_layers(self.first)
        with self.assertRaisesRegex(ValueError, 'took 188 history tensors; the shared set is 192'):
            self.capture_layers(self.second, layers=47)
        with self.assertRaisesRegex(ValueError, 'more history tensors than the first built'):
            self.capture_layers(self.second, layers=49)

    def test_a_launch_of_another_width_inside_a_capture_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'took 0 history tensors'):      # the capture itself then ends short, as it must
            with self.pool.capture(self.first, self.fake):
                with self.assertRaisesRegex(ValueError, 'a launch asked for'):
                    self.pool.take(self.fake, quad_tests.four_mesh(), 8, 12)
                with self.assertRaisesRegex(ValueError, 'a launch asked for'):
                    self.pool.take(self.fake, quad_tests.four_mesh(), 16, 24)

    def test_the_pool_hands_out_nothing_outside_a_capture_and_only_one_capture_opens_at_a_time(self):
        with self.assertRaisesRegex(ValueError, 'only inside a verify capture'):
            self.pool.take(self.fake, quad_tests.four_mesh(), 16, 12)
        with self.assertRaisesRegex(ValueError, 'took 0 history tensors'):
            with self.pool.capture(self.first, self.fake):
                with self.assertRaisesRegex(ValueError, 'already open'):
                    with self.pool.capture(self.second, self.fake):
                        pass
        stranger = block()
        with self.assertRaisesRegex(ValueError, 'Only an attached block'):
            with self.pool.capture(stranger, self.fake):
                pass

    def test_the_block_that_built_the_set_cannot_capture_on_it_twice(self):
        self.capture_layers(self.first)
        with self.assertRaisesRegex(ValueError, 'cannot capture on it twice'):
            self.capture_layers(self.first)

    def test_a_launch_that_fails_inside_the_second_capture_frees_its_outputs_and_never_the_pools_tensors(self):
        self.capture_layers(self.first)
        freed_before = len(self.fake.freed)
        self.fake.get_device_tensors = lambda value: quad_tests.FourChipTTNN.get_device_tensors(self.fake, value)[:3]
        with self.assertRaisesRegex(ValueError, 'Every chip of the mesh'):
            with self.pool.capture(self.second, self.fake):
                self.launch(users=1)
        freed = self.fake.freed[freed_before:]
        self.assertEqual(len(freed), 1, 'the one private output')
        pooled = {states.name for states in self.pool.tensors}
        self.assertFalse(pooled & set(freed))

    def test_release_owned_leaves_the_pools_tensors_alone_and_frees_everything_else(self):
        with self.pool.capture(self.first, self.fake):
            layer = self.launch(users=4)
            for _ in range(47):
                self.launch(users=4)
        outputs = [output for output, states in layer]
        histories = [states for output, states in layer]
        ops = SimpleNamespace(get_device_tensors=self.fake.get_device_tensors, deallocate=self.fake.deallocate)
        tp_addresses.release_owned(ops, [*outputs, *histories])
        self.assertEqual(sorted(self.fake.freed), sorted(output.name for output in outputs))
        # a process that never loaded the module frees everything it is given
        with patch.dict(sys.modules, {'gdn_shared_history': None}):
            self.fake.freed.clear()
            tp_addresses.release_owned(ops, [*outputs[:1], *histories[:1]])
        self.assertEqual(sorted(self.fake.freed), sorted([outputs[0].name, histories[0].name]))

    def test_the_last_block_out_frees_each_tensor_once_and_the_first_out_frees_none(self):
        self.capture_layers(self.first)
        self.capture_layers(self.second)
        names = sorted(states.name for states in self.pool.tensors)
        lines = []
        self.pool.log = lines.append
        self.pool.detach(self.first, self.fake)
        self.assertEqual(self.fake.freed, [])
        self.assertTrue(all(shared.holds(states) for states in self.pool.tensors))
        self.pool.detach(self.second, self.fake)
        self.assertEqual(sorted(self.fake.freed), names)
        self.assertEqual(len(set(self.fake.freed)), 192)
        self.assertEqual(self.pool.tensors, [])
        self.assertFalse(shared.holds(object()))
        self.assertTrue(self.pool.closed)
        self.assertEqual(lines, ['%s tensors=192 bytes_per_chip=%d' % (shared.CLOSED_MARKER, 192 * 6 * MIB)])
        self.pool.detach(self.second, self.fake)
        self.assertEqual(len(self.fake.freed), 192, 'detaching twice frees nothing twice')

    def test_a_closed_pool_is_replaced_by_the_next_attach(self):
        self.pool.detach(self.first, self.fake)
        self.pool.detach(self.second, self.fake)
        again = shared.join(block(), environ=self.env)
        self.assertIsNot(again, self.pool)
        self.assertFalse(again.closed)


class ClaimTests(Clean):
    def setUp(self):
        super().setUp()
        self.env = {shared.FLAG: '1', **FOUR}
        self.a, self.b = block(), block()
        self.pool = shared.join(self.a, environ=self.env)
        shared.join(self.b, environ=self.env)

    def test_a_block_may_claim_the_history_again_and_the_first_claim_finds_no_other_writer(self):
        self.pool.claim(self.a)
        self.pool.claim(self.a)
        self.assertEqual(self.pool.counts['claims'], 2)
        self.assertIs(self.pool.writer, self.a)

    def test_a_block_with_deferred_commits_is_flushed_before_the_other_block_claims(self):
        flushed = []
        self.a.deferred_commits = [(0, 3), (1, 5)]
        self.a.flush_commits = lambda site: (flushed.append(site), self.a.deferred_commits.clear())[0]
        self.pool.claim(self.a)
        self.pool.claim(self.b)
        self.assertEqual(flushed, ['shared-history'])
        self.assertEqual((self.pool.counts['flushes'], self.pool.counts['flushed_commits']), (1, 2))
        self.assertIs(self.pool.writer, self.b)

    def test_a_block_with_undecided_segments_refuses_the_other_blocks_round(self):
        self.pool.claim(self.a)
        self.a.phase, self.a.pending_segments = 'verified', {0, 1, 2, 3}
        with self.assertRaisesRegex(ValueError, 'Commit-before-reuse refused.*verified and not committed'):
            self.pool.claim(self.b)
        self.assertIs(self.pool.writer, self.a, 'a refusal changes nothing')
        self.a.phase, self.a.pending_segments = 'idle', {2}
        with self.assertRaisesRegex(ValueError, r'uncommitted segments \[2\]'):
            self.pool.claim(self.b)

    def test_a_flush_that_leaves_commits_behind_still_refuses(self):
        self.pool.claim(self.a)
        self.a.deferred_commits = [(0, 1)]
        self.a.flush_commits = lambda site: 0
        with self.assertRaisesRegex(ValueError, '1 deferred commit traces not enqueued'):
            self.pool.claim(self.b)

    def test_a_failed_or_closed_writer_holds_nothing_the_next_block_waits_for(self):
        for phase in ('failed', 'closed'):
            self.a.phase = phase
            self.a.pending_segments = {0}
            self.a.deferred_commits = [(0, 1)]
            self.pool.writer = self.a
            self.pool.claim(self.b)
            self.assertIs(self.pool.writer, self.b)

    def test_only_an_attached_block_claims(self):
        with self.assertRaisesRegex(ValueError, 'Only an attached block'):
            self.pool.claim(block())


class LedgerTests(unittest.TestCase):
    """The real memory ledger over the shared set: a buffer counts once, so the second block's `packed_block` line drops by exactly the
    shared bytes and every delta check still passes."""

    @staticmethod
    def operations():
        import test_memory_ledger as ledger_tests

        return ledger_tests, ledger_tests.FakeOperations()

    def run_blocks(self, share):
        ledger_tests, operations = self.operations()
        ledger, lines, reports = ledger_tests.ledger_for(operations)
        shape = (16, 12, 128, 128)
        pooled = [operations.tensor(shape) for tensor in range(shared.TENSORS)] if share else None

        def one_block():
            histories = pooled if share else [operations.tensor(shape) for tensor in range(shared.TENSORS)]
            other = [operations.tensor((1, 1, 64, 5120)) for tensor in range(5)]    # taps: per block either way
            return SimpleNamespace(records=[SimpleNamespace(states=value) for value in histories], taps=other)

        first, second = one_block(), one_block()
        ledger.phase('P6', point='block1', packed_block=first)
        ledger.phase('P6', point='block2', packed_block=second)
        return ledger, reports

    def block_bytes(self, ledger):
        return [reading for phase, point, reading in ledger.readings if phase == 'P6']

    def category_total(self, ledger):
        return sum(size for (chip, address), (category, size, phase) in ledger.known.items()
                   if category == 'packed_block' and chip == 0)

    def test_sharing_removes_exactly_the_shared_bytes_from_the_ledger(self):
        private, private_reports = self.run_blocks(False)
        sharing, sharing_reports = self.run_blocks(True)
        saved = self.category_total(private) - self.category_total(sharing)
        self.assertEqual(saved, shared.freed_bytes(2))
        self.assertEqual(saved, 1207959552)
        self.assertEqual(self.category_total(sharing), 2 * 5 * 64 * 5120 * 2 + 1152 * MIB)
        self.assertEqual(self.category_total(private), 2 * 5 * 64 * 5120 * 2 + 2304 * MIB)

    def test_the_second_blocks_line_carries_only_its_own_buffers_and_the_delta_checks_hold(self):
        for share in (False, True):
            with self.subTest(shared=share):
                ledger, reports = self.run_blocks(share)
                checks = [check for check in ledger.checks if check['phase'] == 'P6']
                self.assertEqual(len(checks), 2)
                self.assertTrue(all(check['status'] != 'FAILED' for check in checks), checks)
                second = [(phase, point) for phase, point, reading in ledger.readings if phase == 'P6'][-1]
                self.assertEqual(second, ('P6', 'block2'))
        sharing, reports = self.run_blocks(True)
        walked = {}
        for (chip, address), (category, size, phase) in sharing.known.items():
            if chip == 0:
                walked.setdefault(phase, 0)
                walked[phase] += size
        self.assertIn('P6', walked)


class WalkedBlock:
    """A block as the ledger's walker sees one: an instance dict reaching its records, its taps and the pool."""

    users, rows_per_user, block_rows = 4, 16, 64
    phase, pending_segments, deferred_commits = 'idle', (), ()

    def __init__(self):
        self.records, self.taps, self.shared_history = [], [], None


class LedgerAttributionTests(unittest.TestCase):
    def test_the_pool_does_not_lead_the_walk_from_one_block_into_the_other(self):
        import test_memory_ledger as ledger_tests

        shared.reset()
        self.addCleanup(shared.reset)
        operations = ledger_tests.FakeOperations()
        ledger, lines, reports = ledger_tests.ledger_for(operations)
        env = {shared.FLAG: '1', **FOUR}
        first, second = WalkedBlock(), WalkedBlock()
        pool = shared.join(first, environ=env)
        shared.join(second, environ=env)
        pooled = [operations.tensor((16, 12, 128, 128)) for tensor in range(shared.TENSORS)]
        pool.tensors = pooled
        for block_ in (first, second):
            block_.shared_history = pool
            block_.records = [SimpleNamespace(states=value) for value in pooled]
            block_.taps = [operations.tensor((1, 1, 64, 5120)) for tap in range(5)]
        taps = 5 * 64 * 5120 * 2
        added_first = ledger.claim('packed_block', first, 'P6')[0]
        added_second = ledger.claim('packed_block', second, 'P6')[0]
        self.assertEqual(added_first[0], 1152 * MIB + taps, 'the first block carries the shared set and its own taps')
        self.assertEqual(added_second[0], taps, 'the second carries only its own: the pool did not lead the walk into the first block')
        self.assertFalse(hasattr(pool._held, '__dict__'))
        with self.assertRaises(TypeError):
            vars(pool._held)


ENVIRONMENT = json.loads((Path(__file__).parents[2] / 'docker' / 'qwen-c2-v235-environment.json').read_text(encoding='utf-8'))


def effective_environment(profile_name):
    """What the card runs: the image's baked ENV (qwen_configuration) with the profile's env laid over it, exactly as
    serving_c2_contract.apply_environment does."""
    environ = dict(ENVIRONMENT['qwen_configuration'])
    contract.apply_environment(dict(PROFILES[profile_name], name=profile_name), environ)
    return environ


class EffectiveEnvironmentTests(Clean):
    """Every earlier test built its blocks from {QWEN_FAST_TP, QWEN_FAST_M3_BLOCKS}; the cards run the image's environment plus the
    profile's, which bakes K5-A and the deferred GDN commits."""

    def test_the_image_serves_k5a_and_defers_the_gdn_commits(self):
        image = ENVIRONMENT['qwen_configuration']
        for flag in ('QWEN_FAST_GDN_SEQ_BLOCK', 'QWEN_FAST_GDN_USER_BATCH', 'QWEN_FAST_EARLY_DRAFT', 'QWEN_FAST_GDN_AFTER_PAIRS',
                     'QWEN_FAST_ROUND_FENCES', 'QWEN_FAST_PIPELINED_COMMITS', 'QWEN_FAST_MEMORY_LEDGER'):
            self.assertEqual(image.get(flag), '1', flag)

    def test_no_four_e_profile_is_refused_in_the_environment_it_runs_in(self):
        for name in (ARM, PLAIN, GROW):
            with self.subTest(profile=name):
                environ = effective_environment(name)
                self.assertEqual(environ[shared.FLAG], '1')
                self.assertEqual(environ['QWEN_FAST_GDN_SEQ_BLOCK'], '1', 'the lever rides on the served K5-A launch')
                self.assertIsNone(shared.refusal(environ))

    def test_the_schedule_in_that_environment_defers_the_commits_and_flushes_at_the_shared_history_site(self):
        import early_draft

        for name in (ARM, PLAIN, GROW, CONTROL):
            with self.subTest(profile=name):
                environ = effective_environment(name)
                self.assertTrue(early_draft.gdn_after_pairs_enabled(environ))
        self.assertIn(shared.FLUSH_SITE, early_draft.IN_STEP_SITES)

    def test_the_ledger_is_already_on_in_the_image_so_the_control_and_the_audited_ship_differ_by_nothing_at_run_time(self):
        control, ship = effective_environment(CONTROL), effective_environment(SHIP_AUDIT)
        self.assertEqual(control, ship)
        arm = effective_environment(ARM)
        self.assertEqual({key for key in {*arm, *control} if arm.get(key) != control.get(key)}, {shared.FLAG})

    def test_split_v_is_still_refused_because_its_twin_is_not_plumbed(self):
        environ = effective_environment(ARM)
        for value in ('', '0', '1'):
            self.assertIsNone(shared.refusal(dict(environ, QWEN_FAST_GDN_SPLIT_V=value)))
        self.assertIn('QWEN_FAST_GDN_SPLIT_V', shared.refusal(dict(environ, QWEN_FAST_GDN_SPLIT_V='2')))


class TwinBindingTests(Clean):
    def test_the_k5a_execute_twin_is_bound_only_when_the_flag_is_on(self):
        production = tp_addresses.bound_twins(effective_environment(SHIP))
        self.assertNotIn(('gdn_seq_block', 'execute', 'gdn_shared_history', 'seq_block_execute'), production[0])
        self.assertEqual(production, tp_addresses.bound_twins(effective_environment(CONTROL)),
                         'production binds exactly what it did before the lever existed (the control is its twin)')
        for name in (ARM, PLAIN, GROW):
            rows = tp_addresses.bound_twins(effective_environment(name))[0]
            self.assertEqual([row for row in rows if row[:2] == ('gdn_seq_block', 'execute')],
                             [('gdn_seq_block', 'execute', 'gdn_shared_history', 'seq_block_execute')])
            self.assertEqual([row for row in rows if row[:2] != ('gdn_seq_block', 'execute')],
                             [row for row in production[0] if row[:2] != ('gdn_seq_block', 'execute')])

    def test_installing_rebinds_the_pinned_execute_to_the_twin_and_uninstalling_puts_it_back(self):
        import gdn_seq_block

        pinned = gdn_seq_block.execute
        environ = effective_environment(ARM)
        with patch.dict(os.environ, environ):
            tp_addresses.install()
            self.addCleanup(tp_addresses.uninstall)
            self.assertIs(gdn_seq_block.execute, shared.seq_block_execute)
            self.assertIs(shared._PINNED, pinned)
            tp_addresses.uninstall()
        self.assertIs(gdn_seq_block.execute, pinned)

    def test_outside_a_capture_the_twin_is_the_pinned_function(self):
        calls = []
        with patch.object(shared, '_PINNED', lambda *args, **kwargs: calls.append((args, kwargs)) or 'pinned'):
            self.assertIsNone(shared.active())
            self.assertEqual(shared.seq_block_execute('mesh', ['users'], 'ops', output_memory='dram', kernels='build'), 'pinned')
        self.assertEqual(calls, [(('mesh', ['users'], 'ops'), dict(output_memory='dram', kernels='build'))])


class ServedPathFake(quad_tests.FourChipTTNN):
    """The four-chip launch fake plus the few ttnn calls gdn_user_batch_conv makes around the recurrence."""

    def __init__(self):
        super().__init__()
        self.made = 0
        self.freed_objects = []
        self.transformer = SimpleNamespace(gdn_decode_conv_gates=self.conv_gates)

    def make(self, name, shape):
        self.made += 1
        return quad_tests.FakeTensor(name, shape, 10 ** 6 + 16 * self.made)

    def to_memory_config(self, value, memory):
        return value

    def clone(self, value, memory_config=None):
        return self.make('clone:' + value.name, value.shape)

    def slice(self, value, start, stop, memory_config=None):
        return self.make('z:' + value.name, (1, stop[1] - start[1], stop[2] - start[2]))

    def conv_gates(self, projected, windows, taps, a, b, dt_bias, neg_exp_A, batch=None, memory_config=None, channels=None,
                   a_col=None, b_col=None):
        found = tp_shapes.active()
        return [self.make('conv', (1, batch, channels)), self.make('beta', (1, batch, found.gdn_nv)),
                self.make('gate', (1, batch, found.gdn_nv))]

    def deallocate(self, value):
        super().deallocate(value)
        self.freed_objects.append(value)


class ServedPathTests(Clean):
    """The served launch itself: gdn_user_batch_conv.run_user_batched_projected under the K5-A flag (the image's), the K5-A audit on one
    layer, and the shared history, with the twins installed as the card process installs them. The pool must hand out exactly 192
    tensors a block capture, nothing to the warm forward or to the audit's served launch, and free none of its own."""

    LAYERS = 48
    AUDIT_LAYER = 23

    def setUp(self):
        super().setUp()
        from test_gdn_seq_block import build as seq_build

        environ = {'QWEN_FAST_TP': '4', 'QWEN_FAST_M3_BLOCKS': '2', 'QWEN_FAST_GDN_USER_BATCH': '1',
                   'QWEN_FAST_GDN_SEQ_BLOCK': '1', 'QWEN_FAST_GDN_SEQ_BLOCK_AUDIT': str(self.AUDIT_LAYER),
                   shared.FLAG: '1'}
        patcher = patch.dict(os.environ, environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        tp_addresses.install()
        self.addCleanup(tp_addresses.uninstall)
        for name, value in (('gdn_multitoken.validate_handoff_runtime', None),
                            ('gdn_seq_block.served_kernels', seq_build())):
            patched = patch(name, return_value=value)
            patched.start()
            self.addCleanup(patched.stop)
        self.fake = ServedPathFake()
        self.first, self.second = block(), block()
        self.pool = shared.join(self.first, environ=os.environ)
        shared.join(self.second, environ=os.environ)

    def layer(self, index):
        import gdn_seq_block
        import gdn_user_batch_conv

        fake, found = self.fake, tp_shapes.active()
        groups = [(fake.make('projected', (1, 16, found.gdn_qkvzab)), fake.make('initial', (1, found.gdn_nv, 128, 128)),
                   [fake.make('conv', (1, 1, found.gdn_qkv)) for tap in range(4)]) for user in range(4)]
        taps = [fake.make('tap', (1, 1, found.gdn_qkv)) for tap in range(4)]
        extras = (fake.make('dt', (1, 1, found.gdn_nv)), fake.make('nega', (1, 1, found.gdn_nv)), fake.make('norm_w', (1, 1, 128)))

        def windows(mesh, projected, history):
            return [fake.make('window', (1, projected.shape[1], found.gdn_qkv)) for slot in range(4)]

        with patch('gdn_conv_windows.build_windows', side_effect=windows), gdn_seq_block.audit_scope(index):
            results = gdn_user_batch_conv.run_user_batched_projected(
                quad_tests.four_mesh(), groups, taps, *extras, KERNELS, fake)
        owned = [value for result in results for value in result['owned']]
        held = gdn_seq_block.audit_held_of(dict(results[0], segment_results=tuple(results)))
        tp_addresses.release_owned(fake, owned + held)
        return results

    def capture(self, block_):
        results = []
        with self.pool.capture(block_, self.fake):
            for index in range(self.LAYERS):
                results.append(self.layer(index))
        return results

    def private_histories(self):
        return [value for value in self.fake.allocated if tuple(value.shape) == (16, 12, 128, 128)]

    def test_each_block_capture_takes_exactly_192_tensors_and_the_second_is_handed_the_first_ones(self):
        warm = self.layer(0)
        self.assertEqual((self.pool.cursor, len(self.pool.tensors)), (0, 0), 'the warm forward takes nothing from the pool')
        self.assertTrue(all(result.get('seq_block') for result in warm), 'the launch is the served K5-A one')
        before = len(self.private_histories())
        first = self.capture(self.first)
        self.assertEqual((self.pool.cursor, len(self.pool.tensors), self.pool.expected), (192, 192, 192))
        self.assertEqual(len(self.private_histories()) - before, 192 + 4,
                         'the pool built 192; the audited layer\'s served launch allocated its own 4 privately')
        states = [result['states'] for layer in first for result in layer]
        self.assertTrue(all(shared.holds(value) for value in states))
        self.assertEqual(len({id(value) for value in states}), 192)
        self.assertTrue(all(result.get('seq_block') for layer in first for result in layer))
        before = len(self.private_histories())
        second = self.capture(self.second)
        self.assertEqual(len(self.private_histories()) - before, 4, 'only the audit\'s served launch allocates a history')
        self.assertEqual(self.pool.counts['reused'], 192)
        self.assertTrue(all(a is b for a, b in zip(states, [result['states'] for layer in second for result in layer], strict=True)))

    def test_nothing_the_pool_owns_is_ever_freed_by_the_layers_own_cleanup_and_everything_else_is(self):
        results = self.capture(self.first)
        pooled = {id(value) for value in self.pool.tensors}
        self.assertFalse(pooled & {id(value) for value in self.fake.freed_objects})
        freed = {id(value) for value in self.fake.freed_objects}
        for layer in results:
            for result in layer:
                self.assertIn(id(result['output']), freed, 'the private K5-A output is released with the layer')

    def test_the_audits_served_launch_inside_a_capture_allocates_privately(self):
        with self.pool.capture(self.first, self.fake):
            for index in range(self.LAYERS):
                self.layer(index)
            self.assertEqual(self.pool.cursor, 192)
            self.assertIsNone(shared.active_batched(), 'under K5-A the batched launch never takes from the pool')
            self.assertIs(shared.active(), self.pool)

    def test_a_k5a_launch_that_fails_leaves_the_pooled_tensors_alone(self):
        self.capture(self.first)
        freed = len(self.fake.freed_objects)
        self.fake.generic_op = lambda tensors, program: (_ for _ in ()).throw(RuntimeError('device'))
        with self.assertRaisesRegex(RuntimeError, 'device'):
            with self.pool.capture(self.second, self.fake):
                self.layer(0)
        pooled = {id(value) for value in self.pool.tensors}
        self.assertFalse(pooled & {id(value) for value in self.fake.freed_objects[freed:]})
        self.assertGreater(len(self.fake.freed_objects), freed, 'the launch\'s own private tensors were released')

    def test_an_aliasing_wrapper_of_a_pooled_buffer_is_still_the_pools(self):
        first = self.capture(self.first)
        states = first[0][0]['states']
        alias = quad_tests.FakeTensor('alias', states.shape, states.address)
        self.assertFalse(shared.holds(alias))
        self.fake.freed_objects.clear()
        tp_addresses.release_owned(self.fake, [alias])
        self.assertEqual(self.fake.freed_objects, [], 'released by address: the pool still owns that buffer')
        stranger = quad_tests.FakeTensor('stranger', states.shape, 777777)
        tp_addresses.release_owned(self.fake, [stranger])
        self.assertEqual(self.fake.freed_objects, [stranger])


class RefusalLoggingTests(Clean):
    def test_a_runtime_refusal_in_claim_is_logged_with_the_refused_marker_before_it_raises(self):
        lines = []
        env = {shared.FLAG: '1', **FOUR}
        a, b = block(), block()
        pool = shared.join(a, log=lines.append, environ=env)
        shared.join(b, environ=env)
        pool.claim(a)
        a.phase = 'verified'
        with self.assertRaisesRegex(ValueError, 'Commit-before-reuse refused'):
            pool.claim(b)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(shared.REFUSED_MARKER + ' site=claim reason=Commit-before-reuse refused'))

    def test_the_flush_is_logged_with_its_site_and_count(self):
        lines = []
        env = {shared.FLAG: '1', **FOUR}
        a, b = block(), block()
        pool = shared.join(a, log=lines.append, environ=env)
        shared.join(b, environ=env)
        a.deferred_commits = [(0, 1), (1, 2), (2, 3), (3, 4)]
        a.flush_commits = lambda site: a.deferred_commits.clear()
        pool.claim(a)
        pool.claim(b)
        self.assertEqual(lines, ['%s site=shared-history commits=4' % shared.FLUSH_MARKER])

    def test_the_smoke_check_demands_both_roles_and_no_refusal(self):
        import c2_smoke_check as smoke

        engaged = [('%s role=%s blocks=2 users=4 layers=48 tensors=192 tensor_bytes=6291456 freed_per_chip=1207959552 '
                    'kv_blocks_gained=2168') % (shared.ENGAGED_MARKER, role) for role in ('owner', 'sharer')]
        env = {shared.FLAG: '1'}
        self.assertEqual(smoke.shared_history_problems(env, '\n'.join(engaged)), [])
        self.assertEqual(smoke.shared_history_problems({}, 'nothing'), [])
        self.assertTrue(smoke.shared_history_problems({}, engaged[0]), 'lines on a profile without the flag')
        self.assertTrue(smoke.shared_history_problems(env, engaged[0]), 'one role missing')
        self.assertTrue(smoke.shared_history_problems(env, '\n'.join(engaged + engaged[:1])), 'an owner twice')
        self.assertTrue(smoke.shared_history_problems(env, '\n'.join(engaged).replace('tensors=192', 'tensors=191')))
        refused = '%s site=claim reason=Commit-before-reuse refused' % shared.REFUSED_MARKER
        self.assertTrue(any('refused' in problem for problem in smoke.shared_history_problems(env, '\n'.join(engaged + [refused]))))


class ConstructionFailureTests(unittest.TestCase):
    def test_the_join_is_inside_the_constructors_try_so_a_refusal_closes_the_block(self):
        text = (Path(__file__).with_name('packed_verifier.py')).read_text(encoding='utf-8')
        marker = 'self.shared_history = gdn_shared_history.join(self, log=diagnostic)'
        self.assertEqual(text.count(marker), 1)
        before = text[:text.index(marker)]
        self.assertGreater(before.rindex('        try:\n'), before.rindex("        started = time.perf_counter()\n"),
                           'the join comes after the try that closes the block on a failure')


class OffByDefaultTests(unittest.TestCase):
    def test_nothing_in_the_serving_path_imports_the_module_unless_a_flag_is_set(self):
        import re

        base = Path(__file__).parent
        for name in ('packed_verifier.py', 'gdn_user_batch_tp.py', 'tp_addresses.py', 'serving_c2_contract.py'):
            text = (base / name).read_text(encoding='utf-8')
            imports = [line for line in text.splitlines() if re.match(r'\s*(import|from) gdn_shared_history', line)]
            for line in imports:
                self.assertTrue(line.startswith(' '), '%s imports gdn_shared_history at module level: %r' % (name, line))
        verifier = (base / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn("os.environ.get('QWEN_FAST_GDN_SHARED_HISTORY', '0') != '0'", verifier)

    def test_the_overlay_lists_the_module_and_the_module_is_stdlib_only(self):
        overlay = (Path(__file__).parents[2] / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()
        self.assertIn('scripts/ci/gdn_shared_history.py', overlay)
        import ast

        tree = ast.parse((Path(__file__).with_name('gdn_shared_history.py')).read_text(encoding='utf-8'))
        modules = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        modules |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        # gdn_seq_block (the pinned execute the K5-A twin wraps) and ttnn (the default `operations`) are imported inside functions
        # only, which the K5-A twin needs; nothing else leaves the standard library.
        self.assertEqual(modules, {'contextlib', 'os', 'gdn_seq_block', 'ttnn'})
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {alias.name for alias in node.names} | ({node.module} if isinstance(node, ast.ImportFrom) else set())
                self.assertFalse(names & {'ttnn', 'gdn_seq_block'}, 'imported at module level')


if __name__ == '__main__':
    unittest.main()
