"""D0, the one-user lane (QWEN_FAST_SOLO_LANE, gate only): a 16-row packed block beside M3 over pool slot 0.

Four layers, none needing a device:
  - the admission (serving_solo_lane.solo_lane_admission): off by default and strict, refused at every shape, flag, width
    or run it does not admit, every reason at once, an UNQUALIFIED line per missing piece;
  - the routing (serving_packed_step.PackedStep given `solo=`), on the fakes test_serving_packed_step drives the step with:
    a lone slot-0 request drafts and steps on the solo block, every other round is exactly what it was;
  - the REAL PackedVerifierEngine at one user beside the real M3 block on test_packed_extent_block's fake device: the carry
    is shared by identity, the segment stages its own family, the block commits and returns to idle;
  - the attach (serving_runtime.attach_combined_runtime on test_tp4_attach_profile's fakes) and the gate-only profiles.
The flag off changes nothing: every existing test of the step, the block and the attach is unchanged."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch  # noqa: F401 - imported before any test patches sys.modules

import packed_any_admission
from packed_shapes import PackedShape, m3_shape, solo_shape, validate_shape
import serving_solo_lane as lane
import serving_packed_step
from serving_packed_step import PackedStep, packed_device_step
from test_c2_packed_tp4_profiles import image_env, profiles
import test_serving_packed_step as step_tests
from test_serving_packed_step import FakeBlock, FakeRequest, entry
import verifier_engine

M3 = (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1')
NOT_M3 = (False, 'users=2 FOUR_AS_TWO=unset PACKED_STEP=1')


def gate_environment(**changes):
    """A gate run of c2-packed-tp4-solo-gate: the image's ENV under the profile's env, plus the switch the workflow sets."""
    import serving_c2_contract as contract

    environ = dict(image_env())
    contract.apply_environment(dict(profiles()['c2-packed-tp4-solo-gate'], name='c2-packed-tp4-solo-gate'), environ)
    environ['QWEN_C2_GATE'] = '1'
    environ.update(changes)
    return {name: value for name, value in environ.items() if value is not None}


class Lines:
    def __init__(self):
        self.lines = []

    def __call__(self, template, *values):
        self.lines.append(template.format(*values))


class AdmissionTests(unittest.TestCase):
    def test_off_by_default_and_nothing_is_read_while_off(self):
        for environ in ({}, {'QWEN_FAST_SOLO_LANE': '0'}):
            self.assertIsNone(lane.solo_lane_admission(NOT_M3, environ))
            self.assertFalse(lane.solo_lane_requested(environ))

    def test_only_zero_or_one_is_a_value(self):
        for value in ('2', '', 'true', 'yes', '01', ' 1'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    lane.solo_lane_requested({'QWEN_FAST_SOLO_LANE': value})
                with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    lane.solo_lane_admission(M3, {'QWEN_FAST_SOLO_LANE': value})

    def test_a_gate_run_of_the_solo_profile_is_admitted_and_logs_every_missing_qualification(self):
        log = Lines()
        record = lane.solo_lane_admission(M3, gate_environment(), log=log)
        self.assertEqual((record['slot'], record['rows'], record['users']), (0, 16, 1))
        unqualified = [line for line in log.lines if line.startswith(lane.UNQUALIFIED_MARKER)]
        self.assertEqual(len(unqualified), len(record['unqualified']))
        self.assertTrue(all('/%d)' % len(unqualified) in line for line in unqualified))
        admitted = [line for line in log.lines if line.startswith(lane.ADMITTED_MARKER)]
        self.assertEqual(admitted, ['[SOLO-LANE] admitted slot=0 rows=16 users=1 (gate only)'])
        self.assertTrue(all(len(line) < 180 for line in log.lines))

    def test_every_refusal_is_named_at_once_and_logged_line_by_line(self):
        cases = {
            'the shape': (NOT_M3, {}, 'beside the 64-row M3 block'),
            'extent replay': (M3, dict(QWEN_FAST_EXTENT_REPLAY='0'), 'QWEN_FAST_EXTENT_REPLAY=0, not 1'),
            'padded block': (M3, dict(QWEN_FAST_PADDED_BLOCK=None), r'QWEN_FAST_PADDED_BLOCK=\(unset\), not 1'),
            'single gate/up': (M3, dict(QWEN_FAST_SINGLE_GATEUP='0'), 'one MLP arithmetic'),
            'four as two': (M3, dict(QWEN_FAST_FOUR_AS_TWO='1'), 'one 64-row M3 block'),
            'the pair': (M3, dict(QWEN_FAST_TP='2'), 'four-card lane'),
            'not a gate run': (M3, dict(QWEN_C2_GATE=None), 'gate run of a gate-only'),
            'not a gate profile': (M3, dict(QWEN_C2_GATE_PROFILE=None), 'gate run of a gate-only'),
        }
        for name, (m3, changes, text) in cases.items():
            with self.subTest(refused=name):
                log = Lines()
                with self.assertRaisesRegex(ValueError, text):
                    lane.solo_lane_admission(m3, gate_environment(**changes), log=log)
                self.assertTrue(log.lines and all(line.startswith(lane.MARKER + ' refused') for line in log.lines))

    def test_two_m3_blocks_are_refused_by_name_whatever_else_holds(self):
        # QWEN_FAST_M3_BLOCKS=2: eight seats on two M3 blocks. The solo block is bound to pool slot 0, segment 0 of block
        # A, and nothing has taken block B into account: refused naming the flag, with every other reason still listed.
        eight = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')
        log = Lines()
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_M3_BLOCKS=2: the solo lane is built beside the ONE 64-row M3 block'):
            lane.solo_lane_admission(eight, gate_environment(QWEN_FAST_M3_BLOCKS='2'), log=log)
        self.assertTrue(log.lines and all(line.startswith(lane.MARKER + ' refused') for line in log.lines))
        # the value is strict: anything but '1' is refused here, never read as one block
        for value in ('2', '3', '', 'x'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_M3_BLOCKS='):
                lane.solo_lane_admission(M3, gate_environment(QWEN_FAST_M3_BLOCKS=value), log=Lines())
        # one block, stated or not, is today's admission
        for value in (None, '1'):
            lane.solo_lane_admission(M3, gate_environment(QWEN_FAST_M3_BLOCKS=value), log=Lines())
        self.assertEqual(lane.M3_BLOCKS_FLAG, 'QWEN_FAST_M3_BLOCKS')

    def test_the_fused_commit_is_refused(self):
        for flag in lane.UNSUPPORTED_FLAGS:
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, flag + '=1: the solo block cannot take it'):
                    lane.solo_lane_admission(M3, gate_environment(**{flag: '1'}), log=Lines())
        # an explicit 0 is the default
        for flag in lane.UNSUPPORTED_FLAGS:
            lane.solo_lane_admission(M3, gate_environment(**{flag: '0'}), log=Lines())

    def test_a_traffic_profile_is_refused_whatever_else_holds(self):
        import serving_c2_contract as contract

        environ = dict(image_env())
        contract.apply_environment(dict(profiles()['c2-packed-tp4'], name='c2-packed-tp4'), environ)
        environ['QWEN_FAST_SOLO_LANE'] = '1'
        with self.assertRaisesRegex(ValueError, 'gate run of a gate-only'):
            lane.solo_lane_admission(M3, environ, log=Lines())
        environ['QWEN_C2_GATE'] = '1'      # the workflow's switch alone is not a gate-only profile (packed_any_admission)
        with self.assertRaisesRegex(ValueError, 'gate run of a gate-only'):
            lane.solo_lane_admission(M3, environ, log=Lines())

    def test_the_gate_rule_is_the_extent_paths_own_at_four_cards(self):
        import itertools

        self.assertEqual((lane.GATE_ENV, lane.GATE_PROFILE_ENV),
                         (packed_any_admission.GATE_ENV, packed_any_admission.GATE_PROFILE_ENV))
        for gate, marker in itertools.product((None, '0', '1'), repeat=2):
            environ = {'QWEN_FAST_TP': '4'}
            environ.update({key: value for key, value in (('QWEN_C2_GATE', gate), ('QWEN_C2_GATE_PROFILE', marker))
                            if value is not None})
            self.assertEqual(lane.gate_run(environ), packed_any_admission.unqualified_allowed(environ), (gate, marker))


class SoloShapeTests(unittest.TestCase):
    def test_one_t16_user_in_sixteen_rows_over_the_serving_width(self):
        shape = solo_shape(2052)
        self.assertEqual(shape, PackedShape(1, 16, 16, 2052, 2052 * 64))
        self.assertEqual(validate_shape(shape), shape)
        self.assertEqual(shape.users * shape.rows_per_user, shape.block_rows)
        with self.assertRaises(ValueError):
            solo_shape(67)

    def test_it_is_not_a_serving_count(self):
        import packed_shapes

        self.assertNotIn(1, packed_shapes.SERVING_SHAPES)
        self.assertIsNone(packed_shapes.serving_shape(3, 2052))
        self.assertEqual(packed_shapes.serving_shape(4, 2052), m3_shape(2052))


class PaddedFakeBlock(FakeBlock):
    """FakeBlock with the M3 block's padded rounds (packed_verifier.PackedVerifierEngine.pads): two or three live of four,
    the idle slot rule holding."""

    def __init__(self):
        super().__init__(users=4)
        self.extent = True

    def pads(self, count):
        return 2 <= count < 4

    def padded_refusal(self, users):
        return None

    def admits(self, position):
        return 128 <= position and position + 16 <= 131328


class SoloFakeBlock(FakeBlock):
    def __init__(self):
        super().__init__(users=1)
        self.extent = True

    def admits(self, position):
        return 128 <= position and position + 16 <= 131328

    def accept_limit(self, position):
        return None


class RoutingTests(unittest.TestCase):
    """PackedStep(m3, solo=solo): the one-user block serves a lone slot-0 round and nothing else."""

    def setUp(self):
        verifier_engine.note_prefill()
        serving_packed_step._SOLO_NOTED.clear()
        self.m3, self.solo = PaddedFakeBlock(), SoloFakeBlock()
        self.step = PackedStep(self.m3, solo=self.solo)
        self.stepped = []

    def request(self, name, slot, position=4100, solo_bound=None, rows=16, propose=True):
        request = FakeRequest(name, position, self.stepped)
        self.m3.bind(request.engine, slot)
        if slot == 0 if solo_bound is None else solo_bound:
            self.solo.bind(request.engine, 0)     # the carry is shared by identity: slot 0 is both blocks' segment 0
        if propose:
            request.propose(self.m3.predictions_for(slot), accept=6, rows=rows)
        return request

    def run_round(self, *requests):
        return self.step([entry(request) for request in requests], cancelled=lambda: False)

    def test_a_lone_slot_zero_request_drafts_sixteen_rows_and_its_round_runs_on_the_solo_block(self):
        lone = self.request('A', 0, propose=False)
        self.assertEqual(self.step.proposal_rows([lone]), 16)
        lone.propose(self.m3.predictions_for(0), accept=6)
        outputs = self.run_round(lone)
        self.assertEqual(self.solo.calls, [('verify', ['A']), ('commit', 0, 7)])
        self.assertEqual(self.m3.calls, [])
        self.assertEqual(self.stepped, [], 'not the per-request engines')
        self.assertEqual([output.request_id for output in outputs], ['A'])
        self.assertEqual(lone.engine.adopted[0][1:], (self.solo, 0), 'adopted on the solo block, segment 0')
        self.assertEqual((self.solo.phase, self.solo.pending_segments), ('idle', set()))

    def test_without_the_solo_block_the_same_lone_request_is_todays_sequential_round(self):
        plain = PackedStep(self.m3)
        lone = self.request('A', 0, propose=False)
        self.assertIsNone(plain.proposal_rows([lone]))
        self.assertIsNone(plain.solo)
        lone.propose(self.m3.predictions_for(0), accept=6, rows=4)
        from test_serving_packed_step import CommittingRequest

        committing = CommittingRequest('B', 4100, self.stepped)
        self.m3.bind(committing.engine, 0)
        committing.propose([5, 6, 7, 8], 3, rows=4)
        plain([entry(committing)], cancelled=lambda: False)
        self.assertEqual(self.stepped[-1][0], 'B')
        self.assertEqual(self.solo.calls + self.m3.calls, [])

    def test_a_lone_request_in_any_other_slot_stays_on_the_per_request_engines(self):
        survivor = self.request('C', 2, propose=False)
        self.assertIsNone(self.step.proposal_rows([survivor]))
        from test_serving_packed_step import CommittingRequest

        committing = CommittingRequest('C', 4100, self.stepped)
        self.m3.bind(committing.engine, 2)
        committing.propose([5, 6, 7, 8], 3, rows=4)
        self.step([entry(committing)], cancelled=lambda: False)
        self.assertEqual(self.stepped[-1][0], 'C')
        self.assertEqual(self.solo.calls, [])

    def test_two_three_and_four_live_users_are_m3s_rounds_and_the_solo_block_is_never_asked(self):
        requests = [self.request(name, slot, 4100 + slot * 50, propose=False) for slot, name in enumerate('ABCD')]
        for count in (2, 3, 4):
            live = requests[:count]
            self.assertEqual(self.step.proposal_rows(live), 16, count)
        for request in requests:
            request.propose(self.m3.predictions_for(request.session.request_id and 'ABCD'.index(request.session.request_id)),
                            accept=15)
        self.run_round(*requests)
        self.assertEqual([call[0] for call in self.m3.calls][:1], ['verify'])
        self.assertEqual(self.solo.calls, [])

    def test_a_lone_request_with_under_sixteen_tokens_left_or_outside_the_extent_is_not_drafted_for_the_block(self):
        lone = self.request('A', 0, propose=False)
        lone.session.emitted = [1] * 241
        self.assertIsNone(self.step.proposal_rows([lone]))
        lone.session.emitted = [1] * 240
        self.assertEqual(self.step.proposal_rows([lone]), 16)
        for position in (127, 131313):
            lone.session.position = position
            self.assertIsNone(self.step.proposal_rows([lone]), position)
        lone.session.position = 128
        self.assertEqual(self.step.proposal_rows([lone]), 16)

    def test_a_finished_request_is_not_live_and_two_requests_are_not_a_solo_round(self):
        done, lone = self.request('A', 0, propose=False), self.request('B', 1, propose=False)
        done.session.finished = True
        self.assertEqual(serving_packed_step.PackedStep(self.m3, solo=self.solo).solo, self.solo)
        # B alone in slot 1: not the solo block's slot
        self.assertIsNone(self.step.proposal_rows([done, lone]))
        alone = self.request('C', 0, propose=False)
        both = [alone, self.request('D', 0, propose=False, solo_bound=True)]
        self.assertIsNone(self.step.proposal_rows(both), 'two live requests are never one solo round')

    def test_a_narrow_ticket_of_the_slot_zero_request_is_not_a_solo_round(self):
        lone = self.request('A', 0, propose=False)
        lone.propose(self.m3.predictions_for(0), accept=1, rows=4)
        self.assertFalse(lane.solo_serves(self.solo, [entry(lone)]))
        lone2 = self.request('B', 0, propose=True)
        self.assertTrue(lane.solo_serves(self.solo, [entry(lone2)]))
        self.assertFalse(lane.solo_serves(self.solo, [entry(lone2), entry(lone2)]))
        self.assertFalse(lane.solo_serves(self.solo, []))

    def test_the_skip_reason_is_logged_once_and_the_route_line_names_the_slot(self):
        lines = []
        with patch.object(serving_packed_step, 'padded_log', lambda text, once=False: lines.append((text, once))):
            other = self.request('C', 2, propose=False)
            self.step.proposal_rows([other])
            self.step.proposal_rows([other])
            lone = self.request('A', 0, propose=False)
            lone.propose(self.m3.predictions_for(0), accept=6)
            self.run_round(lone)
        skipped = [text for text, once in lines if text.startswith(lane.SKIP_MARKER)]
        self.assertEqual(len(skipped), 2, 'the helper is called every tick; the log itself dedupes by text (once=True)')
        self.assertTrue(all(once for text, once in lines if text.startswith(lane.SKIP_MARKER)))
        self.assertIn('borrowed a pool slot other than 0', skipped[0])
        routed = [text for text, _ in lines if text.startswith(lane.ROUTE_MARKER)]
        self.assertEqual(len(routed), 1)
        self.assertIn('slot=0 request=A', routed[0])
        self.assertIn('block=solo', routed[0])
        self.assertTrue(all(len(text) < 180 for text, _ in lines))

    def test_the_solo_block_is_a_separate_one_user_sixteen_row_block(self):
        with self.assertRaisesRegex(ValueError, 'separate one-user 16-row'):
            PackedStep(self.m3, solo=self.m3)
        with self.assertRaisesRegex(ValueError, 'separate one-user 16-row'):
            PackedStep(self.m3, solo=FakeBlock(users=2))
        with self.assertRaisesRegex(ValueError, 'separate one-user 16-row'):
            PackedStep([self.m3, self.solo], solo=self.solo)

    def test_the_fence_window_belongs_to_the_block_of_the_coming_round(self):
        self.solo.prestaged = object()
        self.m3.prestaged = object()
        lone = self.request('A', 0, propose=False)
        self.assertIs(self.step.while_waiting([lone]).block, self.solo)
        pair = [self.request('B', 0, propose=False, solo_bound=True), self.request('C', 1, propose=False)]
        self.assertIs(self.step.while_waiting(pair).block, self.m3)
        self.assertIs(PackedStep(self.m3).while_waiting([lone]).block, self.m3, 'without the solo block: as it was')

    def test_a_round_on_the_other_block_bumps_the_fixture_epoch_once_and_a_repeat_does_not(self):
        bumps = []
        with patch.object(serving_packed_step, 'note_fixture_writer', bumps.append):
            first = self.request('A', 0)
            self.run_round(first)                         # the first round on either block: nothing to switch from
            second = self.request('A2', 0)
            self.run_round(second)                        # solo again
            self.assertEqual(bumps, [])
            pair = [self.request('B', 0, propose=False, solo_bound=True), self.request('C', 1, propose=False)]
            for request in pair:
                request.propose(self.m3.predictions_for(0), accept=15)
            self.run_round(*pair)                         # solo -> M3
            self.assertEqual(bumps, ['lane-switch'])
            third = self.request('A3', 0)
            self.run_round(third)                         # M3 -> solo
            self.assertEqual(bumps, ['lane-switch'] * 2)

    def test_an_announced_switch_bumps_once_at_the_announcement_and_the_step_does_not_bump_again(self):
        bumps = []
        with patch.object(serving_packed_step, 'note_fixture_writer', bumps.append):
            self.run_round(self.request('A', 0))          # solo
            self.step.announce_round(False)                # the plan for the next round: M3 - the bump, before any window
            self.assertEqual(bumps, ['lane-switch'])
            pair = [self.request('B', 0, propose=False, solo_bound=True), self.request('C', 1, propose=False)]
            for request in pair:
                request.propose(self.m3.predictions_for(0), accept=15)
            self.run_round(*pair)                         # the step: already noted
            self.assertEqual(bumps, ['lane-switch'])
            self.step.announce_round(False)
            self.assertEqual(bumps, ['lane-switch'], 'a repeat of the same block is no switch')

    def test_arming_and_flushing_deferred_commits_reach_the_solo_block_too(self):
        armed, flushed = [], []
        for name, block in (('m3', self.m3), ('solo', self.solo)):
            block.arm_deferred_commits = lambda name=name: armed.append(name) or True
            block.flush_commits = lambda site, name=name: flushed.append((name, site)) or 1
        self.assertTrue(self.step.arm_deferred_commits())
        self.assertEqual(self.step.flush_deferred_commits('end'), 2)
        self.assertEqual((armed, flushed), (['m3', 'solo'], [('m3', 'end'), ('solo', 'end')]))
        armed.clear(), flushed.clear()
        PackedStep(self.m3).arm_deferred_commits()
        self.assertEqual(armed, ['m3'])

    def test_the_sequential_step_is_the_solo_rounds_fallback_for_nothing(self):
        # a cancellation before the verify is the request's own step, exactly as on any block
        lone = self.request('A', 0)
        outputs = packed_device_step([entry(lone)], cancelled=lambda: True, block=self.solo)
        self.assertEqual(self.stepped, [('A', True)])
        self.assertEqual(len(outputs), 1)


# The real block: the extent fixture's fake device, and its pool grown to the two shapes the attach builds.
from test_packed_extent_block import ExtentFixture, WIDTH, owner  # noqa: E402
import test_packed_verifier as base  # noqa: E402
from packed_verifier import PackedVerifierEngine  # noqa: E402
from serving_buffer_pool import PackedExtentStorage  # noqa: E402


class RealBlockTests(ExtentFixture):
    """The REAL PackedVerifierEngine at one user beside the real M3 block over one four-slot pool, as the attach builds them."""

    def setUp(self):
        super().setUp()

        def integers(shape):
            return self.ttnn.allocate(shape, 'int32', 'row_major', torch.zeros(shape, dtype=torch.int32))

        self.storages = {users: PackedExtentStorage(users, 16, [[integers((2, WIDTH))] for user in range(users)],
                                                    [[integers((2,))] for user in range(users)]) for users in (4, 1)}
        self.pool.packed_extent = lambda users, rows: self.storages[users]

    def build_both(self):
        m3 = self.build(padded_min_users=2)
        solo = PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
                                    shared_weights=self.weights, shape=solo_shape(WIDTH), feature_taps=base.TAPS,
                                    pool_slots=(0,))
        self.addCleanup(lambda: solo.deadline.close() if solo.deadline is not None else None)
        return m3, solo

    def test_the_solo_block_builds_beside_m3_as_a_one_segment_extent_block_over_slot_zero(self):
        m3, solo = self.build_both()
        self.assertEqual((solo.shape.users, solo.rows_per_user, solo.block_rows), (1, 16, 16))
        self.assertTrue(solo.extent and m3.extent)
        self.assertEqual((solo.pool_slots, m3.pool_slots), ((0,), (0, 1, 2, 3)))
        self.assertIsNone(solo.padded_min_users, 'one segment: no idle segments to pad')
        self.assertEqual((solo.replay_capacity, solo.capture_position), (m3.replay_capacity, m3.capture_position))
        self.assertTrue(self.storages[1].taken and self.storages[4].taken)
        self.assertEqual(len(self.readers(solo)), 1)
        self.assertEqual([reader.sdpa_modes_applied for reader in self.readers(solo)],
                         [reader.sdpa_modes_applied for reader in self.readers(m3)][:1], 'one program: the same flag set')

    def test_slot_zeros_carry_is_one_segment_of_each_block_and_no_other_slots_is_the_solos(self):
        m3, solo = self.build_both()
        first, second = owner(self, 0, 5000), owner(self, 1, 5000)
        self.assertEqual((m3.segment_of(first.engine), solo.segment_of(first.engine)), (0, 0))
        self.assertEqual(m3.segment_of(second.engine), 1)
        with self.assertRaisesRegex(ValueError, 'borrows no carry'):
            solo.segment_of(second.engine)

    def test_a_solo_round_stages_its_own_family_verifies_commits_and_leaves_the_block_idle(self):
        m3, solo = self.build_both()
        for start in (1500, 60000, 131312):
            request = owner(self, 0, start)
            predictions, metrics = solo.verify([base.entry(request, range(10, 26))])
            self.assertEqual((metrics['segments'], len(predictions), solo.phase), ((0,), 1, 'verified'))
            reader = self.readers(solo)[0]
            self.assertEqual(reader.start, start)
            self.assertEqual(reader.positions.value.tolist(), [start & 255] + [0] * 7)
            solo.commit_user(0, 5)
            self.assertEqual((solo.phase, solo.pending_segments), ('idle', set()))
        self.assertEqual(solo.rounds, 3)
        self.assertEqual(m3.rounds, 0, 'M3 was never asked')

    def test_the_blocks_alternate_on_one_carry_without_either_refusing_the_others_round(self):
        m3, solo = self.build_both()
        request = owner(self, 0, 5000)
        for block, count in ((solo, 1), (m3, 2), (solo, 1), (m3, 3), (solo, 1)):
            entries = [base.entry(owner(self, segment, 5000 + 7 * segment, 7 + segment), range(10, 26))
                       for segment in range(count if block is m3 else 1)]
            predictions, metrics = block.verify(entries)
            self.assertEqual(len(predictions), len(entries))
            for segment in sorted(block.pending_segments):
                block.commit_user(segment, 3 if segment not in block.idle_segments else 0)
            self.assertEqual(block.phase, 'idle')
        self.assertEqual((solo.rounds, m3.rounds), (3, 2))

    def test_a_block_that_is_verified_refuses_the_solo_round_of_the_same_carry_until_it_commits(self):
        m3, solo = self.build_both()
        m3.verify([base.entry(owner(self, segment, 5000, 7 + segment), range(10, 26)) for segment in range(4)][::-1])
        self.assertEqual(m3.phase, 'verified')
        for segment in sorted(m3.pending_segments):
            m3.commit_user(segment, 1)
        solo.verify([base.entry(owner(self, 0, 5000), range(10, 26))])
        solo.commit_user(0, 1)
        self.assertEqual((m3.phase, solo.phase), ('idle', 'idle'))

    def test_the_solo_shape_is_validated_like_any_other(self):
        with self.assertRaises(ValueError):
            PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
                                 shared_weights=self.weights, shape=PackedShape(1, 16, 32, WIDTH, WIDTH * 64),
                                 feature_taps=base.TAPS, pool_slots=(0,))
        with self.assertRaisesRegex(ValueError, 'padded_min_users'):
            PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
                                 shared_weights=self.weights, shape=solo_shape(WIDTH), feature_taps=base.TAPS,
                                 pool_slots=(0,), padded_min_users=1)


# The attach, on test_tp4_attach_profile's fakes.
import test_tp4_attach_profile as attach_tests  # noqa: E402

SOLO_PROFILE = 'c2-packed-tp4-solo-gate'


ORIGINAL_ENVIRONMENT = attach_tests.environment


class AttachTests(unittest.TestCase):
    def attach(self, name, **changes):
        def laid(profile):
            environ = dict(ORIGINAL_ENVIRONMENT(profile), QWEN_C2_GATE='1')
            environ.update({key: value for key, value in changes.items() if value is not None})
            for key, value in changes.items():
                if value is None:
                    environ.pop(key, None)
            return environ

        return laid

    def test_the_solo_profile_builds_m3_then_the_solo_block_over_slot_zero_and_hands_both_to_the_step(self):
        laid = self.attach(SOLO_PROFILE)
        with patch.object(attach_tests, 'environment', laid):
            with attach_tests.Attach(SOLO_PROFILE).run() as seen:
                self.assertEqual([(shape.users, shape.rows_per_user) for shape in seen['engines']], [(4, 16), (1, 16)],
                                 'M3 first, then the solo block')
                self.assertEqual(seen['pool']['packed_shapes'], ((4, 16), (1, 16)))
                self.assertTrue(seen['pool']['extent_replay'])
                self.assertEqual(seen['attached']['lifecycle'] is not None, True)

    def test_the_solo_block_is_built_over_slot_zero_only_and_the_step_carries_it(self):
        laid = self.attach(SOLO_PROFILE)
        with patch.object(attach_tests, 'environment', laid):
            with attach_tests.Attach(SOLO_PROFILE).run() as seen:
                options = seen['block_options']
                self.assertEqual(len(options), 2)
                self.assertNotIn('pool_slots', options[0], 'M3 is built as it always was')
                self.assertEqual(options[1]['pool_slots'], (0,))
                self.assertNotIn('padded_min_users', options[1], 'one segment: nothing to pad')
                self.assertIn('padded_min_users', options[0], 'M3 keeps its padded rounds')
                blocks, step_options = seen['step']
                self.assertNotIsInstance(blocks, (list, tuple), 'one M3 block, as ever')
                self.assertEqual(sorted(step_options), ['solo'])
                self.assertEqual(seen['attached']['runtime'] is not None, True)

    def test_a_profile_without_the_flag_builds_exactly_one_block_and_a_step_without_solo(self):
        with attach_tests.Attach('c2-packed-tp4-gate').run() as seen:
            self.assertEqual(len(seen['engines']), 1)
            self.assertEqual(seen['pool']['packed_shapes'], ((4, 16),))
            self.assertEqual(seen['step'][1], {}, 'PackedStep(block): no keyword at all')

    def test_the_flag_on_a_traffic_profile_or_the_pair_is_refused_before_anything_is_built(self):
        for name, changes, text in (
                ('c2-packed-tp4-gate', dict(QWEN_FAST_SOLO_LANE='1', QWEN_C2_GATE=None), 'gate run of a gate-only'),
                ('c2-packed-tp4-solo-gate', dict(QWEN_FAST_SOLO_LANE='2'), 'must be 0 or 1'),
                ('c2-packed-tp4-solo-gate', dict(QWEN_FAST_FUSED_COMMIT='1'), 'the solo block cannot take it')):
            with self.subTest(profile=name, changes=changes):
                laid = self.attach(name, **changes)
                with patch.object(attach_tests, 'environment', laid):
                    attach = attach_tests.Attach(name)
                    with self.assertRaisesRegex(ValueError, text):
                        with attach.run():
                            pass
                self.assertEqual(attach.seen['engines'], [], 'nothing was built')
                self.assertIsNone(attach.seen['pool'], 'not even the pool')


class ProfileTests(unittest.TestCase):
    NEW = ('c2-packed-tp4-time-gate', 'c2-packed-tp4-solo-gate', 'c2-packed-tp4-solo-time-gate',
           'c2-packed-tp4-lanes-gate', 'c2-packed-tp4-lanes-time-gate')

    def test_each_is_the_gate_profiles_engine_and_limits_with_only_the_documented_environment(self):
        base_profile = profiles()['c2-packed-tp4-gate']
        for name in self.NEW:
            with self.subTest(profile=name):
                profile = profiles()[name]
                for key in ('engine', 'eos_ids', 'snapshots', 'mesh_device', 'mesh_graph_descriptor', 'min_answer_tokens',
                            'default_max_tokens', 'parser_rechunk', 'gate_only'):
                    self.assertEqual(profile[key], base_profile[key], key)
                self.assertIs(profile['gate_only'], True)
                differing = {key for key in set(profile['env']) | set(base_profile['env'])
                             if profile['env'].get(key) != base_profile['env'].get(key)}
                audits = {'QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'}
                solo = {'QWEN_FAST_SOLO_LANE'} if 'solo' in name or 'lanes' in name else set()
                lanes = {'QWEN_FAST_LANE'} if 'lanes' in name else set()
                timing = audits if name.endswith('time-gate') else set()
                self.assertEqual(differing, timing | solo | lanes)

    def test_the_flags_are_in_exactly_the_profiles_that_name_them_and_no_traffic_profile(self):
        for name, profile in profiles().items():
            env = profile.get('env', {})
            solo = env.get('QWEN_FAST_SOLO_LANE') == '1'
            lanes = env.get('QWEN_FAST_LANE') == '1'
            with self.subTest(profile=name):
                self.assertEqual(solo, name in ('c2-packed-tp4-solo-gate', 'c2-packed-tp4-solo-time-gate',
                                                'c2-packed-tp4-lanes-gate', 'c2-packed-tp4-lanes-time-gate'))
                self.assertEqual(lanes, name in ('c2-packed-tp4-lanes-gate', 'c2-packed-tp4-lanes-time-gate'))
                if solo or lanes:
                    self.assertIs(profile.get('gate_only'), True)
        self.assertNotIn('QWEN_FAST_SOLO_LANE', image_env())
        self.assertNotIn('QWEN_FAST_LANE', image_env())

    def test_the_audits_are_on_exactly_in_the_exactness_profiles(self):
        for name in self.NEW + ('c2-packed-tp4-gate',):
            env = profiles()[name]['env']
            want = '0' if name.endswith('time-gate') else '1'
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), (want, want), name)
            self.assertEqual(env['QWEN_C2_GATE_PROFILE'], '1')

    def test_every_new_profile_passes_the_solo_admission_where_it_asks_for_it(self):
        import serving_c2_contract as contract

        for name in self.NEW:
            environ = dict(image_env())
            contract.apply_environment(dict(profiles()[name], name=name), environ)
            environ['QWEN_C2_GATE'] = '1'
            with self.subTest(profile=name):
                if environ.get('QWEN_FAST_SOLO_LANE') == '1':
                    self.assertIsNotNone(lane.solo_lane_admission(M3, environ, log=Lines()))
                else:
                    self.assertIsNone(lane.solo_lane_admission(M3, environ, log=Lines()))


if __name__ == '__main__':
    unittest.main()
