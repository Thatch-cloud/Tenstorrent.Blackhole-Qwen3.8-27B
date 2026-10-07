"""tp4/next-5: the eight-seat quad draft (QWEN_FAST_QUAD_DRAFT_BLOCKS=2), two quads of four seats, (0, 1, 2, 3) and (4, 5, 6, 7).

What is held, on the CPU, with the coordinator's own tests' fakes (the quad trace is test_quad_draft's FakeQuadTrace; no card runs it):
  - flag off ('' / '0' / unset): the coordinator is the one tp4/next-5's parent commit shipped, round for round, at four and at eight seats;
  - eight live packable users form two quads, one trace each over its own devices, no pair runs, one select line naming both groups;
  - only slots 0-3 live: one quad; slots 0-3 and a pair: a quad and that pair; only 4-7: the second quad alone;
  - the refusal shapes, each by name: a value other than 2, QWEN_FAST_TP other than 4, QWEN_FAST_QUAD_DRAFT off, the quad audit, a pool
    that is not eight slots, a block that is not the 16-row T16 block, a quad_draft with no QUADS (the pair's pinned module); a
    refusal that concerns one block blocks that block only;
  - the capture headroom arithmetic: one quad is 450 MiB per chip, two are 900 MiB, the packed reserve is added once, and the coordinator
    checks one quad at a time (the second against what the first left);
  - the smoke check expects two engaged markers (one per block) and a round both quads served, in concurrent8_steady.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c2_smoke_check as check  # noqa: E402
import quad_draft  # noqa: E402
import quad_draft_tp  # noqa: E402
import test_quad_draft as base  # noqa: E402

FLAG = 'QWEN_FAST_QUAD_DRAFT'
BLOCKS = 'QWEN_FAST_QUAD_DRAFT_BLOCKS'
REQUIRED = dict(base.REQUIRED)
LOG = base.LOG
# The coordinator this change started from (tp4/next-5 before it): flag off, every round is that module's.
PARENT = '8d2da7ee'
MIB = 2 ** 20
FOUR_CARDS = {'QWEN_FAST_TP': '4'}


def environment(**flags):
    return base.clean_environment(**dict(REQUIRED, QWEN_FAST_PACKED_AUDIT='1', **flags))


class Harness(unittest.TestCase):
    """CoordinatorTests' setUp, with the four-card twin standing in for quad_draft (where QUADS lives) and its refusal answered by the
    test (the fake devices have no mesh)."""

    TP = '4'
    blocks = '2'

    def setUp(self):
        flags = {FLAG: '1', 'QWEN_FAST_TP': self.TP}
        if self.blocks is not None:
            flags[BLOCKS] = self.blocks
        env = environment(**flags)
        env.start()
        self.addCleanup(env.stop)
        del LOG[:]
        base.FakeQuadTrace.instances, base.FakeQuadTrace.failures = [], []
        self.Pair = base.pair_trace_class()
        self.Pair.instances = []
        from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture

        self.refusals = []
        twin = patch.dict(sys.modules, {'quad_draft': quad_draft_tp})
        twin.start()
        self.addCleanup(twin.stop)
        for target, value in (('quad_draft_tp.PreparedQuadDFlashProposal', base.FakeQuadTrace),
                              ('quad_draft_tp.refusal', Mock(side_effect=lambda devices, batched, environ=None: self.refuse(devices))),
                              ('dflash_proposal_trace.PreparedPackedDFlashProposal', self.Pair),
                              ('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture),
                              ('dflash_packed_proposal.select_packed_batched', Mock(side_effect=base.selected_tokens))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: LOG.append(('sync',))))
        self.mesh = object()

    def refuse(self, devices):
        """The twin's engage-time reason for these four devices: the first of self.refusals whose slots match, else None."""
        slots = tuple(device.pool_slot.index for device in devices)
        for wanted, reason in self.refusals:
            if wanted == slots:
                return reason
        return None

    def coordinator(self):
        import dflash_packed_proposal_coordinator

        return dflash_packed_proposal_coordinator.PackedProposalCoordinator()

    def prepare(self, coordinator, bridges, **options):
        lines, loguru = base.logged()
        with loguru:
            prepared = coordinator.prepare(bridges, **options)
        return prepared, lines

    def bridges(self, slots, **options):
        return base.quad_bridges(self.operations, self.mesh, slots, **options)

    def quiet(self, lines):
        return not [line for line in lines if 'fallback' in line or 'disabled' in line]


EIGHT = tuple(range(8))


class TwoQuadsTests(Harness):
    def test_eight_live_packable_users_form_two_quads_and_no_pair(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView

        coordinator, bridges = self.coordinator(), self.bridges(EIGHT)
        prepared, lines = self.prepare(coordinator, bridges)
        first, second = base.FakeQuadTrace.instances
        self.assertEqual(self.Pair.instances, [])
        self.assertEqual([[device.pool_slot.index for device in trace.devices] for trace in (first, second)],
                         [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(first.prepared, [(100, 101, 102, 103)])
        self.assertEqual(second.prepared, [(104, 105, 106, 107)])
        self.assertEqual([id(device) for device in prepared], [id(bridge.request.runtime.drafter) for bridge in bridges])
        for index, bridge in enumerate(bridges):
            capture = bridge.request.runtime.drafter.proposal_capture
            self.assertIsInstance(capture, _PackedCaptureView)
            self.assertIs(capture._trace, (first, second)[index // 4])
            self.assertEqual(capture._which, index % 4)
            self.assertTrue(bridge.request.runtime.drafter._packed_capture_released, 'R6: singles released')
            bridge.request.runtime.drafter.prepare_device.assert_not_called()
        self.assertEqual(LOG, [('quad', (100, 101, 102, 103)), ('quad', (104, 105, 106, 107)), ('sync',), ('collect', 'quad'),
                               ('collect', 'quad'), ('audit', 1), ('audit', 1)])
        self.assertTrue(any(line.startswith('[PACKED-SELECT] round=1 pairs=[[0, 1, 2, 3], [4, 5, 6, 7]] users=8 ') for line in lines), lines)
        self.assertTrue(self.quiet(lines), lines)
        self.assertEqual(coordinator.quad_rounds, 2)
        self.assertEqual(sorted(coordinator.quad_blocks), [(0, 1, 2, 3), (4, 5, 6, 7)])
        self.assertIsNone(coordinator.quad, 'the first block lives in quad_blocks under the flag')
        _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(len(base.FakeQuadTrace.instances), 2, 'the same devices replay the same two quads')
        self.assertEqual(sum('built=0' in line for line in lines), 2, lines)

    def test_only_slots_zero_to_three_live_form_one_quad(self):
        coordinator = self.coordinator()
        _, lines = self.prepare(coordinator, self.bridges((0, 1, 2, 3)))
        self.assertEqual(len(base.FakeQuadTrace.instances), 1)
        self.assertEqual([device.pool_slot.index for device in base.FakeQuadTrace.instances[0].devices], [0, 1, 2, 3])
        self.assertEqual(self.Pair.instances, [])
        self.assertTrue(any('pairs=[[0, 1, 2, 3]] users=4 ' in line for line in lines), lines)
        self.assertTrue(self.quiet(lines), lines)
        self.assertEqual(sorted(coordinator.quad_blocks), [(0, 1, 2, 3)])

    def test_only_slots_four_to_seven_live_form_the_second_quad_alone(self):
        coordinator = self.coordinator()
        _, lines = self.prepare(coordinator, self.bridges((4, 5, 6, 7)))
        self.assertEqual([[device.pool_slot.index for device in trace.devices] for trace in base.FakeQuadTrace.instances], [[4, 5, 6, 7]])
        self.assertTrue(any('pairs=[[4, 5, 6, 7]] users=4 ' in line for line in lines), lines)
        self.assertTrue(self.quiet(lines), lines)

    def test_a_quad_and_a_leftover_pair_run_together(self):
        coordinator = self.coordinator()
        _, lines = self.prepare(coordinator, self.bridges((0, 1, 2, 3, 4, 5)))
        self.assertEqual(len(base.FakeQuadTrace.instances), 1)
        self.assertEqual(len(self.Pair.instances), 1)
        self.assertEqual((self.Pair.instances[0].device_a.pool_slot.index, self.Pair.instances[0].device_b.pool_slot.index), (4, 5))
        self.assertTrue(any('pairs=[[0, 1, 2, 3], [4, 5]] users=6 ' in line for line in lines), lines)

    def test_a_block_with_a_ramping_user_leaves_its_pairs_to_the_pair_path(self):
        coordinator, bridges = self.coordinator(), self.bridges(EIGHT)
        bridges[7].request.runtime.drafter.history_rows = 1024
        self.prepare(coordinator, bridges)
        self.assertEqual([[device.pool_slot.index for device in trace.devices] for trace in base.FakeQuadTrace.instances], [[0, 1, 2, 3]])
        self.assertEqual(len(self.Pair.instances), 1, 'pair (4, 5) packs; (6, 7) drafts singly while 7 ramps')
        self.assertFalse(coordinator.quad_blocked)

    def test_a_departed_user_retires_only_its_blocks_quad(self):
        coordinator, bridges = self.coordinator(), self.bridges(EIGHT)
        self.prepare(coordinator, bridges)
        first, second = base.FakeQuadTrace.instances
        bridges[5].request.runtime.drafter.closed = True
        coordinator.release_closed()
        self.assertFalse(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(sorted(coordinator.quad_blocks), [(0, 1, 2, 3)])

    def test_each_quad_audits_the_pair_traces_of_its_own_slots_never_the_other_block(self):
        coordinator = self.coordinator()
        devices = [bridge.request.runtime.drafter for bridge in self.bridges((4, 5, 6, 7))]
        pairs = coordinator._quad_audit_pairs(devices, [1, 2, 3, 4], 1, ((4, 5), (6, 7)))
        self.assertEqual([labels for labels, _ in pairs], [[4, 5], [6, 7]])
        self.assertEqual([(trace.device_a.pool_slot.index, trace.device_b.pool_slot.index) for _, trace in pairs], [(4, 5), (6, 7)])
        pinned = coordinator._quad_audit_pairs([bridge.request.runtime.drafter for bridge in self.bridges((0, 1, 2, 3))], [1, 2, 3, 4], 1)
        self.assertEqual([labels for labels, _ in pinned], [[0, 1], [2, 3]], 'the four-seat call is unchanged')


class FenceTests(Harness):
    def test_a_failure_in_the_second_block_fences_the_first_blocks_replay_before_its_work_is_dropped(self):
        # Block B raises before the caller has set its fence: block A's replay may still run, so it is fenced first (the sync is logged).
        coordinator, bridges = self.coordinator(), self.bridges(EIGHT)
        real = quad_draft_tp.refusal
        calls = []

        def refusal(devices, batched, environ=None):
            calls.append(tuple(device.pool_slot.index for device in devices))
            if len(calls) == 2:
                raise RuntimeError('block B failed before its own guard')
            return None

        with patch('quad_draft_tp.refusal', Mock(side_effect=refusal)):
            with self.assertRaises(RuntimeError):
                self.prepare(coordinator, bridges)
        self.assertEqual(calls, [(0, 1, 2, 3), (4, 5, 6, 7)])
        self.assertIn(('sync',), LOG, 'block A was fenced before the exception reached the caller')
        self.assertLess(LOG.index(('quad', (100, 101, 102, 103))), LOG.index(('sync',)))
        self.assertIsNotNone(real)

    def test_a_first_block_failure_has_nothing_to_fence(self):
        coordinator, bridges = self.coordinator(), self.bridges(EIGHT)

        def refusal(devices, batched, environ=None):
            raise RuntimeError('block A failed')

        with patch('quad_draft_tp.refusal', Mock(side_effect=refusal)):
            with self.assertRaises(RuntimeError):
                self.prepare(coordinator, bridges)
        self.assertNotIn(('sync',), LOG)


class RefusalTests(Harness):
    def test_a_value_but_two_is_refused_by_name_and_no_quad_forms(self):
        for value in ('1', '3', 'two'):
            with self.subTest(value=value), patch.dict(os.environ, {BLOCKS: value}):
                base.FakeQuadTrace.instances = []
                coordinator = self.coordinator()
                _, lines = self.prepare(coordinator, self.bridges(EIGHT))
                self.assertEqual(base.FakeQuadTrace.instances, [])
                self.assertEqual(len(self.Pair.instances), 4, 'every pair drafts packed: no quad, no loss')
                disabled = [line for line in lines if quad_draft.DISABLED_MARKER in line]
                self.assertEqual(len(disabled), 2, lines)
                self.assertIn('must_be_2', disabled[0])
                self.assertEqual(sorted(coordinator.quad_blocked), [(0, 1, 2, 3), (4, 5, 6, 7)])
                self.Pair.instances = []

    def test_the_named_reasons(self):
        ok = {FLAG: '1', BLOCKS: '2', 'QWEN_FAST_TP': '4', 'QWEN_FAST_EXTENT_REPLAY': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1'}
        self.assertIsNone(quad_draft_tp.blocks_refusal(8, ok))
        self.assertIsNone(quad_draft_tp.blocks_refusal(None, ok))
        # a round has no pool to ask about: the extent replay and the packed proposal are the attach's question (users given) only
        self.assertIsNone(quad_draft_tp.blocks_refusal(None, {k: v for k, v in ok.items() if k != 'QWEN_FAST_EXTENT_REPLAY'}))
        self.assertIsNone(quad_draft_tp.blocks_refusal(3, {}), 'off: nothing to refuse')
        self.assertIsNone(quad_draft_tp.blocks_refusal(3, {BLOCKS: '0'}))
        for environ, users, text in ((dict(ok, **{BLOCKS: '3'}), None, 'must be 2'),
                                     (dict(ok, **{BLOCKS: '1'}), None, 'must be 2'),
                                     (dict(ok, QWEN_FAST_TP='2'), None, 'QWEN_FAST_TP=4 only'),
                                     ({k: v for k, v in ok.items() if k != 'QWEN_FAST_TP'}, None, 'QWEN_FAST_TP=4 only'),
                                     (dict(ok, **{FLAG: '0'}), None, 'needs QWEN_FAST_QUAD_DRAFT=1'),
                                     ({k: v for k, v in ok.items() if k != FLAG}, None, 'needs QWEN_FAST_QUAD_DRAFT=1'),
                                     (dict(ok, **{quad_draft.AUDIT_FLAG: 'all'}), None, 'QWEN_FAST_QUAD_DRAFT_AUDIT'),
                                     ({k: v for k, v in ok.items() if k != 'QWEN_FAST_EXTENT_REPLAY'}, 8, 'needs QWEN_FAST_EXTENT_REPLAY=1'),
                                     (dict(ok, QWEN_FAST_EXTENT_REPLAY='0'), 8, 'needs QWEN_FAST_EXTENT_REPLAY=1'),
                                     ({k: v for k, v in ok.items() if k != 'QWEN_FAST_PACKED_PROPOSAL'}, 8, 'needs QWEN_FAST_PACKED_PROPOSAL=1'),
                                     (ok, 4, 'needs eight seats (the pool has 4)'),
                                     (ok, 16, 'needs eight seats (the pool has 16)')):
            with self.subTest(text=text, users=users):
                reason = quad_draft_tp.blocks_refusal(users, environ)
                self.assertIsNotNone(reason)
                self.assertIn(text, reason)

    def test_the_pairs_pinned_module_names_its_refusal(self):
        import dflash_packed_proposal_coordinator as coordinator

        self.assertFalse(hasattr(quad_draft, 'QUADS'))
        reason = coordinator.quad_blocks_refusal(quad_draft)
        self.assertIn('QWEN_FAST_TP=4 only', reason)
        self.assertIn('two-quad draft', reason)

    def test_a_refusal_that_concerns_one_block_blocks_that_block_only(self):
        self.refusals = [((4, 5, 6, 7), 'the four devices do not share one draft weight set')]
        coordinator = self.coordinator()
        _, lines = self.prepare(coordinator, self.bridges(EIGHT))
        self.assertEqual([[device.pool_slot.index for device in trace.devices] for trace in base.FakeQuadTrace.instances], [[0, 1, 2, 3]])
        disabled = [line for line in lines if quad_draft.DISABLED_MARKER in line]
        self.assertEqual(len(disabled), 1)
        self.assertIn('slots=4,5,6,7', disabled[0])
        self.assertIn('do_not_share_one_draft_weight_set', disabled[0])
        self.assertEqual(sorted(coordinator.quad_blocked), [(4, 5, 6, 7)])
        self.assertFalse(coordinator.quad_disabled, 'the process-wide give-up is the four-seat quad\'s')
        self.assertEqual(len(self.Pair.instances), 2, 'block B drafts as its two pairs')
        self.assertFalse(base.FakeQuadTrace.instances[0].closed, 'the other block still serves')
        _, lines = self.prepare(coordinator, self.bridges(EIGHT))
        self.assertFalse([line for line in lines if quad_draft.DISABLED_MARKER in line], 'logged once')

    def test_two_failures_in_a_row_block_one_quad_and_the_other_keeps_serving(self):
        coordinator, bridges = self.coordinator(), self.bridges(EIGHT)
        base.FakeQuadTrace.failures = [False, True, False, True]
        for _ in range(2):
            _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(coordinator.quad_block_failures[(4, 5, 6, 7)], 2)
        self.assertEqual(sorted(coordinator.quad_blocked), [(4, 5, 6, 7)])
        self.assertEqual(sorted(coordinator.quad_blocks), [(0, 1, 2, 3)])
        self.assertEqual(coordinator.quad_block_failures.get((0, 1, 2, 3)), 0)
        self.assertTrue(any('consecutive_failures=2' in line and 'slots=4,5,6,7' in line for line in lines), lines)

    def test_the_attach_refuses_a_pool_that_is_not_eight_slots_by_name(self):
        import dflash_packed_proposal_coordinator as coordinator

        for users in (4, 6, 16):
            for shapes in (coordinator.pooled_draft_mask_shapes, coordinator.pooled_draft_output_shapes):
                with self.subTest(users=users, shapes=shapes.__name__), self.assertRaises(ValueError) as caught:
                    shapes(users, 16)
                self.assertIn('needs eight seats (the pool has %d)' % users, str(caught.exception))
                self.assertIn(BLOCKS, str(caught.exception))

    def test_the_attach_refuses_the_flag_whatever_else_is_set(self):
        # Requested with the quad flag off, or without the extent replay or the packed proposal, the attach refuses by name: a later round never
        # silently skips the flag (and a quad never builds without the pool's pre-trace buffers).
        import dflash_packed_proposal_coordinator as coordinator

        base_flags = {FLAG: '1', BLOCKS: '2', 'QWEN_FAST_TP': '4', 'QWEN_FAST_EXTENT_REPLAY': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1'}
        for name, flags, text in (('quad flag off', {FLAG: '0'}, 'needs QWEN_FAST_QUAD_DRAFT=1'),
                                  ('quad flag unset', {FLAG: None}, 'needs QWEN_FAST_QUAD_DRAFT=1'),
                                  ('no extent replay', {'QWEN_FAST_EXTENT_REPLAY': None}, 'QWEN_FAST_EXTENT_REPLAY=1'),
                                  ('no packed proposal', {'QWEN_FAST_PACKED_PROPOSAL': None}, 'QWEN_FAST_PACKED_PROPOSAL=1')):
            for shapes in (coordinator.pooled_draft_mask_shapes, coordinator.pooled_draft_output_shapes):
                with self.subTest(name=name, shapes=shapes.__name__):
                    merged = dict(base_flags, **flags)
                    with patch.dict(os.environ, {key: value for key, value in merged.items() if value is not None}, clear=False):
                        for key, value in merged.items():
                            if value is None:
                                os.environ.pop(key, None)
                        with self.assertRaises(ValueError) as caught:
                            shapes(8, 16)
                    self.assertIn(text, str(caught.exception))
                    self.assertIn(BLOCKS, str(caught.exception))
        with patch.dict(os.environ, base_flags):
            self.assertIn((4, 5, 6, 7), coordinator.pooled_draft_mask_shapes(8, 16))
            coordinator.refuse_quad_blocks(8)

    def test_the_attach_calls_the_refusal_outside_the_extent_branch(self):
        # Without QWEN_FAST_EXTENT_REPLAY serving_runtime builds no pool shapes at all, so the refusal must not live only in them.
        with open(os.path.join(HERE, 'serving_runtime.py'), encoding='utf-8') as handle:
            text = handle.read()
        call = text.index('refuse_quad_blocks(policy')
        self.assertLess(call, text.index('draft_masks = pooled_draft_mask_shapes('))
        self.assertIn("QWEN_FAST_QUAD_DRAFT_BLOCKS', '') not in ('', '0')", text[call - 700:call])
        self.assertNotIn('if extent_replay', text[call - 700:call].split('draft_masks = {}')[-1], 'unconditional on the extent replay')

    def test_the_attach_refuses_another_block_width(self):
        import dflash_packed_proposal_coordinator as coordinator

        with patch.dict(os.environ, {'QWEN_FAST_EXTENT_REPLAY': '1'}), self.assertRaises(ValueError) as caught:
            coordinator._pooled_quad_blocks(quad_draft_tp, 8, 32)
        self.assertIn('16-row T16', str(caught.exception))


class PooledShapesTests(Harness):
    def test_eight_seats_pool_a_mask_and_an_output_set_for_each_quad(self):
        import dflash_packed_proposal_coordinator as coordinator

        with patch.dict(os.environ, {'QWEN_FAST_EXTENT_REPLAY': '1'}):
            masks = coordinator.pooled_draft_mask_shapes(8, 16)
            outputs = coordinator.pooled_draft_output_shapes(8, 16)
        for group in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1), (2, 3), (4, 5), (6, 7)):
            self.assertIn(group, masks)
        self.assertEqual(masks[(0, 1, 2, 3)], masks[(4, 5, 6, 7)])
        self.assertEqual(outputs[(0, 1, 2, 3)], outputs[(4, 5, 6, 7)])
        self.assertEqual(outputs[(4, 5, 6, 7)]['head'], (1, 1, quad_draft.ROWS, 16))


class BlocksOffTests(unittest.TestCase):
    """The flag off, the pool and the coordinator are the parent's."""

    def test_no_second_quad_buffers_without_the_flag(self):
        import dflash_packed_proposal_coordinator as coordinator

        for blocks in (None, '', '0'):
            flags = {FLAG: '1', 'QWEN_FAST_TP': '4'}
            if blocks is not None:
                flags[BLOCKS] = blocks
            with self.subTest(blocks=blocks), environment(**flags):
                masks = coordinator.pooled_draft_mask_shapes(8, 16)
                outputs = coordinator.pooled_draft_output_shapes(8, 16)
                self.assertIn((0, 1, 2, 3), masks)
                self.assertNotIn((4, 5, 6, 7), masks)
                self.assertNotIn((4, 5, 6, 7), outputs)

    def test_the_marker_is_once_per_process_without_the_flag(self):
        with environment(), patch.object(quad_draft, '_NOTED', []):
            self.assertTrue(quad_draft_tp.note((0, 1, 2, 3), 'fold', '110', log=lambda text: None))
            self.assertFalse(quad_draft_tp.note((4, 5, 6, 7), 'fold', '110', log=lambda text: None))

    def test_the_marker_is_once_per_quad_with_the_flag(self):
        lines = []
        with environment(**{BLOCKS: '2', 'QWEN_FAST_TP': '4'}), patch.object(quad_draft, '_NOTED', []):
            self.assertTrue(quad_draft_tp.note((0, 1, 2, 3), 'fold', '110', log=lines.append))
            self.assertFalse(quad_draft_tp.note((0, 1, 2, 3), 'fold', '110', log=lines.append))
            self.assertTrue(quad_draft_tp.note((4, 5, 6, 7), 'fold', '110', log=lines.append))
        self.assertEqual(len(lines), 2)
        self.assertIn('slots=[0,1,2,3]', lines[0])
        self.assertIn('slots=[4,5,6,7]', lines[1])
        self.assertTrue(all(line.startswith(quad_draft.MARKER) for line in lines))


class FlagOffIsTheParentTests(Harness):
    """With the blocks flag off the coordinator is tp4/next-5's parent's, call for call, at four and eight seats and for every mix
    the quad flag meets - including QWEN_FAST_QUAD_DRAFT=1, whose first-four-slots quad is today's."""

    blocks = None

    def parent(self):
        from test_dflash_proposal_trace import pinned_module

        module = pinned_module('dflash_packed_proposal_coordinator.py', 'dflash_packed_proposal_coordinator_blocks_parent', commit=PARENT)
        if module is None:
            self.skipTest('no git history for %s' % PARENT)
        return module

    def run_round(self, module, slots, quad_flag, blocks, rows=2048):
        del LOG[:]
        base.FakeQuadTrace.instances, self.Pair.instances = [], []
        flags = {'QWEN_FAST_TP': '4'}
        if quad_flag is not None:
            flags[FLAG] = quad_flag
        if blocks is not None:
            flags[BLOCKS] = blocks
        with environment(**{FLAG: '0'}), patch.dict(os.environ, flags):
            coordinator = module.PackedProposalCoordinator()
            bridges = self.bridges(slots, history_rows=rows)
            _, lines = self.prepare(coordinator, bridges)
            _, more = self.prepare(coordinator, bridges)
        return list(LOG), [base.strip_ms(line) for line in lines + more], [len(base.FakeQuadTrace.instances), len(self.Pair.instances)]

    def test_every_mix_matches_the_parent_with_the_blocks_flag_unset_empty_or_zero(self):
        import dflash_packed_proposal_coordinator as today

        parent = self.parent()
        for slots, rows in ((EIGHT, 2048), ((0, 1, 2, 3), 2048), ((0, 1, 2, 3, 4, 5), 2048), ((4, 5, 6, 7), 2048), ((0, 1), 2048),
                            (EIGHT, 1024), ((0, 1, 2, 3), 1024)):
            for quad_flag in (None, '0', '1'):
                before = self.run_round(parent, slots, quad_flag, None, rows)
                for blocks in (None, '', '0'):
                    with self.subTest(slots=slots, rows=rows, quad=quad_flag, blocks=blocks):
                        self.assertEqual(self.run_round(today, slots, quad_flag, blocks, rows), before)

    def test_at_eight_seats_with_the_quad_flag_only_no_quad_forms_today(self):
        import dflash_packed_proposal_coordinator as today

        _, lines, counts = self.run_round(today, EIGHT, '1', None)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 4)


class CaptureHeadroomTests(Harness):
    def test_the_arithmetic(self):
        reserve = 256 * MIB
        self.assertEqual(quad_draft.QUAD_CAPTURE_BYTES_EST, 450 * MIB)
        self.assertEqual(quad_draft_tp.blocks_capture_bytes(1), 450 * MIB)
        self.assertEqual(quad_draft_tp.blocks_capture_bytes(2), 900 * MIB)
        self.assertEqual(quad_draft_tp.blocks_capture_bytes(), 900 * MIB)
        self.assertEqual(quad_draft_tp.blocks_capture_need(1, reserve), 706 * MIB)
        self.assertEqual(quad_draft_tp.blocks_capture_need(2, reserve), 1156 * MIB, 'the reserve is added once, not per quad')
        self.assertEqual(quad_draft_tp.blocks_capture_need(2, reserve) - quad_draft_tp.blocks_capture_need(1, reserve), 450 * MIB)

    def test_the_coordinators_check_for_one_quad_and_for_the_pair_of_them(self):
        import dflash_packed_proposal_coordinator as coordinator

        device = SimpleNamespace()
        need_one = quad_draft.QUAD_CAPTURE_BYTES_EST + coordinator.dram_reserve_bytes()
        need_two = quad_draft_tp.blocks_capture_need(2, coordinator.dram_reserve_bytes())
        self.assertEqual(need_one, 706 * MIB)
        self.assertEqual(need_two, 1156 * MIB)
        for headroom, estimate, fits in ((need_one, quad_draft.QUAD_CAPTURE_BYTES_EST, True),
                                         (need_one - 1, quad_draft.QUAD_CAPTURE_BYTES_EST, False),
                                         (need_two, quad_draft_tp.blocks_capture_bytes(2), True),
                                         (need_two - 1, quad_draft_tp.blocks_capture_bytes(2), False),
                                         (need_one, quad_draft_tp.blocks_capture_bytes(2), False)):
            with self.subTest(headroom=headroom, estimate=estimate), patch.object(coordinator, 'dram_headroom', return_value=headroom):
                short, reading = coordinator.capture_headroom(device, estimate)
                self.assertEqual(short == (), fits)
                self.assertEqual(reading, dict(largest_free=headroom))

    def test_the_second_quad_is_checked_after_the_first_and_falls_back_to_its_pairs(self):
        import dflash_packed_proposal_coordinator as coordinator

        seen = []

        def headroom(device, estimate, reserve=True):
            seen.append((device.pool_slot.index, estimate))
            if estimate != quad_draft.QUAD_CAPTURE_BYTES_EST:
                return (), dict(largest_free=3)          # the pair release's own check
            # room for the first quad, none left for the second
            first = len([call for call in seen if call[1] == quad_draft.QUAD_CAPTURE_BYTES_EST]) == 1
            return ((), dict(largest_free=1)) if first else (('largest_free',), dict(largest_free=2))

        with patch.object(coordinator, 'capture_headroom', side_effect=headroom):
            coord = self.coordinator()
            _, lines = self.prepare(coord, self.bridges(EIGHT))
        self.assertEqual([call for call in seen if call[1] == quad_draft.QUAD_CAPTURE_BYTES_EST], [(0, 450 * MIB), (4, 450 * MIB)])
        self.assertEqual([[device.pool_slot.index for device in trace.devices] for trace in base.FakeQuadTrace.instances], [[0, 1, 2, 3]])
        self.assertEqual(len(self.Pair.instances), 2, 'slots 4-7 draft as pairs (4, 5) and (6, 7)')
        self.assertTrue(any('[QUAD-DRAFT] fallback' in line and 'dram_reserve' in line for line in lines), lines)
        self.assertFalse(coord.quad_blocked, 'a short headroom is a round\'s fallback, never a give-up')


class SmokeRulesTests(unittest.TestCase):
    ENV = {FLAG: '1', BLOCKS: '2'}

    @staticmethod
    def log(*, markers=('0,1,2,3', '4,5,6,7'), rounds=5, extra=(), pooled=('0,1,2,3', '4,5,6,7')):
        lines = ['[PINDIAG] quad draft engaged slots=[%s] heads=32/8 rows=64 sdpa=fold conv=110' % slots for slots in markers]
        lines += ['[PINDIAG] draft mask pooled slots=[%s] shape=1x1x64x4160' % slots for slots in pooled]
        lines += ['[PINDIAG] draft outputs pooled slots=[%s] head=1x1x64x16 projected=1x1x64x256' % slots for slots in pooled]
        lines += ['[PACKED-SELECT] round=%d pairs=[[0, 1, 2, 3], [4, 5, 6, 7]] users=8 calls=1 collect_ms=1.0 select_ms=2.0' % n
                  for n in range(1, rounds + 1)]
        lines += list(extra)
        return chr(10).join(lines)

    def problems(self, text, env=None, steady_eight=True):
        return check.draft_problems(check.draft_facts(text), self.ENV if env is None else env, False, steady_eight)

    def test_two_markers_one_per_block_and_a_round_of_both_pass(self):
        self.assertEqual(self.problems(self.log()), [])

    def test_the_four_user_rules_are_not_the_blocks_rules(self):
        # concurrent4_steady alone cannot judge two quads: the log is not read, and the profile is reported as not judged
        problems = self.problems(self.log(markers=('0,1,2,3',), rounds=0), steady_eight=False)
        self.assertEqual(len(problems), 1)
        self.assertIn('the blocks were not judged', problems[0])
        self.assertIn('concurrent8_steady', problems[0])

    def test_a_blocks_profile_without_concurrent8_steady_fails_the_whole_check(self):
        smoke = {'warmup': {}, 'concurrent4_steady': {}}
        env = {FLAG: '1', BLOCKS: '2', 'QWEN_FAST_TP': '4'}
        drafts = check.draft_facts(self.log())
        steady = 'concurrent4_steady' in smoke
        self.assertTrue(check.draft_problems(drafts, env, steady, 'concurrent8_steady' in smoke))
        self.assertEqual(check.draft_problems(drafts, env, steady, True), [])

    def test_the_pooled_buffers_of_both_quads_are_required(self):
        for pooled in (('0,1,2,3',), ('4,5,6,7',), ()):
            with self.subTest(pooled=pooled):
                problems = self.problems(self.log(pooled=pooled))
                self.assertEqual(len(problems), 2, problems)
                self.assertTrue(any('draft mask pooled' in problem for problem in problems))
                self.assertTrue(any('draft outputs pooled' in problem for problem in problems))
        only_mask = self.log().replace('[PINDIAG] draft outputs pooled slots=[4,5,6,7]', '[PINDIAG] draft outputs x slots=[4,5,6,7]')
        problems = self.problems(only_mask)
        self.assertEqual(len(problems), 1)
        self.assertIn('slots 4,5,6,7', problems[0])

    def test_any_pooled_refused_line_fails(self):
        for line in ('[PINDIAG] draft mask pooled refused slots=[4,5,6,7] shape=1x1x64x4160: the pool holds []',
                     '[PINDIAG] draft outputs pooled refused slots=[4,5,6,7]: copy refused: X: y',
                     '[PINDIAG] draft outputs pooled refused at attach groups=[[4, 5, 6, 7]]: ValueError: x'):
            with self.subTest(line=line):
                problems = self.problems(self.log(extra=[line]))
                self.assertEqual(len(problems), 1, problems)
                self.assertIn('pooled refused', problems[0])

    def test_with_the_singles_audit_each_quad_must_have_an_equal_audit_line(self):
        env = dict(self.ENV, QWEN_FAST_DRAFT_SINGLES_AUDIT='all')

        def audit(group, equal=1):
            return '[DRAFT-SINGLES-AUDIT] round=3 group=%s equal=%d stage=head checks=4 ' % (group, equal)

        both = self.log(extra=[audit('[0, 1, 2, 3]'), audit('[4, 5, 6, 7]')])
        self.assertEqual(self.problems(both, env), [])
        for kept in ('[0, 1, 2, 3]', '[4, 5, 6, 7]'):
            with self.subTest(kept=kept):
                problems = self.problems(self.log(extra=[audit(kept)]), env)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn('no [DRAFT-SINGLES-AUDIT]', problems[0])
        # a pair's audit line (group=[4, 5]) is not the quad's
        problems = self.problems(self.log(extra=[audit('[0, 1]'), audit('[4, 5]')]), env)
        self.assertEqual(len(problems), 1)
        # an unequal line fails by the existing rule
        self.assertTrue(self.problems(self.log(extra=[audit('[0, 1, 2, 3]'), audit('[4, 5, 6, 7]'), audit('[4, 5, 6, 7]', 0)]), env))
        # without the audit flag the audit lines are not required
        self.assertEqual(self.problems(self.log()), [])

    def test_one_marker_a_wrong_pair_of_markers_or_three_fail(self):
        for markers in (('0,1,2,3',), ('4,5,6,7',), ('0,1,2,3', '0,1,2,3'), ('0,1,2,3', '4,5,6,7', '0,1,2,3'), ()):
            with self.subTest(markers=markers):
                problems = self.problems(self.log(markers=markers))
                self.assertTrue(problems)
                self.assertIn('not once for each of 0,1,2,3 and 4,5,6,7', problems[0])

    def test_no_round_of_both_quads_fails(self):
        problems = self.problems(self.log(rounds=0))
        self.assertEqual(len(problems), 1)
        self.assertIn('no round was served by both quads', problems[0])

    def test_a_fallback_or_a_disable_fails(self):
        self.assertTrue(self.problems(self.log(extra=['[QUAD-DRAFT] fallback round=9 reason=RuntimeError:x'])))
        self.assertTrue(self.problems(self.log(extra=['[PINDIAG] quad draft disabled round=9 failures=2 reason=x slots=4,5,6,7'])))

    def test_a_blocks_value_other_than_two_or_without_the_quad_flag_fails_whatever_ran(self):
        self.assertIn('only 2 is served', self.problems(self.log(), {FLAG: '1', BLOCKS: '3'}, False)[0])
        self.assertIn('no quad can form', self.problems(self.log(), {BLOCKS: '2', FLAG: '0'}, False)[0])
        # the image sets QWEN_FAST_QUAD_DRAFT=1: a profile that does not name it relies on that default and is not refused for it
        self.assertFalse([problem for problem in self.problems(self.log(), {BLOCKS: '2'}, True) if 'no quad can form' in problem])
        self.assertEqual(self.problems(self.log(), {FLAG: '1', BLOCKS: '0'}, False), [])

    def test_the_profile_flag_makes_the_profile_a_fast_path_one(self):
        self.assertTrue(check.fast_path({BLOCKS: '2'}))

    def test_without_the_blocks_flag_the_four_user_rules_are_unchanged(self):
        text = chr(10).join(['[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=32/8 rows=64 sdpa=fold conv=110',
                             '[PACKED-SELECT] round=1 pairs=[[0, 1, 2, 3]] users=4 calls=1 collect_ms=1.0 select_ms=2.0'])
        self.assertEqual(check.draft_problems(check.draft_facts(text), {FLAG: '1'}, True), [])
        self.assertTrue(check.draft_problems(check.draft_facts(''), {FLAG: '1'}, True))


class ProfileTests(unittest.TestCase):
    def profiles(self):
        import json

        with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
            return json.load(handle)

    def test_the_profile_is_the_eight_best_plus_exactly_the_blocks_flag_and_is_gate_only(self):
        found = self.profiles()
        mine, parent = found['profiles']['c2-packed-tp4-8-best-quad'], found['profiles']['c2-packed-tp4-8-best']
        self.assertEqual(mine['env'], dict(parent['env'], **{BLOCKS: '2'}))
        self.assertEqual(mine['env'][FLAG], '1')
        self.assertEqual(sorted(key for key in set(mine) | set(parent) if key not in ('env', 'description') and mine.get(key) != parent.get(key)), [])
        self.assertIs(mine['gate_only'], True)
        self.assertTrue(mine['description'].startswith('GATE ONLY'))
        self.assertEqual(mine['engine']['max-num-seqs'], 8)
        self.assertEqual(found['default'], 'c2-packed-tp4')

    def test_the_flag_is_in_no_other_profile(self):
        self.assertEqual(sorted(name for name, body in self.profiles()['profiles'].items() if BLOCKS in body.get('env', {})),
                         ['c2-packed-tp4-8-best-quad', 'c2-packed-tp4-8-best-quad-dbf16', 'c2-packed-tp4-8-best-quad-gate', 'c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit', 'c2-packed-tp4-8x262k-best-levern-audit', 'c2-packed-tp4-8x262k-best-levern-control-audit', 'c2-packed-tp4-8x262k-best-levern-final-hold-time-gate', 'c2-packed-tp4-8x262k-best-levern-foreign-time-gate', 'c2-packed-tp4-8x262k-best-levern-hang-gate', 'c2-packed-tp4-8x262k-best-levern-r1-time-gate', 'c2-packed-tp4-8x262k-best-levern-time-gate', 'c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-sdpamulti', 'c2-packed-tp4-8x262k-best-sdpamulti-audit', 'c2-packed-tp4-8x262k-best-stack-audit',
                          'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-time-gate-d2', 'c2-packed-tp4-8x262k-best-time-gate-dbf16', 'c2-packed-tp4-8x262k-best-time-gate-lookup', 'c2-packed-tp4-8x262k-best-time-gate-nosamp', 'c2-packed-tp4-8x262k-best-time-gate-s1', 'c2-packed-tp4-8x262k-best-time-gate-stack', 'c2-packed-tp4-8x262k-best-time-gate-u1', 'c2-packed-tp4-8x262k-best-u1-audit', 'c2-packed-tp4-8x262k-hostgap-1', 'c2-packed-tp4-8x262k-hostgap-1-audit', 'c2-packed-tp4-8x262k-hostgap-2', 'c2-packed-tp4-8x262k-hostgap-2-audit', 'c2-packed-tp4-8x262k-prefix-gate', 'c2-packed-tp4-8x262k-prefix-time-gate', 'c2-packed-tp4-8x262k-ship', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-dbf16', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1', 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit'])

    def test_the_smoke_rules_read_the_profiles_env(self):
        env = self.profiles()['profiles']['c2-packed-tp4-8-best-quad']['env']
        self.assertEqual((env[FLAG], env[BLOCKS]), ('1', '2'))
        self.assertTrue(check.draft_problems(check.draft_facts(''), env, False, True), 'a log with no marker fails it')
        gate = self.profiles()['profiles']['c2-packed-tp4-8-best-quad-gate']['env']
        self.assertEqual(gate['QWEN_FAST_DRAFT_SINGLES_AUDIT'], 'all')
        problems = check.draft_problems(check.draft_facts(SmokeRulesTests.log()), gate, False, True)
        self.assertEqual(len(problems), 1, 'the markers and the pooled lines pass; the missing audit lines fail')
        self.assertIn('no [DRAFT-SINGLES-AUDIT]', problems[0])

    def test_the_description_names_no_host_address_registry_or_digest(self):
        from test_tp4_next5 import BANNED

        self.assertIsNone(BANNED.search(self.profiles()['profiles']['c2-packed-tp4-8-best-quad']['description']))


if __name__ == '__main__':
    unittest.main()
