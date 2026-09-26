"""S2 W3: the packed block and model_batch at any position (design s2-design.md section 4, W3).

The block is the REAL PackedVerifierEngine over a fake two-chip device, and its attention is the
REAL extent reader (extent_attention_replay.PackedExtentReplayReader) over pool-lent storage of the
exact geometry serving_buffer_pool.PackedExtentStorage lends: test_packed_verifier's fake fixture
builds it the way model_batch does under `packed_extent`. The pinned mask kernel's program is
stood in for (prepare_narrow needs ttnn) and its in-trace refresh is simulated by the host
transliteration of the kernel (narrow_mask_host), run from the device word when the verify trace
replays - so what the extent audit reads back is what the kernel would have written from what was
staged. Covered:
  - a four-user round at mixed families stages each segment's word (start & 255), cur_pos (E - 1)
    and full-width table, and logs the executed path's round line;
  - a padded round: idle segments at start 0 and 32 (E = 256) on the zero table, on page 0's tile
    rows 0 and 1, the T2 tile rows disjoint;
  - the boundary cap's block backstop (before commit_user's try) and admits' floor and ceiling;
  - the extent page-0 range; the prestage diff (cur_pos written only on a family change);
  - the extent audit, clean and against an absolute word, a stale cur_pos and a corrupted mask tile;
  - the gate-only capture position (flag off in family 16640, and on the extent block);
  - the replay deadline;
  - validate_bindings over the new buffers, describe, the construction refusals;
  - model_batch's own extent branch, and that verify_prestage.py and padded_probe.py need no edit.
Flag off - a pool without extent storage - every existing test of the block is unchanged
(test_packed_verifier, test_padded_block, test_verify_prestage, test_padded_probe and the parent
comparisons in test_round_fences, test_fused_commit and test_gdn_seq_block).
"""

from contextlib import ExitStack
from itertools import count
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from attention_mask_replay import validate_ticket
import extent_attention_replay
from extent_attention_replay import accept_limit, extent, extent_values, narrow_mask_host
import packed_verifier
from packed_verifier import PackedVerifierEngine, m1_shape, m3_shape, page_zero_index
from serving_buffer_pool import PackedExtentStorage
import test_packed_verifier as base
import verifier_engine
import verify_prestage
import verify_trace_t2

HERE = Path(__file__).resolve().parent
C = 131328                      # the served capacity, page width 2052
WIDTH = C // 64
ENV = {'QWEN_FAST_SDPA_MODES': 'tail,share,slice', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1',
       'QWEN_FAST_REPLAY_GROUP_ROWS': '8'}
# Everything the block or the step reads that a developer's shell could hold: cleared for each test.
CLEARED = ('QWEN_FAST_EXTENT_AUDIT', 'QWEN_FAST_REPLAY_DEADLINE_S', 'QWEN_FAST_PRESTAGE', 'QWEN_FAST_PRESTAGE_AUDIT',
           'QWEN_FAST_ROUND_FENCES', 'QWEN_FAST_PADDED_PROBE', 'QWEN_FAST_PACKED_AUDIT', 'QWEN_FAST_VERIFY_T1',
           'QWEN_FAST_VERIFY_T2', 'QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_GDN_AFTER_PAIRS', 'QWEN_FAST_PIPELINED_COMMITS',
           'QWEN_FAST_GATE_FORCE_CAP', 'QWEN_FAST_PACKED_CAPTURE_POSITION', 'QWEN_FAST_PROFILE_DUMP_ROUND',
           'QWEN_FAST_GDN_SEQ_BLOCK', 'QWEN_FAST_GDN_SEQ_BLOCK_AUDIT', 'QWEN_CONTEXT_LADDER_SIM')


class ExtentTTNN(base.FakeTTNN):
    """test_packed_verifier's fake device, plus what the extent readers also read (each device
    tensor's memory config) and a hook on every trace replay."""

    def __init__(self):
        super().__init__()
        self.on_trace = None

    def allocate(self, shape, dtype='bf16', layout='tile', value=None, mapper=None):
        tensor = super().allocate(shape, dtype, layout, value, mapper)
        tensor.memory_config = lambda: self.DRAM_MEMORY_CONFIG
        return tensor

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        super().execute_trace(mesh, trace, cq_id=cq_id, blocking=blocking)
        if self.on_trace is not None:
            self.on_trace(trace)


def extent_pool(ttnn, shared, users=4, rows=16, page_width=WIDTH):
    """A pool built with extent_replay: its PackedExtentStorage for (users, rows) - per user one
    bundle's (2, page_width) table and (2,) cur_pos, zeroed - and no per-family tables."""
    slots = [SimpleNamespace(index=index, lent=False, verifier=SimpleNamespace(carry=base.snapshot_set(ttnn)))
             for index in range(users)]

    def integers(shape):
        return ttnn.allocate(shape, 'int32', 'row_major', torch.zeros(shape, dtype=torch.int32))

    storage = PackedExtentStorage(users, rows, [[integers((2, page_width))] for user in range(users)],
                                  [[integers((2,))] for user in range(users)])

    def packed_extent(count_, width):
        if (count_, width) != (users, rows):
            raise ValueError('The pool holds packed extent storage for shapes [%r]; %r was asked for'
                             % ((users, rows), (count_, width)))
        return storage

    def packed_replay(count_, width):
        raise ValueError('The pool holds S2 extent storage (extent_replay) and no per-family packed replay tables')

    return SimpleNamespace(closed=False, helpers=shared, page_width=page_width, slots=slots, extent_replay=True,
                           packed_extent=packed_extent, packed_replay=packed_replay, storage=storage)


class ExtentModelBatch(base.FakeModelBatch):
    """test_packed_verifier's fake fixture, with the REAL extent reader built the way model_batch
    builds it under `packed_extent`: one ExtentSegmentReader per segment over the block's lent
    storage, each constructed (and so staged) at its segment's capture start."""

    ttnn = None
    instances = []

    def __init__(self, model, tokens, start, pages, helpers, checkpoints, prefix, **options):
        storage = options.get('packed_extent')
        if storage is None or 'packed_replay_pages' in options:
            raise AssertionError('The extent block passes packed_extent and no per-family tables')
        super().__init__(model, tokens, start, pages, helpers, checkpoints, prefix, **dict(options, attention_replay=False))
        from extent_attention_replay import PackedExtentReplayReader
        from target_packed_pages import segments

        spans, total = segments(self.pack)
        self.replay_capacity = pages.shape[1] * 64
        self.replay_reader = PackedExtentReplayReader(type(self).ttnn, model.mesh_device, spans, pages.shape[1],
            [user['pages'] for user in self.pack], storage=storage, max_group_rows=options['replay_group_rows'],
            starts=tuple(user['start'] for user in self.pack))
        self.grouped_readers.append(self.replay_reader)
        self.readers = [self.replay_reader] * 16


def simulated_mask_kernel(positions, mask, program):
    """attention_mask_replay.execute's stand-in: the pinned kernel at capacity 256, from the device's
    own word (its host transliteration, narrow_mask_host, is what CB2b R1 compares the kernel with)."""
    mask.value = narrow_mask_host(int(positions.value[0]), program['rows'], program['batches'], program['offset'])


def owner(block_fixture, segment, position, page_value=7):
    return base.request('R%d' % segment, block_fixture.pool.slots[segment], position, page_value, page_width=WIDTH)


class ExtentFixture(unittest.TestCase):
    """The M3 block (four T16 users) at the served page width over a pool with extent storage."""

    USERS = 4

    def setUp(self):
        verifier_engine.note_prefill()
        stack = ExitStack()
        self.addCleanup(stack.close)
        # patch.dict restores the whole environment at cleanup, the names popped here included
        stack.enter_context(patch.dict(os.environ, ENV))
        for name in CLEARED:
            os.environ.pop(name, None)
        self.stack = stack
        self.ttnn = ExtentTTNN()
        ExtentModelBatch.ttnn, ExtentModelBatch.instances = self.ttnn, []
        base.FakeFeatures.instances = []
        self.helpers = base.helpers(self.ttnn)
        self.pool = extent_pool(self.ttnn, self.helpers, users=self.USERS)
        self.weights, self.model = base.weights(), base.model()
        self.traces = count(1)
        self.ids = self.ttnn.allocate((64,), 'uint32', 'row_major', torch.arange(1000, 1064, dtype=torch.int32))
        self.lines, self.engaged = [], []

        def capture_operation(operations, mesh, operation):
            return 'trace%d' % next(self.traces), operation()

        def program(mesh, positions, mask, *, rows, batches, offset):
            return dict(rows=rows, batches=batches, offset=offset)

        for target, value in (('packed_verifier.ModelBatch', ExtentModelBatch),
                              ('packed_verifier.PreparedTargetFeatures', base.FakeFeatures),
                              ('packed_verifier.prepare', Mock(return_value=Mock(name='publication'))),
                              ('packed_verifier.capture_operation', Mock(side_effect=capture_operation)),
                              ('packed_verifier.sample_rows', Mock(side_effect=lambda *args, **options: self.ids)),
                              ('packed_verifier.diagnostic', Mock(side_effect=self.lines.append)),
                              ('pooled_attention_replay._binary_checked', []),
                              ('pooled_attention_replay._pindiag', Mock()),
                              ('pooled_attention_replay.loaded_binary_has_modes',
                               Mock(return_value=('/opt/qwen-c2/opgraft-K64j/_ttnncpp.so', True))),
                              ('extent_attention_replay._pindiag', Mock(side_effect=self.engaged.append)),
                              ('extent_attention_replay.prepare_narrow', Mock(side_effect=program)),
                              ('attention_mask_replay.execute', Mock(side_effect=simulated_mask_kernel))):
            stack.enter_context(patch(target, value))

    def env(self, **values):
        self.stack.enter_context(patch.dict(os.environ, values))

    def build(self, **options):
        block = PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
                                     shared_weights=self.weights, shape=m3_shape(WIDTH), feature_taps=base.TAPS,
                                     **options)
        self.addCleanup(lambda: block.deadline.close() if block.deadline is not None else None)

        def replayed(trace):
            # The verify trace refreshes every segment's narrow masks from its staged word, in-trace.
            if trace == block.trace and block.fixture is not None:
                block.fixture.replay_reader.refresh()

        self.ttnn.on_trace = replayed
        return block

    def round(self, positions, pages=None):
        """One entry per segment in `positions` (None: that segment idle), presented last first."""
        pages = pages or {}
        entries = []
        for segment, position in enumerate(positions):
            if position is None:
                continue
            request = owner(self, segment, position, 7 + segment)
            if segment in pages:
                request.engine.pages = pages[segment]
            entries.append(base.entry(request, range(10 + segment, 26 + segment)))
        return entries[::-1]

    def readers(self, block):
        return block.fixture.replay_reader.readers

    def finish(self, block, prefix=1):
        for segment in sorted(block.pending_segments):
            block.commit_user(segment, prefix if segment not in block.idle_segments else 0)


class ExtentRoundTests(ExtentFixture):
    def test_a_four_user_round_at_mixed_families_stages_each_segments_own_family(self):
        starts = (1500, 20000, 60000, 131312)
        for start in starts:
            # the pinned family check refuses every one of them in the block's family at C
            with self.assertRaises(ValueError):
                validate_ticket(start, 16, C, short_context=False)
        block = self.build()
        self.assertTrue(block.extent)
        self.assertEqual((block.replay_capacity, block.capture_position), (C, C - 256))
        self.assertEqual(self.pool.storage.taken, True)
        self.assertEqual([reader.sdpa_modes_applied for reader in self.readers(block)], [(0x27,)] * 4)
        self.assertEqual(len(self.engaged), 2, 'one engaged line per fixture built (warm and capture)')
        self.assertTrue(all(line.startswith(extent_attention_replay.ENGAGED_MARKER) for line in self.engaged))
        entries = self.round(starts)
        predictions, metrics = block.verify(entries)
        self.assertEqual(metrics['segments'], (3, 2, 1, 0))
        self.assertEqual(predictions[0], list(range(1048, 1064)))
        for segment, (reader, start) in enumerate(zip(self.readers(block), starts)):
            word, position = extent_values(start)
            self.assertEqual(reader.start, start)
            self.assertEqual(reader.positions.value.tolist(), [start & 255] + [0] * 7)
            self.assertEqual(reader.cur_pos[0].value.tolist(), [extent(start) - 1] * 2)
            table = reader.metadata[0][1]
            self.assertEqual(tuple(table.shape), (2, WIDTH))
            self.assertTrue(torch.equal(table.value, torch.full((2, WIDTH), 7 + segment, dtype=torch.int32)))
            # the in-trace refresh wrote the narrow mask from that word
            self.assertTrue(torch.equal(reader.metadata[0][2].value.view(torch.int16),
                                        narrow_mask_host(word, 8, 2, 0).view(torch.int16)))
        self.assertEqual(block.round_starts, dict(enumerate(starts)))
        rounds = [line for line in self.lines if line.startswith(packed_verifier.EXTENT_ROUND_MARKER)]
        self.assertEqual(rounds, ['[PINDIAG] packed extent round round=1 live=4 families=[0:1536,1:20224,2:60160,'
                                  '3:131328] idle=[] capped=[]'])
        self.assertLess(len(rounds[0]), 180)
        self.assertEqual(block.extent_counts, dict(rounds=1, cap_events=0, mixed_rounds=1, cap_refused=0,
                                                   audit_mismatches=0))
        self.finish(block)
        self.assertEqual(block.phase, 'idle')
        self.assertFalse(any(line.startswith(packed_verifier.EXTENT_AUDIT_MARKER) for line in self.lines),
                         'the audit is gate-only: off without QWEN_FAST_EXTENT_AUDIT')

    def test_the_boundary_cap_backstop_refuses_before_commit_users_try_and_leaves_the_block_verified(self):
        block = self.build()
        capped = 20224 - 6            # s & 255 = 250: rows 6..15 lie at or past E = 20224
        self.assertEqual(capped & 255, 250)
        self.assertEqual((block.accept_limit(capped), accept_limit(capped, 16)), (6, 6))
        block.verify(self.round((5000, capped, 70000, 131072)))
        self.assertIn('capped=[1:6]', [line for line in self.lines
                                       if line.startswith(packed_verifier.EXTENT_ROUND_MARKER)][0])
        with self.assertRaisesRegex(ValueError, 'commits at most 6 rows; prefix 7 refused'):
            block.commit_user(1, 7)
        self.assertEqual(block.phase, 'verified', 'the backstop sits before the try')
        self.assertIn(1, block.pending_segments)
        refused = [line for line in self.lines if line.startswith(packed_verifier.EXTENT_CAP_REFUSED_MARKER)]
        self.assertEqual(refused, ['[PINDIAG] packed extent cap refused round=1 segment=1 start=20218 prefix=7 limit=6'])
        self.assertEqual(block.extent_counts['cap_refused'], 1)
        executed = len(self.ttnn.executed)
        block.commit_user(1, 6)
        self.assertEqual(len(self.ttnn.executed), executed + 1, 'the capped prefix commits through its own trace')
        # every other segment is uncapped: the full ticket commits
        for segment in (0, 2, 3):
            block.commit_user(segment, 16)
        self.assertEqual(block.phase, 'idle')
        self.assertEqual(block.extent_counts['cap_events'], 1)

    def test_admits_is_the_extent_range_and_verify_refuses_outside_it_before_anything_is_staged(self):
        block = self.build()
        self.assertEqual([block.admits(position) for position in (127, 128, 4095, C - 16, C - 15, -1, 128.0, None)],
                         [False, True, True, True, False, False, False, False])
        copies, executed = len(self.ttnn.host_copies), len(self.ttnn.executed)
        for position in (127, C - 15):
            with self.subTest(position=position), \
                    self.assertRaisesRegex(ValueError, 'outside the extent path: 128 <= start and start \\+ 16 <= 131328'):
                block.verify(self.round((position, 20000, 30000, 40000)))
            self.assertEqual(block.phase, 'idle')
        self.assertEqual((len(self.ttnn.host_copies), len(self.ttnn.executed)), (copies, executed))

    def test_a_padded_round_stages_idle_segments_at_zero_and_thirty_two_on_the_zero_table(self):
        block = self.build(padded_min_users=2)
        self.assertEqual(block.idle_inputs((0, 2))[1][1], 0)
        self.assertEqual(block.idle_inputs((0, 2))[3][1], 32)
        predictions, metrics = block.verify(self.round((70000, None, 90000, None)))
        self.assertEqual(metrics['idle'], [1, 3])
        readers = self.readers(block)
        for segment, start in ((1, 0), (3, 32)):
            self.assertEqual(readers[segment].positions.value.tolist(), [start] + [0] * 7)
            self.assertEqual(readers[segment].cur_pos[0].value.tolist(), [255, 255])
            self.assertFalse(bool(readers[segment].metadata[0][1].value.any()), 'the zero table: page 0')
        positions, pages = block.fixture.positions.value, block.fixture.pages.value
        self.assertEqual(positions[16:32].tolist(), list(range(0, 16)), 'page 0, tile row 0')
        self.assertEqual(positions[48:64].tolist(), list(range(32, 48)), 'page 0, tile row 1')
        self.assertFalse(bool(pages[16:32].any() or pages[48:64].any()))
        self.assertIsNone(verify_trace_t2.kv_conflict(verify_trace_t2.block_users(positions, pages, 16, 4)))
        self.assertIn('[PINDIAG] packed extent round round=1 live=2 families=[0:70144,2:90112] idle=[1,3] capped=[]',
                      self.lines)
        # the idle segments decided at prefix 0 inside verify; the live ones commit as always
        self.assertEqual(block.pending_segments, {0, 2})
        self.finish(block)
        self.assertEqual(block.phase, 'idle')

    def test_the_page_zero_rule_checks_every_page_the_extent_reads(self):
        # start 20000: the round reads and writes pages [0, 313); its extent reads [0, 316)
        table = torch.full((1, WIDTH), 9, dtype=torch.int32)
        table[0, 314] = 0
        self.assertIsNone(page_zero_index(table, 20000, 16))
        self.assertEqual(page_zero_index(table, 20000, 16, extent=True), 314)
        # a crossing ticket still writes past E: the range is the larger of the two
        crossing = torch.full((1, WIDTH), 9, dtype=torch.int32)
        crossing[0, 316] = 0
        self.assertEqual(page_zero_index(crossing, 20218, 16, extent=True), 316, 'a crossing ticket writes past E')
        self.assertIsNone(page_zero_index(crossing, 20000, 16, extent=True))
        with self.assertRaisesRegex(ValueError, r'cannot map positions \[0, 20224\)'):
            page_zero_index(table[:, :300], 20000, 16, extent=True)
        block = self.build(padded_min_users=2)
        with self.assertRaisesRegex(ValueError, 'page0 in a live table: segment 0 position 20000 page_index 314'):
            block.verify(self.round((20000, None, 30000, None), pages={0: table}))
        self.assertEqual(block.phase, 'idle')

    def test_the_prestage_diff_writes_cur_pos_only_when_the_family_changes(self):
        self.env(QWEN_FAST_PRESTAGE='1')
        block = self.build()
        first = (20218, 30000, 40000, 50000)
        block.verify(self.round(first))
        self.finish(block)
        readers = self.readers(block)
        cur_pos = [reader.cur_pos[0] for reader in readers]
        words = [reader.positions for reader in readers]
        tables = [reader.metadata[0][1] for reader in readers]
        # the window's pre-stage at this round's frontiers, then a verify one round further on: the
        # verify-time diff against it writes what the new frontiers change and nothing else
        users = [((0,) * 16, start, torch.full((1, WIDTH), 7 + segment, dtype=torch.int32))
                 for segment, start in enumerate(first)]
        block.prestaged.prestage(users)
        marked = len(self.ttnn.host_copies)
        second = (20226, 30005, 40005, 50005)      # segment 0 crosses 20224; the others stay in family
        block.verify(self.round(second))
        written = [destination for host, destination in self.ttnn.host_copies[marked:]]
        self.assertEqual(block.prestaged.last['path'], 'diff')
        self.assertEqual([any(value is target for target in written) for value in cur_pos], [True, False, False, False])
        self.assertEqual([any(value is target for target in written) for value in words], [True] * 4)
        self.assertEqual([any(value is target for target in written) for value in tables], [False] * 4)
        self.assertEqual(readers[0].cur_pos[0].value.tolist(), [20479, 20479])
        self.assertEqual([reader.start for reader in readers], list(second))
        self.finish(block)

    def test_verify_prestage_and_padded_probe_need_no_edit(self):
        """Both stage through packed_values (design 2.5): neither names a family, a word or a table."""
        for name in ('verify_prestage.py', 'padded_probe.py'):
            source = (HERE / name).read_text(encoding='utf-8')
            with self.subTest(module=name):
                self.assertNotIn('validate_ticket', source)
                self.assertNotIn('replay_capacity', source)
                self.assertNotIn('words[0]', source)
                self.assertNotIn('extent', source)
        prestage = (HERE / 'verify_prestage.py').read_text(encoding='utf-8')
        self.assertEqual(prestage.count('packed_values('), 3)


class ExtentAuditTests(ExtentFixture):
    STARTS = (5000, 20218, 70000, 131100)

    def setUp(self):
        super().setUp()
        self.env(QWEN_FAST_EXTENT_AUDIT='1')

    def audit_lines(self):
        return [line for line in self.lines if line.startswith(packed_verifier.EXTENT_AUDIT_MARKER)]

    def corrupt_after_the_replay(self, block, damage):
        replayed = self.ttnn.on_trace

        def hook(trace):
            replayed(trace)
            if trace == block.trace:
                damage(self.readers(block))

        self.ttnn.on_trace = hook

    def test_a_clean_round_logs_every_segment_ok_and_rotates_the_mask_and_table_reads(self):
        block = self.build()
        self.assertTrue(block.extent_audit)
        for number in (1, 2):
            block.verify(self.round(self.STARTS))
            self.finish(block)
            (line,) = [line for line in self.audit_lines() if ' round=%d ' % number in line]
            self.assertRegex(line, r'^\[EXTENT-AUDIT\] round=%d segments=4 words_ok=4 cur_pos_ok=4 mask_ok=1 '
                                   r'tables_ok=1 rotated=%d ms=[0-9]+[.][0-9]{2}$' % (number, number - 1))
        self.assertEqual(block.extent_counts['audit_mismatches'], 0)

    def check_mismatch(self, damage, what, restaged):
        block = self.build()
        self.corrupt_after_the_replay(block, damage)
        block.verify(self.round(self.STARTS))
        mismatch = [line for line in self.audit_lines() if line.startswith(packed_verifier.EXTENT_AUDIT_MISMATCH_MARKER)]
        self.assertEqual(mismatch, ['[EXTENT-AUDIT] MISMATCH round=1 at=%s' % what])
        summary = [line for line in self.audit_lines() if not line.startswith(packed_verifier.EXTENT_AUDIT_MISMATCH_MARKER)]
        self.assertEqual(len(summary), 1, 'the round still logs its audit line')
        # restaged in full: the device holds what was meant again
        restaged(self.readers(block))
        self.assertEqual(block.extent_counts['audit_mismatches'], 1)
        self.finish(block)

    def test_an_absolute_word_is_a_mismatch_and_the_round_is_restaged(self):
        def damage(readers):
            readers[2].positions.value = torch.tensor([70000] + [0] * 7, dtype=torch.int32)

        self.check_mismatch(damage, 'word:2',
                            lambda readers: self.assertEqual(readers[2].positions.value.tolist(), [70000 & 255] + [0] * 7))

    def test_a_stale_cur_pos_is_a_mismatch_and_the_round_is_restaged(self):
        def damage(readers):
            readers[1].cur_pos[0].value = torch.tensor([19967, 19967], dtype=torch.int32)

        self.check_mismatch(damage, 'cur_pos:1',
                            lambda readers: self.assertEqual(readers[1].cur_pos[0].value.tolist(), [20223, 20223]))

    def test_a_corrupted_mask_tile_of_the_rotated_segment_is_a_mismatch(self):
        def damage(readers):
            mask = readers[0].metadata[0][2]
            value = mask.value.clone()
            value[1, 0, 95, 255] = 0.0 if bool(torch.isinf(value[1, 0, 95, 255])) else float('-inf')
            mask.value = value

        self.check_mismatch(damage, 'mask:0', lambda readers: None)

    def test_a_table_the_device_does_not_hold_is_a_mismatch(self):
        def damage(readers):
            readers[0].metadata[0][1].value = torch.zeros(2, WIDTH, dtype=torch.int32)

        self.check_mismatch(damage, 'table:0', lambda readers: self.assertTrue(bool(readers[0].metadata[0][1].value.all())))

    def test_the_audit_is_off_by_default_and_refuses_a_bad_value(self):
        self.env(QWEN_FAST_EXTENT_AUDIT='0')
        block = self.build()
        self.assertFalse(block.extent_audit)
        block.close()
        self.env(QWEN_FAST_EXTENT_AUDIT='yes')
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_EXTENT_AUDIT must be 0 or 1'):
            self.build()
        self.assertFalse(self.pool.storage.taken, 'refused before the storage is taken')


class CapturePositionTests(ExtentFixture):
    def test_the_extent_block_captures_at_the_knobs_position(self):
        block = self.build(capture_position=16384)
        self.assertEqual((block.capture_position, block.replay_capacity), (16384, C))
        for reader in self.readers(block):
            self.assertEqual(reader.start, 16384)
            self.assertEqual(reader.cur_pos[0].value.tolist(), [16639, 16639])
            self.assertEqual(reader.positions.value.tolist(), [0] * 8)

    def test_an_extent_capture_position_the_path_does_not_admit_is_refused_before_the_storage_is_taken(self):
        # below the floor: the extent path's own refusal; past the table: the block's range check, first
        for position, message in ((100, 'captures at a start the extent path admits'),
                                  (127, 'captures at a start the extent path admits'),
                                  (C - 15, 'Capture position must leave one segment within the page capacity')):
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, message):
                self.build(capture_position=position)
            self.assertFalse(self.pool.storage.taken)

    def test_flag_off_the_knob_captures_the_family_block_in_family_16640(self):
        """G3b's flag-off arm: the family block captured at 16384 builds 16640-family readers over the
        pool's 16640 tables - the family the pinned validate_ticket admits."""
        ttnn = base.FakeTTNN()
        base.FakeModelBatch.ttnn, base.FakeModelBatch.instances = ttnn, []
        width = 16640 // 64
        pool = base.pool(ttnn, base.helpers(ttnn), users=2, page_width=width,
                         packed={(2, 16): base.packed_tables(ttnn, users=2, families=(16384, 16640))})
        shared = pool.helpers
        with patch('packed_verifier.ModelBatch', base.FakeModelBatch), patch('attention_replay.prepare',
                                                                             return_value='mask-program'), \
                patch.dict(os.environ, {'QWEN_FAST_REPLAY_GROUP_ROWS': '4', 'QWEN_FAST_SDPA_MODES': ''}):
            block = PackedVerifierEngine(ttnn, self.model, shared, 'sampler', pool=pool, shared_weights=self.weights,
                                         shape=m1_shape(width), feature_taps=base.TAPS, capture_position=16384)
            self.assertFalse(block.extent)
            self.assertEqual((block.replay_capacity, block.describe()['attention']['family']), (16640, 16640))
            self.assertEqual([reader.capacity for reader in block.fixture.replay_reader.readers], [16640, 16640])
            self.assertIsNone(block.deadline)
            block.close()
            with self.assertRaisesRegex(ValueError, 'Capture position must leave one segment within the page capacity'):
                PackedVerifierEngine(ttnn, self.model, shared, 'sampler', pool=pool, shared_weights=self.weights,
                                     shape=m1_shape(width), feature_taps=base.TAPS, capture_position=16640 - 15)


class ReplayDeadlineTests(ExtentFixture):
    def test_the_deadline_is_read_strictly(self):
        read = packed_verifier.replay_deadline_seconds
        self.assertEqual((read({}), read({'QWEN_FAST_REPLAY_DEADLINE_S': '5'}),
                          read({'QWEN_FAST_REPLAY_DEADLINE_S': '0.25'})), (30.0, 5.0, 0.25))
        for value in ('0', '0.0', '-1', '1e3', 'abc', '', ' 5', 'inf'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'QWEN_FAST_REPLAY_DEADLINE_S must be'):
                read({'QWEN_FAST_REPLAY_DEADLINE_S': value})

    def test_a_replay_past_the_deadline_logs_its_families_and_exits_seventy(self):
        self.env(QWEN_FAST_REPLAY_DEADLINE_S='0.05')
        block = self.build()
        self.assertEqual(block.deadline.seconds, 0.05)
        released, exits = threading.Event(), []

        def exit_(code):
            exits.append(code)
            released.set()

        block.deadline.exit = exit_
        replayed = self.ttnn.on_trace

        def blocking(trace):
            replayed(trace)
            if trace == block.trace:
                # a hung card: the replay returns only once the watchdog has fired
                self.assertTrue(released.wait(10), 'the deadline never fired')

        self.ttnn.on_trace = blocking
        block.verify(self.round((5000, 20218, 70000, 131100)))
        self.assertEqual(exits, [packed_verifier.REPLAY_DEADLINE_EXIT_CODE])
        fired = [line for line in self.lines if line.startswith(packed_verifier.REPLAY_DEADLINE_MARKER)]
        self.assertEqual(fired, ['[PINDIAG] replay deadline exceeded round=1 trace=verify segments=3,2,1,0 '
                                 'families=131328,70144,20224,5120 seconds=0.05'])
        stack = [line for line in self.lines if line.startswith('[PINDIAG] replay deadline traceback')]
        self.assertEqual(len(stack), 1)
        self.assertIn('in blocking', stack[0], "the stack of the thread the replay holds")
        self.ttnn.on_trace = replayed
        self.finish(block)

    def test_a_replay_inside_the_deadline_never_fires_and_commits_are_armed_too(self):
        block = self.build()
        armed = []
        original = block.deadline.armed

        def spy(what):
            armed.append(what)
            return original(what)

        block.deadline.armed = spy
        block.verify(self.round((5000, 20218, 70000, 131100)))
        block.commit_user(1, 3)
        self.assertEqual(armed, ['round=1 trace=verify segments=3,2,1,0 families=131328,70144,20224,5120',
                                 'round=1 trace=commit segments=1 families=20224 prefix=3'])
        self.assertIsNone(block.deadline.fired)
        self.assertIsNone(block.deadline.current)
        self.finish(block)
        block.close()
        self.assertTrue(block.deadline.closed)

    def test_nested_arming_is_one_deadline_and_a_closed_one_cannot_be_armed(self):
        deadline = packed_verifier.ReplayDeadline(30, exit=Mock())
        with deadline.armed('outer'):
            first = deadline.current
            with deadline.armed('inner'):
                self.assertIs(deadline.current, first)
            self.assertIs(deadline.current, first)
        self.assertIsNone(deadline.current)
        deadline.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            with deadline.armed('late'):
                pass
        for seconds in (0, -1, float('inf'), '30', None):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                packed_verifier.ReplayDeadline(seconds)


class BindingAndDescribeTests(ExtentFixture):
    def test_validate_bindings_checks_the_cur_pos_words_and_each_readers_word_and_masks(self):
        block = self.build()
        block.validate_bindings()
        self.assertEqual(len(block.reader_addresses), 4)
        self.assertEqual([len(entry) for entry in block.reader_addresses], [2] * 4, 'the word and one narrow mask')
        cur_pos = block.extent_cur_pos[2][0]
        kept = cur_pos.shards
        cur_pos.shards = [base.FakeShard(cur_pos, 0xdead0), base.FakeShard(cur_pos, 0xdead1)]
        with self.assertRaisesRegex(ValueError, 'A pooled extent cur_pos word moved'):
            block.validate_bindings()
        cur_pos.shards = kept
        mask = self.readers(block)[3].metadata[0][2]
        kept = mask.shards
        mask.shards = [base.FakeShard(mask, 0xbeef0), base.FakeShard(mask, 0xbeef1)]
        with self.assertRaisesRegex(ValueError, "An extent reader's positions word or narrow mask moved"):
            block.validate_bindings()
        mask.shards = kept
        block.validate_bindings()

    def test_describe_names_the_extent_reader_and_its_counts(self):
        block = self.build()
        attention = block.describe()['attention']
        self.assertEqual((attention['reader'], attention['capacity'], attention['mask'], attention['flags']),
                         ('per-user extent replay', C, 'narrow', ['0x27'] * 4))
        self.assertEqual((attention['replay_group_rows'], attention['bundles_per_user'], attention['audit'],
                          attention['deadline_s']), (8, [1] * 4, False, 30.0))
        self.assertEqual(len(attention['cur_pos']), 4)
        self.assertEqual(attention['rounds'], 0)
        self.assertNotIn('family', attention)

    def test_construction_refusals_come_before_the_storage_is_taken(self):
        for environ, message in (({'QWEN_FAST_REPLAY_GROUP_ROWS': '4'}, 'eight-row replay groups only'),
                                 ({'QWEN_FAST_REPLAY_DEADLINE_S': 'soon'}, 'QWEN_FAST_REPLAY_DEADLINE_S must be')):
            with self.subTest(environ=environ), patch.dict(os.environ, environ), self.assertRaisesRegex(ValueError, message):
                self.build()
            self.assertFalse(self.pool.storage.taken)

    def test_close_hands_the_storage_back_and_stops_the_watchdog(self):
        block = self.build()
        block.verify(self.round((5000, 20218, 70000, 131100)))
        self.finish(block)
        self.assertIsNotNone(block.deadline.thread)
        block.close()
        self.assertFalse(self.pool.storage.taken)
        self.assertTrue(block.deadline.closed)
        block.deadline.thread.join(5)
        self.assertFalse(block.deadline.thread.is_alive())


class FlagOffBlockTests(base.BlockFixture):
    """A pool without extent storage: the block is the family one, with every S2 attribute inert."""

    def test_the_family_block_is_untouched_by_the_extent_path(self):
        block = self.build()
        self.assertFalse(block.extent)
        self.assertIsNone(block.deadline)
        self.assertFalse(block.extent_audit)
        self.assertIsNone(block.accept_limit(4100))
        self.assertEqual([block.admits(position) for position in (4095, 4096, 4336, 4337)], [False, True, True, False])
        self.assertEqual(block.replay_deadline('verify', (0, 1)).__class__.__name__, 'nullcontext')
        self.assertNotIn('capacity', block.describe()['attention'])
        self.assertEqual(block.describe()['attention']['reader'], 'per-user bundled replay')
        # a pool whose extent_replay is anything but True is not an extent pool
        self.pool.extent_replay = Mock()
        block.close()
        other = self.build()
        self.assertFalse(other.extent)


class ModelBatchExtentTests(unittest.TestCase):
    """model_batch's own extent branch, on the real constructor up to its reader."""

    def pack(self, starts=(20000, 60000)):
        return [dict(start=start, rows=16, pages=torch.full((1, WIDTH), index + 1, dtype=torch.int32), prefix=0,
                     checkpoints=['c'] * 48, slots=[['s%d' % index] * 5] * 48) for index, start in enumerate(starts)]

    OPTIONS = dict(serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True, hoist_row_layout=True,
                   device_loop_gdn=True, compact_prologue=True, batch_conv=True, packed_checkpoints=True,
                   retain_records=True, ordered_cache=True, norm_batch=True, commit_only_gdn=True,
                   attention_replay=True, replay_group_rows=8, short_context=False)

    def test_the_extent_fixture_builds_the_extent_reader_over_the_lent_storage_at_each_segments_start(self):
        from model_batch import ModelBatch
        import test_model_batch

        ttnn = test_model_batch.FakeTTNN()
        ttnn.TILE_LAYOUT = 'tile'
        model = SimpleNamespace(mesh_device='mesh', layers=[SimpleNamespace(is_full_attention=False)] * 64,
                                args=SimpleNamespace(rope_head_dim=64, max_seq_len=C, rope_theta=1e6))
        built = []

        class Reader:
            def __init__(self, *args, **options):
                built.append((args, options))
                self.borrowed = ['lent-table', 'lent-cur-pos']
                self.starts = options['starts']
                self.stage = Mock()

        class Stop(Exception):
            pass

        with patch.dict(sys.modules, {'ttnn': ttnn, 'models.demos.blackhole.qwen36.tt.attention.rope_tp':
                                      SimpleNamespace(rot_mats_decode=test_model_batch.fake_rope(ttnn))}), \
                patch.dict(os.environ, {'TT_METAL_HOME': '/opt/tt-metal'}), \
                patch('ordered_cache.load_kernels', return_value='kernels'), \
                patch('gdn_records.RetainedGDNBlock', return_value=SimpleNamespace()), \
                patch('extent_attention_replay.PackedExtentReplayReader', Reader), \
                patch('model_batch.chained_writer_options', side_effect=Stop):
            with self.assertRaises(Stop):
                ModelBatch(model, [1] * 32, C - 256, torch.zeros(1, WIDTH, dtype=torch.int32), [None] * 48,
                           [None] * 48, 32, pack=self.pack(), packed_extent=[['pairs0'], ['pairs1']], **self.OPTIONS)
        ((args, options),) = built
        self.assertEqual(args[:4], (ttnn, 'mesh', ((0, 16), (16, 32)), WIDTH))
        self.assertEqual([int(table[0, 0]) for table in args[4]], [1, 2])
        self.assertEqual(options, dict(storage=[['pairs0'], ['pairs1']], max_group_rows=8, starts=(20000, 60000)))

    def test_packed_extent_needs_a_long_context_packed_replay_fixture_and_no_family_tables(self):
        from model_batch import ModelBatch

        options = {key: value for key, value in self.OPTIONS.items() if key not in ('attention_replay', 'short_context')}
        with patch.dict(sys.modules, {'ttnn': SimpleNamespace()}):
            for name, extra in (('no pack', dict(attention_replay=True, pack=None)),
                                ('no replay', dict(attention_replay=False, pack=self.pack(), replay_group_rows=4)),
                                ('family tables too', dict(attention_replay=True, pack=self.pack(),
                                                           packed_replay_pages=[['t'], ['t']]))):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Packed extent storage needs|Packed replay page tables need'):
                    ModelBatch(SimpleNamespace(), [1] * 32, C - 256, torch.zeros(1, WIDTH, dtype=torch.int32),
                               [None] * 48, [None] * 48, 32, packed_extent=[['p'], ['p']], **dict(options, **extra))

    def test_flag_off_the_packed_fixture_is_todays_call(self):
        import inspect

        import model_batch

        source = inspect.getsource(model_batch.ModelBatch.__init__)
        self.assertEqual(source.count('PackedExtentReplayReader('), 1)
        self.assertIn('if self.pack is not None and packed_extent is not None:', source)
        self.assertLess(source.index('if self.pack is not None and packed_extent is not None:'),
                        source.index('elif self.pack is not None:'))
        self.assertIn('self.replay_capacity = packed_replay_family(start, self.pack, short_context=short_context)', source)


if __name__ == '__main__':
    unittest.main()
