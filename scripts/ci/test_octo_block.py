"""The REAL PackedVerifierEngine at EIGHT users x EIGHT rows (packed_shapes.octo_shape), on test_packed_extent_block's fake two-chip device.

What this proves, and what it cannot. The block's host side is shape-generic, and here it runs at the octo geometry: the segments bound to pool slots 0..7 by
carry identity, 8 rows staged per user, one 64-row block, 8 x 8 commit traces, the taps' row offsets, padded rounds of 6 and 7 live users on page 0's two
tile rows, the shared carries beside a real M3 block. It runs with ONE thing patched in: the extent reader's qualification (extent_attention_replay.
EXTENT_BUNDLE_ENTRIES / EXTENT_FLAGS), which refuses an 8-row segment today - that refusal is the 'attention-8row' device piece, and the last tests here pin it
as it stands, so the day K64j is qualified at one group per bundle those tests fail and the piece's text must change with them. Nothing here is a kernel
result: the fake device stands in for every trace."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

import torch

import extent_attention_replay
import packed_verifier
from packed_verifier import PackedVerifierEngine, m3_shape
from packed_shapes import octo_shape, segment_rows
import publication_warm
from serving_buffer_pool import PackedExtentStorage, extent_bundle_batches
import serving_octo as octo
import test_packed_extent_block as ext
from test_packed_extent_block import ExtentFixture, WIDTH, extent_pool, owner
import test_packed_verifier as base
import verify_trace_t2

PAGE_ZERO_STARTS = (0, 32)


@contextmanager
def b1_qualification():
    """The extent reader as K64j would have to be qualified for one eight-row group per bundle: the bundle count and the flag set it would then run (0x25: no KV
    share, a bundle of one batch has nothing to share). The device question, patched in for the host path only."""
    with patch.object(extent_attention_replay, 'EXTENT_BUNDLE_ENTRIES', 1), patch.object(extent_attention_replay, 'EXTENT_FLAGS', 0x25):
        yield


class OctoFixture(ExtentFixture):
    """An eight-slot pool lending extent storage for a (4, 16) M3 block and an (8, 8) octo block, as ServingBufferPool would for the shapes it was asked."""

    USERS = 8

    def setUp(self):
        super().setUp()

        def integers(shape):
            return self.ttnn.allocate(shape, 'int32', 'row_major', torch.zeros(shape, dtype=torch.int32))

        self.pool = extent_pool(self.ttnn, self.helpers, users=8)
        self.storages = {
            (4, 16): PackedExtentStorage(4, 16, [[integers((2, WIDTH))] for user in range(4)], [[integers((2,))] for user in range(4)]),
            (8, 8): PackedExtentStorage(8, 8, [[integers((1, WIDTH))] for user in range(8)], [[integers((1,))] for user in range(8)])}
        self.pool.packed_extent = lambda users, rows: self.storages[(users, rows)]
        self.blocks = []
        self.ttnn.on_trace = self.replayed

    def replayed(self, trace):
        # a verify trace refreshes its own block's narrow masks from the staged words, in-trace
        for block in self.blocks:
            if trace == block.trace and block.fixture is not None:
                block.fixture.replay_reader.refresh()

    def block(self, shape, **options):
        block = PackedVerifierEngine(self.ttnn, self.model, self.helpers, 'sampler', pool=self.pool, shared_weights=self.weights, shape=shape,
                                     feature_taps=base.TAPS, **options)
        self.addCleanup(lambda: block.deadline.close() if block.deadline is not None else None)
        self.blocks.append(block)
        return block

    def octo_block(self, **options):
        with b1_qualification():
            return self.block(octo_shape(WIDTH), pool_slots=tuple(range(8)), **options)

    def m3_block(self, **options):
        return self.block(m3_shape(WIDTH), pool_slots=(0, 1, 2, 3), **options)

    def octo_round(self, starts, pages=None):
        """One entry per segment with a start (None: idle), eight tokens each, presented last first."""
        pages = pages or {}
        entries = []
        for segment, start in enumerate(starts):
            if start is None:
                continue
            request = owner(self, segment, start, 7 + segment)
            if segment in pages:
                request.engine.pages = pages[segment]
            entries.append(base.entry(request, range(10 + segment, 18 + segment)))
        return entries[::-1]

    def finish(self, block, prefix=1):
        for segment in sorted(block.pending_segments):
            block.commit_user(segment, prefix if segment not in block.idle_segments else 0)


class OctoBlockTests(OctoFixture):
    def test_the_block_builds_as_eight_extent_segments_of_eight_rows_over_pool_slots_zero_to_seven(self):
        block = self.octo_block()
        self.assertEqual((block.users, block.rows_per_user, block.block_rows), (8, 8, 64))
        self.assertTrue(block.extent)
        self.assertEqual(block.pool_slots, tuple(range(8)))
        self.assertEqual(len(self.readers(block)), 8)
        self.assertEqual({reader.rows for reader in self.readers(block)}, {8})
        self.assertEqual([tuple(tensor.shape) for tensor in block.taps], [(1, 1, 64, 5120)] * 5)
        self.assertEqual(len(block.checkpoints), 8, 'one checkpoint set per user')
        self.assertEqual([sorted(commits) for commits in block.commits], [list(range(1, 9))] * 8, 'eight nonzero prefixes per user')
        self.assertEqual(sum(len(commits) for commits in block.commits), 64, 'the 64 GDN commit traces of the design')
        self.assertTrue(self.storages[(8, 8)].taken)
        self.assertEqual([tuple(table.shape) for table in self.storages[(8, 8)].tables[0]], [(1, WIDTH)], 'one bundle of one group per 8-row segment')

    def readers(self, block):
        return block.fixture.replay_reader.readers

    def test_an_eight_live_round_stages_eight_rows_per_segment_and_returns_eight_predictions_each(self):
        block = self.octo_block(padded_min_users=6)
        starts = (1500, 20000, 60000, 131312 - 8, 5000, 70000, 90000, 40000)
        predictions, metrics = block.verify(self.octo_round(starts))
        self.assertEqual(metrics['segments'], tuple(range(7, -1, -1)), 'presented last first: the segments follow the entries')
        self.assertEqual([len(item) for item in predictions], [8] * 8)
        fixture = block.fixture
        for segment, start in enumerate(starts):
            rows = slice(8 * segment, 8 * segment + 8)
            self.assertEqual(fixture.tokens.value[rows, 0].tolist(), list(range(10 + segment, 18 + segment)))
            self.assertEqual(fixture.positions.value[rows].tolist(), list(range(start, start + 8)))
            self.assertTrue(bool((fixture.pages.value[rows] == 7 + segment).all()))
            reader = self.readers(block)[segment]
            self.assertEqual(reader.start, start)
            self.assertEqual(reader.positions.value.tolist(), [start & 255] + [0] * 7)
        self.assertEqual(predictions[-1], list(range(1000, 1008)), 'segment 0 is the first eight rows of the block')
        self.assertEqual(block.round_starts, dict(enumerate(starts)))
        self.assertIsNone(verify_trace_t2.kv_conflict(verify_trace_t2.block_users(fixture.positions.value, fixture.pages.value, 8, 8)))
        self.assertEqual(block.pending_segments, set(range(8)))
        self.assertEqual(len([line for line in self.lines if line.startswith(packed_verifier.EXTENT_ROUND_MARKER)]), 1)

    def test_each_segments_commit_takes_a_prefix_up_to_eight_through_its_own_trace_and_the_taps_are_the_segments_eight_rows(self):
        block = self.octo_block()
        block.verify(self.octo_round((5000,) * 8))
        for segment in (3, 0):
            taps = block.features(segment)
            self.assertEqual((taps.row_offset, taps.rows), (8 * segment, 8))
            self.assertEqual(segment_rows(block.shape, segment), (8 * segment, 8 * segment + 8))
        executed = len(self.ttnn.executed)
        block.commit_user(3, 8)
        block.commit_user(5, 1)
        self.assertEqual(self.ttnn.executed[executed:], [block.commits[3][8], block.commits[5][1]], 'each commit replays its own (segment, prefix) trace')
        with self.assertRaises(ValueError):
            block.commit_user(0, 9)
        for segment in sorted(block.pending_segments):
            block.commit_user(segment, 4)
        self.assertEqual(block.phase, 'idle')

    def test_six_and_seven_live_are_padded_rounds_with_the_idle_segments_on_page_zeros_two_tile_rows(self):
        block = self.octo_block(padded_min_users=6)
        self.assertEqual((block.padded_min_users, block.MAX_IDLE_SEGMENTS), (6, 2))
        self.assertEqual([block.pads(count) for count in range(0, 10)], [False] * 6 + [True, True, False, False])
        idle = block.idle_inputs((0, 1, 2, 3, 4, 5))
        self.assertEqual({segment: (tokens, start) for segment, (tokens, start, table) in idle.items()}, {6: ((1,) * 8, 0), 7: ((1,) * 8, 32)})
        predictions, metrics = block.verify(self.octo_round((70000, 20000, 30000, 40000, 50000, 60000, None, None)))
        self.assertEqual(metrics['idle'], [6, 7])
        positions, pages = block.fixture.positions.value, block.fixture.pages.value
        self.assertEqual(positions[48:56].tolist(), list(range(0, 8)), 'page 0, tile row 0')
        self.assertEqual(positions[56:64].tolist(), list(range(32, 40)), 'page 0, tile row 1')
        self.assertFalse(bool(pages[48:64].any()))
        self.assertIsNone(verify_trace_t2.kv_conflict(verify_trace_t2.block_users(positions, pages, 8, 8)))
        self.assertTrue(any('packed padded round live=6' in line and 'idle=6,7' in line for line in self.lines))
        self.assertEqual(block.pending_segments, {0, 1, 2, 3, 4, 5})
        self.finish(block, prefix=8)
        self.assertEqual((block.phase, block.rounds), ('idle', 1))
        # seven live: one idle segment, at start 0
        block.verify(self.octo_round((70000, 20000, 30000, 40000, 50000, 60000, 80000, None)))
        self.assertEqual(block.idle_segments, frozenset({7}))
        self.finish(block)
        self.assertEqual(block.phase, 'idle')

    def test_five_live_is_refused_by_the_block_itself_because_three_idle_segments_would_share_a_tile_row(self):
        block = self.octo_block(padded_min_users=6)
        with self.assertRaisesRegex(ValueError, 'At most 2 idle segments: page 0 holds two 32-row tile rows'):
            block.idle_inputs((0, 1, 2, 3, 4))
        copies = len(self.ttnn.host_copies)
        with self.assertRaisesRegex(ValueError, 'The padded packed block serves 6 to 8 users; 5 entries given'):
            block.verify(self.octo_round((70000, 20000, 30000, 40000, 50000, None, None, None)))
        self.assertEqual((block.phase, len(self.ttnn.host_copies)), ('idle', copies), 'nothing was staged')

    def test_a_padded_minimum_the_block_cannot_hold_is_refused_at_construction(self):
        capacity = PackedVerifierEngine.MAX_IDLE_SEGMENTS
        for minimum in (1, 5):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, r'padded_min_users must be an integer in \[6, 8\) for a 8-user block'):
                self.octo_block(padded_min_users=minimum)
        # the same rule at the M3 shape is what refuses a lone user (the lone-user padded round, QWEN_FAST_SOLO_PACKED)
        with self.assertRaisesRegex(ValueError, r'padded_min_users must be an integer in \[2, 4\) for a 4-user block'):
            self.m3_block(padded_min_users=1)
        self.assertEqual(octo.idle_capacity(), capacity)
        self.assertEqual(self.octo_block(padded_min_users=7).padded_min_users, 7)

    def test_the_octo_block_and_an_m3_block_share_the_carries_by_identity(self):
        m3 = self.m3_block(padded_min_users=2)
        block = self.octo_block(padded_min_users=6)
        first, second = owner(self, 2, 5000), owner(self, 5, 5000)
        self.assertEqual((m3.segment_of(first.engine), block.segment_of(first.engine)), (2, 2))
        self.assertEqual(block.segment_of(second.engine), 5)
        with self.assertRaisesRegex(ValueError, 'borrows no carry'):
            m3.segment_of(second.engine)
        self.assertTrue(self.storages[(4, 16)].taken and self.storages[(8, 8)].taken, 'two storages, one per block')

    def test_the_blocks_alternate_on_the_shared_carries_without_either_refusing_the_others_round(self):
        m3 = self.m3_block(padded_min_users=2)
        block = self.octo_block(padded_min_users=6)
        for which, live in (('octo', 8), ('m3', 4), ('octo', 6), ('m3', 3), ('octo', 7), ('m3', 4)):
            if which == 'octo':
                entries = self.octo_round(tuple(5000 + 100 * segment if segment < live else None for segment in range(8)))
                running = block
            else:
                entries = [base.entry(owner(self, segment, 5000 + 7 * segment, 7 + segment), range(10, 26)) for segment in range(live)]
                running = m3
            predictions, metrics = running.verify(entries)
            self.assertEqual(len(predictions), live)
            self.finish(running, prefix=3)
            self.assertEqual((m3.phase, block.phase), ('idle', 'idle'))
        self.assertEqual((block.rounds, m3.rounds), (3, 3))

    def test_a_block_that_is_verified_refuses_the_others_round_until_it_commits(self):
        m3 = self.m3_block(padded_min_users=2)
        block = self.octo_block(padded_min_users=6)
        block.verify(self.octo_round((5000,) * 8))
        self.assertEqual(block.phase, 'verified')
        for segment in sorted(block.pending_segments):
            block.commit_user(segment, 2)
        m3.verify([base.entry(owner(self, segment, 5000, 7 + segment), range(10, 26)) for segment in range(4)])
        for segment in sorted(m3.pending_segments):
            m3.commit_user(segment, 2)
        self.assertEqual((m3.phase, block.phase), ('idle', 'idle'))

    def test_the_boundary_cap_is_the_extent_of_the_segment_at_eight_rows(self):
        block = self.octo_block()
        capped = 20224 - 3                       # rows 3..7 lie at or past E = 20224
        self.assertEqual(block.accept_limit(capped), 3)
        self.assertEqual(block.accept_limit(5000), 8)
        block.verify(self.octo_round((5000, capped, 70000, 131072, 90000, 6000, 7000, 8000)))
        self.assertIn('capped=[1:3]', [line for line in self.lines if line.startswith(packed_verifier.EXTENT_ROUND_MARKER)][0])
        with self.assertRaisesRegex(ValueError, 'commits at most 3 rows; prefix 4 refused'):
            block.commit_user(1, 4)
        block.commit_user(1, 3)
        self.finish(block)

    def test_the_publication_warm_plan_of_the_octo_block_adds_thirty_two_shapes_no_m3_warm_ran(self):
        from types import SimpleNamespace

        pool = SimpleNamespace(bucket_rows=(1, 2, 4))
        packed = {}
        for name, shape in (('m3', m3_shape(WIDTH)), ('octo', octo_shape(WIDTH))):
            block = SimpleNamespace(block_rows=shape.block_rows, shape=shape, users=shape.users, rows_per_user=shape.rows_per_user)
            plan = publication_warm.plan(block, pool)
            packed[name] = {(item.rows, item.offset, item.prefix) for item in plan if item.path == publication_warm.PACKED}
            self.assertEqual(len(plan), 71, name)
        self.assertEqual(len(packed['octo'] - packed['m3']), 32)
        self.assertEqual(sorted({offset for rows, offset, prefix in packed['octo'] - packed['m3']}), [8, 24, 40, 56])
        self.assertEqual(sorted({prefix for rows, offset, prefix in packed['octo'] - packed['m3']}), list(range(1, 9)))


class EightRowDevicePiecesTests(OctoFixture):
    """The device pieces as they stand: the PINNED readers still refuse an eight-row segment (the octo twin, extent_attention_octo_tp, takes it: test_extent_attention_octo_tp), the pool
    lends the (8, 8) storage, the TP4 GDN sibling carries eight users, and the publication warm is kept for the octo block."""

    def test_the_extent_reader_refuses_an_eight_row_segment_today_it_bundles_as_one_group_not_two(self):
        with self.assertRaisesRegex(ValueError, r'Extent replay is qualified at G8B2 only \(0x27, K64j CB1\): a 8-row segment bundles as \[\[8\]\]'):
            self.block(octo_shape(WIDTH), pool_slots=tuple(range(8)))
        self.assertEqual(extent_attention_replay.EXTENT_BUNDLE_ENTRIES, 2)
        self.assertEqual(extent_attention_replay.EXTENT_FLAGS, 0x27)

    def test_one_group_per_bundle_would_run_the_flag_set_without_kv_share(self):
        # with only the bundle count patched the flags come out 0x25 (tail, slice, extent): the share bit needs a second batch to share
        with patch.object(extent_attention_replay, 'EXTENT_BUNDLE_ENTRIES', 1), \
                self.assertRaisesRegex(ValueError, r"qualified at 0x27 only .*gives \['0x25'\]"):
            self.block(octo_shape(WIDTH), pool_slots=tuple(range(8)))

    def test_the_pool_lends_extent_storage_for_bundles_of_two_groups_only(self):
        self.assertEqual(extent_bundle_batches(16, 8), (2,))
        self.assertEqual(extent_bundle_batches(8, 8), (1,))
        self.assertEqual(extent_bundle_batches(32, 8), (3, 1))

    def test_the_pool_lends_extent_storage_for_the_octo_shape_and_still_refuses_every_other_one_group_shape(self):
        import serving_buffer_pool as pool_module
        from test_serving_buffer_pool import FakeOperations, extent_pool

        self.assertEqual((pool_module.EXTENT_GROUP_ROWS, pool_module.EXTENT_BUNDLE_ENTRIES), (8, 2))
        self.assertEqual((pool_module.EXTENT_OCTO_SHAPE, pool_module.EXTENT_OCTO_BUNDLE_ENTRIES), ((8, 8), 1))
        self.assertEqual([pool_module.extent_bundle_entries(*shape) for shape in ((4, 16), (2, 16), (8, 8), (4, 8))], [2, 2, 1, 2])
        pool = extent_pool(FakeOperations(), users=8, packed_shapes=((4, 16), (8, 8)), packed_replicas={(4, 16): 2})
        storage = pool.packed_extent(8, 8)
        self.assertEqual((storage.users, storage.rows), (8, 8))
        for user in range(8):
            ((table,), (positions,)) = storage.tables[user], storage.cur_pos[user]
            self.assertEqual((table.shape, positions.shape), ((1, 68), (1,)), 'one bundle of one group per eight-row user')
        self.assertEqual(len(pool.extent[(4, 16)]), 2, 'two M3 sets beside the one octo set')
        # a four-user eight-row block (the old M2 shape) is still the refused one: one group per bundle is the octo block's alone
        with self.assertRaisesRegex(ValueError, 'qualified at G8B2 only'):
            extent_pool(FakeOperations(), packed_shapes=((4, 8),))

    def test_the_gdn_launch_carries_four_users_in_the_pinned_module_and_eight_in_the_tp4_sibling(self):
        import gdn_user_batch
        import gdn_user_batch_tp

        self.assertEqual((gdn_user_batch.MAX_USERS, gdn_user_batch_tp.MAX_USERS), (4, 8))
        with self.assertRaisesRegex(ValueError, 'One to 4 packed users per batched GDN launch'):
            gdn_user_batch.core_shares(11, 10, 8)
        self.assertEqual(len(gdn_user_batch_tp.core_shares(11, 10, 8, workers=12)), 8)
        with self.assertRaisesRegex(ValueError, 'One to 8 packed users per batched GDN launch'):
            gdn_user_batch_tp.core_shares(11, 10, 9, workers=12)

    def test_eight_users_of_twelve_heads_would_fit_the_grid_in_one_wave_if_the_limit_were_eight(self):
        # the cores are not the obstacle at four cards (12 value heads a chip): 8 x 12 = 96 of the 110 cores of the 11 x 10 grid, disjoint shares. The limit is the pinned
        # module's, and qualifying the launch at eight users is a card question; this is only the arithmetic the design rests on.
        import gdn_user_batch_tp

        if True:
            shares = gdn_user_batch_tp.core_shares(11, 10, 8, workers=12)
        self.assertEqual(len(shares), 8)
        cores = [point for share in shares for point in share]
        self.assertEqual((len(cores), len(set(cores))), (96, 96), 'disjoint')
        self.assertLessEqual(len(cores), 11 * 10)
        self.assertTrue(all(len(share) == 12 for share in shares))

    def test_the_second_and_later_blocks_skip_the_publication_warm_in_the_two_phase_build(self):
        import serving_runtime

        with open(serving_runtime.__file__, encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn('block.warm_publication = False      # block A\'s warm covered the same plan on the shared program cache', source)
        self.assertIn("getattr(block, 'keep_publication_warm', False) is not True", source, 'except a block whose plan has shapes no earlier warm ran: the octo block\'s')
        self.assertIn('octo_block.keep_publication_warm = True', source)


class FusedThirdBlockTests(unittest.TestCase):
    """The fused commit (fused_commit_tp.FusedCommit) built over a fake four-card block of eight T8 users, on test_fused_commit_tp4's fixture: the design's 8 projection traces and
    64 slide traces. Host path only - the T_proj / slide programs are the fixture's stand-ins; what runs for real is the twin's own geometry code."""

    def setUp(self):
        from types import SimpleNamespace

        import test_fused_commit_tp4 as fused_tests

        class Fixture(fused_tests.TwinFixture):
            def runTest(self):                      # (a fixture, not a test)
                pass

        self.fixture = Fixture()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        fixture.slots = [fused_tests.pooled_slot(fixture.ops, index) for index in range(8)]
        fixture.block = SimpleNamespace(rows_per_user=8, users=8, shape=octo_shape(fused_tests.PAGE_WIDTH), segment_slots=fixture.slots, taps=fixture.taps, rounds=3,
                                        segment_of=lambda engine: engine.segment)

    def test_eight_projection_traces_and_sixty_four_slide_traces_each_warmed_first(self):
        fused = self.fixture.build()
        fused.capture(self.fixture.capture)
        self.assertEqual(fused.trace_count(), 8 + 64)
        for storage in fused.segments:
            self.assertEqual(sorted(storage.slides), list(range(1, 9)), 'one slide per accepted prefix 1..8')
        self.assertEqual(len(fused.segments), 8)
        captures = [entry for entry in self.fixture.ops.log if entry[0] == 'capture']
        self.assertEqual(len(captures), 72)
        self.assertEqual(sum(1 for entry in self.fixture.ops.log if entry[0] == 'generic_op'), 8 * 8 * 2)

    def test_each_segments_feature_projection_reads_its_own_eight_rows(self):
        fused = self.fixture.build()
        fused.capture(self.fixture.capture)
        offsets = [call[3:] for call in self.fixture.calls if call[0] == 'project_features']
        self.assertEqual(offsets[::2], [(8, 8 * segment) for segment in range(8)])

    def test_the_engaged_line_is_the_one_the_smoke_rule_expects_of_a_third_block(self):
        fused = self.fixture.build()
        fused.capture(self.fixture.capture)
        line = fused.engaged_line()
        self.assertTrue(line.startswith('[PINDIAG] fused commit engaged users=8 rows=8 inplace=1 live_banks=0 audit=0 kernel='), line)
        self.assertRegex(line, r' traces=72 layout=1x80 tp=4 workers=8 ')


if __name__ == '__main__':
    unittest.main()
