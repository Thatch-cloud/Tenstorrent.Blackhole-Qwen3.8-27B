"""The eight-seat census (A0'): every site of the S2 four-card serving path that assumes four seats, a 64-row block or a
131,072-token window, stated as BEHAVIOUR on fakes - what attaches, what is refused and by what - not as a scan of the
source for a literal 4 or 64 (at four cards 4 is also the chip count and 64 the page size, so a literal scan would end as
a pinned allowlist).

The eight-seat build is two 64-row M3 blocks (QWEN_FAST_M3_BLOCKS=2): block A over pool slots 0-3 and block B over 4-7,
each exactly today's qualified 4-user block, run back to back in one step. So each site below is one of two kinds:

  UNCHANGED  the site is per block (or per chip, or per page) and the M3x2 build leaves it alone ON PURPOSE. The test
             pins the value or the refusal, so a change of it later - a 128-row block, a widened per-launch user count -
             fails here and sends the reader to the eight-seat design, instead of slipping through.
  CHANGED    the site decides by the scheduler request count or the block count and answers differently under the flag.
             The test shows the refusal without the flag and the attach with it.

The sites are the map's sections 2, 3 and 9 (seat count, the 64-row block and 16 rows per seat, sampler widths) and the
131,072-token window sites of section 4 that live in this lane's files. Sites in files the other lane owns (the profiles,
the pair coordinator's slot pairs, the prefill admission's DRAM terms, the gate and the smoke) are named in the last test
and tested there. The tests need no device."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch  # noqa: F401 - imported before any test patches sys.modules

import gdn_user_batch
import packed_ordered_cache
import packed_shapes
import packed_verifier
import serving_buffer_pool
import serving_fast_policy
import serving_runtime
import verify_trace_t2
from force_argmax import SAMPLE_WIDTHS, SAMPLER_ROWS

M3_ENV = {'QWEN_FAST_FOUR_AS_TWO': '0', 'QWEN_FAST_PACKED_STEP': '1'}


def policy(users):
    return dict(scheduler_requests=users)


class SeatCountSites(unittest.TestCase):
    """Map section 2: sites that decide by the number of scheduler requests."""

    def test_the_native_gdn_slot_ceiling_is_eight_and_nine_seats_stay_refused(self):
        # UNCHANGED: eight seats fit the eight native GDN slots; a ninth needs a graft (NATIVE_GDN_SLOTS), not this build.
        self.assertEqual(serving_fast_policy.NATIVE_GDN_SLOTS, 8)
        from test_serving_fast_policy import FastPolicyTests

        for seats, accepted in ((4, True), (8, True), (9, False)):
            config = FastPolicyTests().fixture()
            config.scheduler_config.max_num_seqs = seats
            config.scheduler_config.scheduler_cls = 'named'
            with self.subTest(seats=seats):
                if accepted:
                    self.assertEqual(serving_fast_policy.validate_fast_config(config)['scheduler_requests'], seats)
                else:
                    with self.assertRaisesRegex(ValueError, 'at most 8 requests'):
                        serving_fast_policy.validate_fast_config(config)

    def test_eight_requests_are_not_the_m3_shape_without_the_block_count(self):
        # CHANGED under QWEN_FAST_M3_BLOCKS=2 (see MeetsUnderTheFlag): without it eight requests are no M3 block, so
        # every site that asks m3_shape - the single gate/up copy, the padded block, the extent admission - refuses.
        met, shape = serving_runtime.m3_shape(policy(8), dict(M3_ENV))
        self.assertFalse(met)
        self.assertTrue(shape.startswith('users=8 '))
        self.assertTrue(serving_runtime.m3_shape(policy(4), dict(M3_ENV))[0])

    def test_the_single_gate_up_copy_and_the_padded_block_are_refused_at_eight_requests_without_the_block_count(self):
        for call, message in ((lambda env: serving_runtime.register_reader_reason(policy(8), env),
                               'QWEN_FAST_SINGLE_GATEUP=1 is admitted only at the 64-row M3 block'),
                              (lambda env: serving_runtime.padded_block_admission(policy(8), env),
                               'QWEN_FAST_PADDED_BLOCK=1 is admitted only at the 64-row M3 block')):
            flag = 'QWEN_FAST_SINGLE_GATEUP' if 'SINGLE' in message else 'QWEN_FAST_PADDED_BLOCK'
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, message):
                call(dict(M3_ENV, **{flag: '1'}))

    def test_the_extent_admission_refuses_eight_requests_without_the_block_count(self):
        import packed_any_admission

        problems = packed_any_admission.check_environment(
            {'QWEN_FAST_REPLAY_GROUP_ROWS': '8', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1',
             'QWEN_FAST_SDPA_MODES': 'tail,share,slice', 'QWEN_FAST_ANY_REQUEST': '1',
             'QWEN_FAST_EXTENT_REPLAY': '1'}, serving_runtime.m3_shape(policy(8), dict(M3_ENV)))
        self.assertTrue(any('64-row M3 block' in problem for problem in problems), problems)

    def test_no_packed_block_serves_eight_requests_by_the_request_count_alone(self):
        # UNCHANGED: packed_shapes.serving_shape picks the block by request count and has none for eight; the two-block
        # build does not go through it (serving_runtime builds m3_shape twice), so a caller of serving_shape still gets
        # None and the sequential step - the 8-seat build is opt-in by flag, never by a count.
        self.assertEqual(sorted(packed_shapes.SERVING_SHAPES), [2, 4])
        for users in (1, 3, 5, 6, 7, 8):
            self.assertIsNone(packed_shapes.serving_shape(users, 68))
        self.assertEqual(packed_shapes.serving_shape(4, 68), packed_shapes.m3_shape(68))

    def test_the_pool_holds_eight_slots_and_the_two_shapes_the_blocks_ask_for(self):
        # UNCHANGED: the pool's ceiling is the native slot count; the default table sets are the M1 and M3 shapes; the
        # second M3 block's set is a REPLICA of the first's shape (packed_replicas), not a new shape.
        self.assertEqual(serving_buffer_pool.PACKED_REPLAY_SHAPES, ((2, 16), (4, 16)))
        self.assertEqual(serving_buffer_pool.default_packed_shapes(8), ((2, 16), (4, 16)))
        self.assertEqual(serving_buffer_pool.default_packed_shapes(4), ((2, 16), (4, 16)))
        self.assertEqual(serving_buffer_pool.default_packed_shapes(2), ((2, 16),))
        self.assertEqual(serving_buffer_pool.validate_packed_shapes(((4, 16),)), ((4, 16),))
        for shape in ((8, 16), (16, 8), (2, 64)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                serving_buffer_pool.validate_packed_shapes((shape,))
        with self.assertRaisesRegex(ValueError, 'within the 8 native GDN slots'):
            serving_buffer_pool.ServingBufferPool(SimpleNamespace(), 'mesh', users=9)

    def test_the_gdn_batched_launch_stays_four_users_per_block(self):
        # UNCHANGED, per block: the batched GDN launch carries at most MAX_USERS users, and a block's users are its own
        # four - eight users in one launch would be 96 of 110 cores (a 128-row block, not this build).
        self.assertEqual(gdn_user_batch.MAX_USERS, 4)
        self.assertEqual(packed_shapes.m3_shape(68).users, gdn_user_batch.MAX_USERS)
        gdn_user_batch.core_shares(11, 10, 4)
        with self.assertRaisesRegex(ValueError, 'One to 4 packed users per batched GDN launch'):
            gdn_user_batch.core_shares(11, 10, 5)
        import gdn_user_batch_tp

        self.assertEqual(gdn_user_batch_tp.MAX_USERS, gdn_user_batch.MAX_USERS)
        with self.assertRaisesRegex(ValueError, 'One to 4 packed users per batched GDN launch'):
            gdn_user_batch_tp.core_shares(11, 10, 8)

    def test_a_padded_round_idles_at_most_two_segments_of_a_block(self):
        # UNCHANGED, per block: page 0 holds two 32-row tile rows, so a 4-user block pads from two live users. Under two
        # blocks that is per block: a block with one live user falls to the sequential step (serving_packed_step).
        self.assertEqual(packed_verifier.PackedVerifierEngine.MAX_IDLE_SEGMENTS, 2)
        shape = packed_shapes.m3_shape(68)
        self.assertEqual(max(1, shape.users - packed_verifier.PackedVerifierEngine.MAX_IDLE_SEGMENTS), 2)


class BlockSites(unittest.TestCase):
    """Map section 3: sites that assume the 64-row block, 16 rows per seat. Every one is per block under M3x2."""

    def test_the_largest_block_is_sixty_four_rows_and_a_128_row_block_is_refused(self):
        # UNCHANGED: M3x2 builds two 64-row blocks; the 128-row block (I3) is a graft, kernel and byte gate away.
        self.assertEqual(max(packed_shapes.BLOCK_ROWS), 64)
        packed_shapes.validate_shape(packed_shapes.m3_shape(68))
        with self.assertRaisesRegex(ValueError, 'within the 64-row block'):
            packed_shapes.validate_shape(packed_shapes.PackedShape(8, 16, 128, 68, 68 * 64))
        self.assertEqual(serving_fast_policy.PACKED_BLOCK_WIDTHS, (32, 64))
        with self.assertRaises(ValueError):
            serving_fast_policy.packed_geometry(8, 128)

    def test_the_per_request_engines_beside_two_m3_blocks_capture_the_sequential_widths_only(self):
        # UNCHANGED per block: beside a 64-row block the engines capture (1, 2, 4); serving_runtime applies the same
        # trim explicitly for two blocks (the 8- and 16-row captures are about 2.4 GB each, eight engines).
        self.assertEqual(packed_shapes.sequential_capture_rows(packed_shapes.m3_shape(68)), 4)
        self.assertEqual(packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS, 4)

    def test_a_blocks_commit_traces_are_its_own_sixty_four(self):
        # UNCHANGED per block: users x rows_per_user commit traces; two blocks capture 2 x 64, not 128 in one.
        self.assertEqual(packed_shapes.commit_traces(packed_shapes.m3_shape(68)), 64)

    def test_the_sampler_launches_at_most_sixty_four_rows_in_two_tiles(self):
        # UNCHANGED (map section 9): force_argmax tiles a 64-row block into two 32-row sampler calls; a 128-row width is
        # not in the table. Under two blocks the sampler runs once per block inside each block's own verify trace.
        self.assertEqual((SAMPLER_ROWS, max(SAMPLE_WIDTHS)), (32, 64))
        self.assertNotIn(128, SAMPLE_WIDTHS)
        import force_argmax

        with self.assertRaisesRegex(ValueError, 'Supported verifier width required'):
            force_argmax.sample_rows(object(), SimpleNamespace(shape=(1, 1, 128, 8)), 128, object())

    def test_the_kv_writer_launches_at_most_sixty_four_rows(self):
        # UNCHANGED: the ordered K/V writer takes 64 rows on 64 cores (or 32); two blocks write one after the other.
        self.assertEqual(packed_ordered_cache.LAUNCH_ROWS, (64, 32))
        self.assertEqual(verify_trace_t2.KV_ROWS, (64, 32))

    def test_the_row_width_tables_stop_at_sixty_four(self):
        # UNCHANGED: each of these tables is a block's, so none moves with the seat count.
        import gdn_records
        import model_batch
        import pooled_attention_replay
        import verifier_engine

        self.assertEqual(pooled_attention_replay.PACKED_BLOCK_ROWS, 64)
        self.assertEqual(gdn_records.BLOCK_ROWS, (2, 4, 8, 16, 32, 64))
        self.assertEqual(verifier_engine.VERIFY_WIDTHS, (1, 2, 4, 8, 16, 32, 64))
        self.assertEqual(verifier_engine.BLOCK_WIDTHS, (16, 32, 64))
        import target_packed_pages

        pages = torch.zeros((1, 68), dtype=torch.int32)
        users = [dict(start=4096, rows=16, pages=pages) for user in range(4)]
        self.assertEqual(target_packed_pages.packed_rows(users)['rows'], 64)
        with self.assertRaisesRegex(ValueError, 'legal verifier block width'):
            target_packed_pages.packed_rows([dict(start=4096, rows=16, pages=pages) for user in range(8)])

    def test_the_window_sites_in_this_lane_keep_the_131k_geometry_at_i1(self):
        # UNCHANGED at I1: the page-table width a 131,328-token window needs is 2,052 pages; the ordered writer admits
        # that width only, so an eight-seat build at I1 serves the same window (4,096 is I2, behind its own evidence).
        import ordered_cache

        self.assertEqual(max(68, -(-131328 // 64)), 2052)
        self.assertEqual(ordered_cache.WIDE_PAGE_WIDTHS, {2052})


class SitesOutsideThisLane(unittest.TestCase):
    """Named, not tested here: the sites of the map in files the profile / gate lane owns. They are the seat count in the
    profiles (max-num-seqs, the block count, the trace region), the pair coordinator's fixed slot pairs, the prefill
    admission's DRAM terms and the gate's four-live arms. Each is tested where it lives; this test only keeps the list."""

    def test_the_hand_off_list_is_the_one_the_design_names(self):
        sites = ('qwen_c2_profiles.json: c2-packed-tp4-8 (max-num-seqs 8, num-gpu-blocks-override 16,416, trace region 512 MiB)',
                 'dflash_packed_proposal*.py: draft pairs (4,5) and (6,7)',
                 'serving_prefill_admission.py: the DRAM split at eight engines and two blocks',
                 'c2_serving_gate.py / c2_serving_smoke.py / acceptance_report.py: eight-live arms')
        self.assertEqual(len(sites), 4)


if __name__ == '__main__':
    unittest.main()
