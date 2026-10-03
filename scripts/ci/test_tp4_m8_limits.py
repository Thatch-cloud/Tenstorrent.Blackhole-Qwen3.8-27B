"""tp4/m8-phase1: the 128-row / 8-user limits gate (the CPU gate the review of block128-design.md asked for).

Under QWEN_FAST_TP=4, every 64-row / 4-user limit of m8_limits.LIMITS is held to the code at 128 rows and eight users:

  accepts   the code takes the 128-row / 8-user input as it stands;
  rebound   before tp4_m8.install() the code refuses (or answers for four users: the flag-off bytes), after it the code takes the input, and
            the twin logged its engaged marker; uninstall() puts the originals back;
  phase2    the code still refuses at 128 / 8 (so phase 2 cannot land without flipping its row), and the line the table points at is still
            where the table says.

Every row's file:line is found by its regular expression, a row marked pinned is listed by a recorded pin manifest (and one that is not, is
not), and tp4_m8.py touches no pinned file (it rebinds names). Nothing here opens a device."""

import inspect
import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import m8_limits  # noqa: E402
import tp4_m8  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BY_ID = {entry['id']: entry for entry in m8_limits.LIMITS}


def refuses(call, *args, **options):
    try:
        call(*args, **options)
    except Exception as error:  # noqa: BLE001 - any refusal is a refusal
        return error
    return None


class Probes(object):
    """One method per row: `before` is what the code does with the 128-row / 8-user input as shipped, `after` once tp4_m8 is installed (or,
    for accepts and phase2 rows, the same state as before)."""

    def __init__(self, case):
        self.case = case

    # -- helpers ------------------------------------------------------------------------------------------------------------
    def pages(self):
        return SimpleNamespace(ndim=2, shape=(1, 68))

    def fake_tensor(self, ops, width=4128, rows=16):
        return SimpleNamespace(shape=(1, rows, width), dtype=ops.bfloat16, layout=ops.TILE_LAYOUT,
                               memory_config=lambda: ops.DRAM_MEMORY_CONFIG)

    def fake_ops(self):
        return SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1')

    # -- rows ---------------------------------------------------------------------------------------------------------------
    def row_1(self, installed):
        import packed_shapes

        shape = packed_shapes.PackedShape(8, 16, 128, 68, 68 * 64)
        if not installed:
            self.case.assertIsNotNone(refuses(packed_shapes.validate_shape, shape))
            self.case.assertIsNotNone(refuses(tp4_m8.m8_shape, 68))
        else:
            self.case.assertEqual(packed_shapes.validate_shape(shape), shape)
            self.case.assertEqual(tp4_m8.m8_shape(68), shape)
            # nothing wider, nothing that is not users x rows_per_user
            self.case.assertIsNotNone(refuses(packed_shapes.validate_shape, shape._replace(block_rows=256)))
            self.case.assertIsNotNone(refuses(packed_shapes.validate_shape, shape._replace(users=4)))

    def row_2(self, installed):
        import serving_buffer_pool

        if not installed:
            self.case.assertIsNotNone(refuses(serving_buffer_pool.validate_packed_shapes, ((8, 16),)))
        else:
            self.case.assertEqual(serving_buffer_pool.validate_packed_shapes(((8, 16),)), ((8, 16),))
            self.case.assertEqual(serving_buffer_pool.PACKED_BLOCK_ROWS, 128)

    def row_3(self, installed):
        import serving_buffer_pool

        self.case.assertEqual(serving_buffer_pool.default_packed_shapes(8), ((2, 16), (4, 16)))

    def row_4(self, installed):
        import packed_shapes

        self.case.assertIsNone(packed_shapes.serving_shape(8, 68))
        self.case.assertEqual(sorted(packed_shapes.SERVING_SHAPES), [2, 4])

    def row_5(self, installed):
        import serving_runtime

        met, text = serving_runtime.m3_shape({'scheduler_requests': 8}, {'QWEN_FAST_FOUR_AS_TWO': '0', 'QWEN_FAST_PACKED_STEP': '1',
                                                                          'QWEN_FAST_M8_BLOCK': '1'})
        self.case.assertFalse(met, text)

    def row_6(self, installed):
        import packed_any_admission

        problems = packed_any_admission.check_environment({}, (False, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1'))
        self.case.assertTrue(any('64-row M3 block' in problem for problem in problems), problems)

    def row_7(self, installed):
        import serving_packed_step

        self.case.assertFalse(serving_packed_step.pads(SimpleNamespace(), 6))

    def row_8(self, installed):
        import packed_verifier

        engine = packed_verifier.PackedVerifierEngine
        self.case.assertEqual(engine.MAX_IDLE_SEGMENTS, 2)
        # the padded rule for one block of eight users: at least 8 - 2 live
        self.case.assertEqual(max(1, 8 - engine.MAX_IDLE_SEGMENTS), 6)

    def row_9(self, installed):
        import target_packed_pages

        users = [dict(start=100 * index, rows=16, pages=self.pages()) for index in range(8)]
        if not installed:
            error = refuses(target_packed_pages.validate_users, users)
            self.case.assertIn('legal verifier block width', str(error))
        else:
            self.case.assertEqual(target_packed_pages.validate_users(users)[1], 128)
            self.case.assertEqual(target_packed_pages.segments(users)[0][-1], (112, 128))
            # four users still total 64, and 96 rows are still no legal width
            self.case.assertEqual(target_packed_pages.validate_users(users[:4])[1], 64)
            self.case.assertIsNotNone(refuses(target_packed_pages.validate_users, users[:6]))

    def row_10(self, installed):
        import verifier_pack

        layers = verifier_pack.GDN_LAYERS
        participants = [dict(start=0, rows=16, pages=self.pages(), prefix=0, checkpoints=(), slots=[object() for _ in range(layers)])
                        for _ in range(8)]
        if not installed:
            self.case.assertIsNotNone(refuses(verifier_pack.build_pack, participants, block_rows=128))
        else:
            self.case.assertEqual(len(verifier_pack.build_pack(participants, block_rows=128)), 8)
            self.case.assertIsNotNone(refuses(verifier_pack.build_pack, participants, block_rows=256))

    def row_11(self, installed):
        import packed_shapes

        m8 = packed_shapes.PackedShape(8, 16, 128, 68, 68 * 64)
        if not installed:
            self.case.assertIsNotNone(refuses(packed_shapes.sequential_capture_rows, m8))
        else:
            self.case.assertEqual(packed_shapes.sequential_capture_rows(m8), packed_shapes.M3_SEQUENTIAL_CAPTURE_ROWS)
        # the narrower blocks answer what they always answered
        self.case.assertEqual(packed_shapes.sequential_capture_rows(packed_shapes.m3_shape(68)), 4)
        self.case.assertEqual(packed_shapes.sequential_capture_rows(packed_shapes.m1_shape(68)), 16)
        self.case.assertEqual(packed_shapes.sequential_capture_rows(None), 16)

    def row_12(self, installed):
        import verifier_engine
        import verifier_engine_tp

        if not installed:
            self.case.assertIsNotNone(refuses(verifier_engine.capture_widths, 0, 4352, 128, 128, 128))
            self.case.assertIs(verifier_engine_tp.VERIFY_WIDTHS, verifier_engine.VERIFY_WIDTHS)
        else:
            # the four-card engine twin imports the same tuple by name: it follows
            self.case.assertEqual(verifier_engine_tp.VERIFY_WIDTHS[-1], 128)
            self.case.assertIs(verifier_engine_tp.VERIFY_WIDTHS, verifier_engine.VERIFY_WIDTHS)
            self.case.assertEqual(verifier_engine.capture_widths(0, 4352, 128, 128, 128)[-1], 128)
            # the default cap stays 32: the per-request engines capture nothing wider
            self.case.assertEqual(verifier_engine.capture_widths(0, 4352, 128, 128)[-1], 32)

    def row_13(self, installed):
        import serving_fast_policy

        if not installed:
            self.case.assertIsNotNone(refuses(serving_fast_policy.packed_geometry, 8, 128))
        else:
            found = serving_fast_policy.packed_geometry(8, 128)
            self.case.assertEqual((found['users'], found['block_rows'], found['proposals_per_user']), (8, 16, 15))

    def row_14(self, installed):
        import dflash_packed_proposal

        self.case.assertEqual(dflash_packed_proposal.BLOCK_WIDTHS, (32, 64))

    def row_15(self, installed):
        import model_batch

        if not installed:
            self.case.assertIsNotNone(refuses(model_batch.validate_checkpoint, 128, 128))
        else:
            model_batch.validate_checkpoint(128, 128)
            self.case.assertIsNotNone(refuses(model_batch.validate_checkpoint, 256, 0))

    def row_16(self, installed):
        text = m8_limits.read(os.path.join(ROOT, 'scripts', 'ci', 'model_batch.py'))
        self.case.assertNotIn('progcfg_128', text)

    def row_17(self, installed):
        import gdn_prefix

        if not installed:
            self.case.assertIsNotNone(refuses(gdn_prefix.validate_rows, (1, 128, 5120)))
        else:
            self.case.assertEqual(gdn_prefix.validate_rows((1, 128, 5120)), 128)
            self.case.assertIsNotNone(refuses(gdn_prefix.validate_rows, (1, 96, 5120)))

    def row_18(self, installed):
        import gdn_device_loop_state as state

        calls = []

        class Layer(object):
            def __init__(self, *attributes):
                self.args = SimpleNamespace(**{name: True for name in attributes})

            def _project_qkvzab_raw(self, packed, rows, memory):
                calls.append(rows)
                return ('raw', rows)

        class Ops(object):
            L1_MEMORY_CONFIG = 'l1'
            slices = 0

            def slice(self, tensor, start, stop):
                Ops.slices += 1
                return SimpleNamespace(shape=(1, stop[1] - start[1], tensor.shape[-1]))

            def deallocate(self, tensor):
                pass

            def concat(self, tensors, dim):
                return ('concat', len(tensors))

        packed = SimpleNamespace(shape=(1, 128, 4128))
        # the shipped function at 128 rows on a graft with only the _64 configs: four 32-row calls and no marker (review item 2)
        result = state.project_qkvzab_by_tile(Ops(), Layer('attn_wo_decode_1d_progcfg_64'), packed, 128)
        self.case.assertEqual((calls, result), ([32, 32, 32, 32], ('concat', 4)))
        if installed:
            self.case.assertEqual(tp4_m8.STATS['engaged'] - 1, 0)    # only the install line so far
            del calls[:]
            self.case.assertEqual(tp4_m8.STATS['fallback'], 1)       # that call said so
            result = state.project_qkvzab_by_tile(Ops(), Layer('attn_wo_decode_1d_progcfg_64', tp4_m8.NATIVE_ATTRIBUTE), packed, 128)
            self.case.assertEqual((calls, result), ([128], ('raw', 128)))
            self.case.assertEqual(tp4_m8.STATS['engaged'], 2)
            # 64 rows and below are the original's own call
            del calls[:]
            state.project_qkvzab_by_tile(Ops(), Layer('attn_wo_decode_1d_progcfg_64'), SimpleNamespace(shape=(1, 64, 4128)), 64)
            self.case.assertEqual(calls, [64])

    def row_19(self, installed):
        import gdn_records

        if not installed:
            self.case.assertIn('Multirow packed-history block required', str(refuses(gdn_records.RetainedGDNBlock, 128, object())))
        else:
            self.case.assertEqual(gdn_records.RetainedGDNBlock(128, object()).rows, 128)

    def row_20(self, installed):
        import extent_attention_replay_tp
        import pooled_attention_replay

        segments = [(16 * index, 16 * index + 16) for index in range(8)]
        if not installed:
            self.case.assertIn('at most a 64-row block', str(refuses(pooled_attention_replay.validate_segments, segments)))
            self.case.assertIsNot(extent_attention_replay_tp.validate_segments, tp4_m8)
        else:
            self.case.assertEqual(pooled_attention_replay.validate_segments(segments), tuple(segments))
            # the pinned reader's own global is the twin now, and nothing wider than the block is taken
            self.case.assertIs(extent_attention_replay_tp.validate_segments, pooled_attention_replay.validate_segments)
            self.case.assertIsNotNone(refuses(pooled_attention_replay.validate_segments, segments + [(128, 144)]))
            self.case.assertEqual(pooled_attention_replay.validate_segments(segments[:4]), tuple(segments[:4]))

    def row_21(self, installed):
        import force_argmax

        error = refuses(force_argmax.sample_rows, SimpleNamespace(), SimpleNamespace(shape=(1, 1, 64)), 128, SimpleNamespace())
        if not installed:
            self.case.assertIn('Supported verifier width required', str(error))
        else:
            self.case.assertNotIn('Supported verifier width required', str(error))

    def row_22(self, installed):
        pass   # count and line are checked for every row; the gates are the graft's (phase 2)

    row_23 = row_24 = row_22

    def row_25(self, installed):
        text = m8_limits.read(os.path.join(ROOT, 'docker', 'qwen-c2-graft', 'graft', 'model_config.py'))
        self.case.assertNotIn('progcfg_128', text)

    def row_26(self, installed):
        pass   # an image-side limit: the B1 harness covers it (test_m8_matmul_plan holds that)

    def row_27(self, installed):
        import two_tile_norm

        self.case.assertEqual(two_tile_norm.validate_two_tile_rows(128), 4)

    def row_28(self, installed):
        pass

    row_29 = row_28

    def row_30(self, installed):
        import packed_ordered_cache

        self.case.assertEqual(packed_ordered_cache.LAUNCH_ROWS, (64, 32))
        error = refuses(packed_ordered_cache.validate_chained, (4096, 1, 64, 256), (1, 128, 32, 256), (128,), (128, 68), 128)
        self.case.assertIn('64 or 32', str(error))

    def row_31(self, installed):
        import verify_trace_t2

        self.case.assertEqual(verify_trace_t2.KV_ROWS, (64, 32))
        with patch.dict(os.environ, {'QWEN_FAST_VERIFY_T2_KV_ROWS': '128'}):
            self.case.assertIsNotNone(refuses(verify_trace_t2.kv_rows))

    def row_32(self, installed):
        import tile_collective_tp

        self.case.assertEqual(tile_collective_tp.tile_spans(128), ((0, 32), (32, 64), (64, 96), (96, 128)))

    def row_33(self, installed):
        import gdn_user_batch

        self.case.assertEqual(gdn_user_batch.MAX_USERS, 4)   # the pinned pair module stays what it is, installed or not

    def row_34(self, installed):
        import gdn_user_batch_tp

        if not installed:
            self.case.assertIsNotNone(refuses(gdn_user_batch_tp.core_shares, 11, 10, 8, workers=12))
        else:
            shares = gdn_user_batch_tp.core_shares(11, 10, 8, workers=12)
            points = [point for share in shares for point in share]
            self.case.assertEqual((len(shares), len(points), len(set(points))), (8, 96, 96))
            self.case.assertLess(max(point[0] for point in points), 11)
            self.case.assertIsNotNone(refuses(gdn_user_batch_tp.core_shares, 11, 10, 9, workers=12))

    def row_35(self, installed):
        import gdn_seq_block

        self.case.assertEqual(gdn_seq_block.batch.MAX_USERS, 8 if installed else 4)

    def row_36(self, installed):
        import gdn_conv_windows_packed as windows
        import gdn_multitoken_conv

        ops = self.fake_ops()
        mesh = SimpleNamespace(shape=(1, 4))
        users = [(self.fake_tensor(ops), [self.fake_tensor(ops) for _ in range(4)]) for _ in range(8)]
        with patch.object(gdn_multitoken_conv, 'validate_projected', lambda shape, history: 16):
            if not installed:
                self.case.assertEqual(windows.unsupported(ops, mesh, users), '8 users outside 1..4')
            else:
                self.case.assertIsNone(windows.unsupported(ops, mesh, users))
                self.case.assertIn('9 users outside', windows.unsupported(ops, mesh, users + users[:1]))
                # a user that is not bf16 is still found, in its own group of four
                bad = list(users)
                bad[5] = (SimpleNamespace(shape=(1, 16, 4128), dtype='f32', layout=ops.TILE_LAYOUT,
                                          memory_config=lambda: ops.DRAM_MEMORY_CONFIG), bad[5][1])
                self.case.assertIn('users 4-7: user 1: not bf16 TILE', windows.unsupported(ops, mesh, bad))
            self.case.assertIsNone(windows.unsupported(ops, mesh, users[:4]))

    def row_37(self, installed):
        import gdn_block_conv_tp
        import gdn_rows_dma_tp

        tasks = gdn_block_conv_tp.plan_unstack(8, 4128)
        self.case.assertEqual(len(tasks), 3600)
        self.case.assertIsNotNone(refuses(gdn_rows_dma_tp.distribute, tasks, 110))   # the original planner's limit is the module's, and stays
        self.case.assertEqual(tp4_m8.launch_capacity(110), 3410)
        calls = []

        def original(mesh, sources, destinations, chunk, **options):
            calls.append(len(chunk))
            if len(chunk) > tp4_m8.launch_capacity(110):
                raise gdn_rows_dma_tp.Unsupported('too long')

        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        twin = tp4_m8.make_launch(original)
        twin(mesh, [], [], tasks)
        self.case.assertEqual(calls, [1800, 1800])
        del calls[:]
        twin(mesh, [], [], tasks[:3410])
        self.case.assertEqual(calls, [3410])                                       # one launch when it fits
        if installed:
            self.case.assertIn('make_launch.<locals>.launch', gdn_rows_dma_tp.launch.__qualname__)
        else:
            self.case.assertEqual(gdn_rows_dma_tp.launch.__qualname__, 'launch')

    def row_38(self, installed):
        import gdn_block_conv_tp

        tasks = gdn_block_conv_tp.plan_unstack(8, 4128)
        # a destination tensor and page that two tasks both write (a different payload each): the four-user layout at eight users
        pairs = [(destination, page) for destination, page, _first, _second in tasks]
        duplicates = len(pairs) - len(set(pairs))
        if not installed:
            self.case.assertGreater(duplicates, 0)
            self.case.assertEqual(max(destination for destination, _page in pairs), 16 + 4 * 7 + 3)
        else:
            self.case.assertEqual(duplicates, 0)
            self.case.assertEqual(sorted({destination for destination, _page in pairs}), list(range(8 * 4 + 8 * 4)))

    def row_39(self, installed):
        import gdn_rows_dma_tp as rows_dma

        for build, expected in ((rows_dma.split_pieces, 1032), (rows_dma.canon_block, 516), (rows_dma.merge_outputs, 516)):
            tasks = build(8, 4128)
            self.case.assertEqual(len(tasks), expected, build.__name__)
            self.case.assertEqual(len(rows_dma.distribute(tasks, 110)), 110)
        self.case.assertLessEqual(len(rows_dma.stack_users(8, 2560)), 3410)

    def row_40(self, installed):
        import attention_block_fold_tp as fold

        segments = [(16 * index, 16 * index + 16) for index in range(8)]
        bundles = [[[dict(offset=0, rows=8), dict(offset=8, rows=8)]] for _ in segments]
        chunks = fold.chunks_of(segments, bundles)
        for plan in (fold.forward_tasks(chunks), fold.inverse_tasks(chunks)):
            flat = [task for chunk in plan for task in chunk]
            per_core = fold.distribute(flat, 110)
            self.case.assertLessEqual(2 + fold.TASK_WORDS * fold.capacity(per_core), 256)
        self.case.assertEqual(sum(len(chunk) for chunk in fold.inverse_tasks(chunks)), 1024)

    def row_41(self, installed):
        import ops_profile_plan

        profiles = {'profiles': {'eight': {'engine': {'max-num-seqs': 8}}, 'four': {'engine': {'max-num-seqs': 4}}}}
        self.case.assertEqual(ops_profile_plan.plan_users(profiles, 'eight'), 8)
        self.case.assertEqual(ops_profile_plan.plan_users(profiles, 'four'), 4)

    def row_42(self, installed):
        pass


class LimitsTests(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        self.environ.start()
        tp4_m8.uninstall()
        self.addCleanup(self.environ.stop)
        self.addCleanup(tp4_m8.uninstall)
        self.probes = Probes(self)

    def probe(self, entry, installed):
        getattr(self.probes, 'row_%d' % entry['id'])(installed)

    # -- the table ----------------------------------------------------------------------------------------------------------
    def test_every_row_has_a_probe_and_a_known_status_and_ids_are_dense(self):
        self.assertEqual([entry['id'] for entry in m8_limits.LIMITS], list(range(1, len(m8_limits.LIMITS) + 1)))
        for entry in m8_limits.LIMITS:
            with self.subTest(row=entry['id']):
                self.assertIn(entry['status'], (m8_limits.ACCEPTS, m8_limits.REBOUND, m8_limits.PHASE2))
                self.assertTrue(callable(getattr(self.probes, 'row_%d' % entry['id'], None)))
                self.assertTrue(entry['action'])

    def test_the_designs_list_the_reviews_seven_and_this_phases_finds_are_all_in_the_table(self):
        sources = {}
        for entry in m8_limits.LIMITS:
            sources[entry['source']] = sources.get(entry['source'], 0) + 1
        self.assertEqual(sorted(entry['id'] for entry in m8_limits.LIMITS if entry['source'] == 'review 1'), [17])
        review_seven = [entry['id'] for entry in m8_limits.LIMITS if entry['source'].startswith('review ') and entry['source'][7:].isdigit()]
        self.assertEqual(sorted(review_seven), [11, 17, 18, 19, 20, 36, 37])   # review items 7, 1, 2, 3, 4, 5, 6
        found = {entry['id'] for entry in m8_limits.LIMITS if entry['source'] == 'phase 1'}
        self.assertTrue({9, 10, 12, 13, 14, 38} <= found, found)

    def test_every_rows_line_is_still_where_the_table_says(self):
        for entry in m8_limits.LIMITS:
            if entry['needle'] is None:
                continue
            with self.subTest(row=entry['id'], file=entry['file']):
                line, count = m8_limits.locate(entry, ROOT)
                self.assertIsNotNone(line, 'the code moved: re-read this limit and fix its pattern')
                self.assertEqual(count, entry['count'])

    def test_pinned_rows_are_listed_by_a_recorded_pin_and_the_others_are_not(self):
        pins = m8_limits.pin_files(ROOT)
        self.assertIn('tp2_pinned_sources.json', pins)
        for entry in m8_limits.LIMITS:
            if entry['needle'] is None or not entry['file'].startswith('scripts/ci/'):
                continue
            with self.subTest(row=entry['id']):
                kind = m8_limits.pin_kind(entry)
                listed = m8_limits.listed_in_pins(entry['pin_via'] or entry['file'], pins)
                if kind == 'json':
                    self.assertTrue(listed, '%s is marked pinned but no recorded pin lists it' % (entry['pin_via'] or entry['file']))
                    if entry['pinned'] == m8_limits.PACKED_ANY:
                        self.assertTrue(any(name.startswith('packed_any_evidence') for name in listed), listed)
                elif kind is None:
                    self.assertEqual(listed, [], '%s is pinned by %s: mark the row' % (entry['file'], listed))

    def test_the_twins_edit_no_pinned_file_they_only_rebind(self):
        source = m8_limits.read(os.path.join(HERE, 'tp4_m8.py'))
        pinned = {os.path.basename(entry['file'])[:-3] for entry in m8_limits.LIMITS if entry['pinned'] and entry['file'].startswith('scripts/ci/')}
        self.assertTrue({'model_batch', 'gdn_prefix', 'gdn_records', 'gdn_device_loop_state', 'force_argmax', 'verifier_engine'} <= pinned)
        for module in pinned:
            self.assertNotRegex(source, r'open\([^)]*%s\.py' % module)
        self.assertNotIn('write_text', source)

    # -- the probes ---------------------------------------------------------------------------------------------------------
    def test_accepts_rows_take_128_rows_and_eight_users_as_they_stand(self):
        for entry in m8_limits.LIMITS:
            if entry['status'] == m8_limits.ACCEPTS:
                with self.subTest(row=entry['id']):
                    self.probe(entry, installed=False)

    def test_the_accepts_row_that_follows_a_rebound_one_follows_it(self):
        # row 35 reads the TP4 sibling's MAX_USERS, which row 34 rebinds: 4 as shipped, 8 installed
        tp4_m8.install()
        self.probe(BY_ID[35], installed=True)

    def test_rebound_rows_refuse_before_install_and_accept_after_with_the_twin_engaged(self):
        rebound = [entry for entry in m8_limits.LIMITS if entry['status'] == m8_limits.REBOUND]
        for entry in rebound:
            with self.subTest(phase='before', row=entry['id']):
                tp4_m8.uninstall()
                self.probe(entry, installed=False)
        tp4_m8.uninstall()
        self.assertGreater(tp4_m8.install(), 0)
        self.assertEqual(tp4_m8.STATS['engaged'], 1)       # the install line
        for entry in rebound:
            if entry['id'] == 18:
                continue    # counts markers from a clean slate: its own test below
            with self.subTest(phase='after', row=entry['id']):
                self.probe(entry, installed=True)

    def test_row_18_counts_its_markers_from_a_clean_slate(self):
        tp4_m8.install()
        self.probes.row_18(True)

    def test_phase2_rows_still_refuse_after_install_so_phase_2_must_flip_them(self):
        tp4_m8.install()
        for entry in m8_limits.LIMITS:
            if entry['status'] == m8_limits.PHASE2:
                with self.subTest(row=entry['id']):
                    self.probe(entry, installed=True)

    def test_install_logs_one_engaged_line_naming_every_rebound_name(self):
        lines = []
        with patch.object(tp4_m8, 'diagnostic', lines.append):
            tp4_m8.uninstall()
            tp4_m8.install()
        self.assertEqual(len(lines), 1)
        for name in [row[1] for row in tp4_m8.TWINS] + ['%s.%s' % (row[0], row[1]) for row in tp4_m8.TABLES]:
            self.assertIn(name, lines[0])
        self.assertTrue(lines[0].startswith(tp4_m8.ENGAGED + ' site=limits'))

    def test_every_twin_and_table_is_a_row_of_the_table(self):
        """A rebind with no row is an unexamined limit: every module that tp4_m8 touches appears in a rebound row's file."""
        files = {os.path.basename(entry['file'])[:-3] for entry in m8_limits.LIMITS if entry['status'] == m8_limits.REBOUND}
        modules = {row[0] for row in tp4_m8.TWINS} | {row[0] for row in tp4_m8.TABLES} | {row[0] for row in tp4_m8.SCALARS} \
            | {row[0] for row in tp4_m8.ALIASES}
        # extent_attention_replay_tp is reached through pooled_attention_replay's function object (row 20)
        self.assertEqual(modules - files, set())

    def test_uninstall_puts_every_original_back_and_install_is_idempotent(self):
        import force_argmax
        import gdn_device_loop_state
        import gdn_prefix
        import packed_shapes

        before = (force_argmax.SAMPLE_WIDTHS, gdn_prefix.ROW_WIDTHS, packed_shapes.BLOCK_ROWS,
                  gdn_device_loop_state.project_qkvzab_by_tile, packed_shapes.sequential_capture_rows)
        self.assertGreater(tp4_m8.install(), 0)
        self.assertEqual(tp4_m8.install(), 0)
        self.assertTrue(tp4_m8.installed())
        self.assertGreater(tp4_m8.uninstall(), 0)
        self.assertFalse(tp4_m8.installed())
        after = (force_argmax.SAMPLE_WIDTHS, gdn_prefix.ROW_WIDTHS, packed_shapes.BLOCK_ROWS,
                 gdn_device_loop_state.project_qkvzab_by_tile, packed_shapes.sequential_capture_rows)
        for was, now in zip(before, after):
            self.assertIs(was, now)

    def test_nothing_changes_at_the_pair(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'stay in place at the pair'):
                tp4_m8.install()

    # -- the twins that copy a body ----------------------------------------------------------------------------------------
    def test_the_two_literal_tuple_twins_are_the_originals_source_with_128_added(self):
        import target_packed_pages
        import verifier_pack

        for original, twin, old, new in ((target_packed_pages.validate_users, tp4_m8.validate_users, '(1, 2, 4, 8, 16, 32, 64):',
                                          '(1, 2, 4, 8, 16, 32, 64, 128):'),
                                         (verifier_pack.build_pack, tp4_m8.build_pack, '(1, 2, 4, 8, 16, 32, 64):',
                                          '(1, 2, 4, 8, 16, 32, 64, 128):')):
            with self.subTest(function=original.__name__):
                want = inspect.getsource(original).replace(old, new)
                have = inspect.getsource(twin)

                def code(text):
                    """The function's code lines: no docstring, no blank line, no import."""
                    text = re.sub(r'^    """(.*?)"""' + chr(10), '', text, count=1, flags=re.S | re.M)
                    return [line for line in text.split(chr(10)) if line.strip()
                            and not line.strip().startswith('from verifier_pack import')]

                self.assertEqual(code(have), code(want))

    def test_plan_unstack_is_the_originals_own_call_to_four_users_and_its_layout_generalised(self):
        import gdn_block_conv_tp

        original = gdn_block_conv_tp.plan_unstack
        twin = tp4_m8.make_plan_unstack(original)
        for users in (1, 2, 3, 4):
            with self.subTest(users=users):
                self.assertTrue(twin(users, 4128) == original(users, 4128))
        # the user-count layout IS the original's list at four users (the only count the original's literals are right for)
        self.assertTrue(tp4_m8.unstack_layout(4, 4128) == original(4, 4128))
        eight = twin(8, 4128)
        self.assertEqual(len(eight), 3600)
        self.assertTrue(eight == tp4_m8.unstack_layout(8, 4128))

    def test_the_unstack_split_covers_every_task_once_in_launches_that_fit(self):
        import gdn_block_conv_tp
        import gdn_rows_dma_tp

        tasks = tp4_m8.unstack_layout(8, 4128)
        parts = tp4_m8.split_tasks(tasks, 110)
        self.assertEqual([len(part) for part in parts], [1800, 1800])
        self.assertTrue([task for part in parts for task in part] == tasks)
        for part in parts:
            self.assertEqual(len(gdn_rows_dma_tp.distribute(part, 110)), 110)
        self.assertEqual(len(gdn_block_conv_tp.plan_unstack(4, 4128)), 1800)   # the N1 figure
        self.assertTrue(tp4_m8.split_tasks(tasks[:100], 110) == [tasks[:100]])
        self.assertEqual(tp4_m8.split_tasks([], 110), [[]])

    # -- the flag -----------------------------------------------------------------------------------------------------------
    def test_the_flag_is_strict_and_a_tp4_lever(self):
        self.assertFalse(tp4_m8.enabled({}))
        self.assertFalse(tp4_m8.enabled({tp4_m8.FLAG: '0'}))
        self.assertTrue(tp4_m8.enabled({tp4_m8.FLAG: '1', 'QWEN_FAST_TP': '4'}))
        for bad in ('', '2', 'true', ' 1', '01', 'on'):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    tp4_m8.enabled({tp4_m8.FLAG: bad, 'QWEN_FAST_TP': '4'})
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_TP=4'):
            tp4_m8.enabled({tp4_m8.FLAG: '1'})

    def test_the_flag_on_is_refused_everywhere_until_phase_2(self):
        tp4_m8.refuse_until_phase2({})
        tp4_m8.refuse_until_phase2({tp4_m8.FLAG: '0', 'QWEN_FAST_TP': '4'})
        with self.assertRaisesRegex(ValueError, 'phase 2'):
            tp4_m8.refuse_until_phase2({tp4_m8.FLAG: '1', 'QWEN_FAST_TP': '4'})
        with self.assertRaises(ValueError):
            tp4_m8.refuse_until_phase2({tp4_m8.FLAG: '1'})   # the pair: refused too

    def test_the_startup_asks_before_it_does_anything_else(self):
        text = m8_limits.read(os.path.join(HERE, 'serving_startup.py'))
        start = text.index('def start(worker):')
        self.assertLess(text.index('tp4_m8.refuse_until_phase2(os.environ)', start), text.index('recipe_paths(worker.vllm_config)', start))
        import serving_startup

        with patch.dict(os.environ, {tp4_m8.FLAG: '1'}):
            with self.assertRaisesRegex(ValueError, 'phase 2'):
                serving_startup.start(SimpleNamespace())

    def test_no_profile_sets_the_flag_in_phase_1(self):
        import json

        with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
            profiles = json.load(handle)['profiles']
        for name, profile in profiles.items():
            self.assertNotIn(tp4_m8.FLAG, profile.get('env', {}), name)

    def test_the_document_carries_the_table_the_code_generates(self):
        text = m8_limits.read(os.path.join(ROOT, 'docs', 'tp4-m8-phase1.md'))
        begin, end = '<!-- m8-limits-table:begin -->' + chr(10), chr(10) + '<!-- m8-limits-table:end -->'
        self.assertEqual(text[text.index(begin) + len(begin):text.index(end)], m8_limits.markdown(ROOT))
        self.assertNotRegex(text, r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}')
        with open(os.path.join(ROOT, 'docs', 'tp4-m8-phase1.md'), 'rb') as handle:
            self.assertNotIn(bytes([13]), handle.read())

    def test_the_module_is_stdlib_only_at_import_and_lf(self):
        text = m8_limits.read(os.path.join(HERE, 'tp4_m8.py'))
        top = [line for line in text.split(chr(10)) if line.startswith(('import ', 'from '))]
        self.assertEqual(sorted(top), ['import os', 'import sys', 'import tp_shapes'])
        for name in ('tp4_m8.py', 'm8_limits.py', 'test_tp4_m8_limits.py'):
            with open(os.path.join(HERE, name), 'rb') as handle:
                self.assertNotIn(b'\r', handle.read(), name)


if __name__ == '__main__':
    unittest.main()
