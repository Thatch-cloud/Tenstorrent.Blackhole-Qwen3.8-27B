"""Verify-trace T1, wave 2: each image cut exact, and the flag off the path it replaced.

  #10 mask once      the per-user replay readers' masks refreshed once per forward: every
                     SDPA call reads the same mask bytes as with a refresh before each call.
  #3  direct carry   the batched recurrence reads each user's carry in place: the launch sees
                     the same values; the stale entries and native slot 0 have no reader
                     (the commit DMA reads an entry only at prefix 0, which never publishes).
  #12 coalescing     the batched recurrence program on rectangle ranges with one descriptor
                     per role: every core runs the same kernel with the same compile-time and
                     runtime arguments.
  #8a shard argmax   per-chip argmax + max and a host combine equal the first-occurrence
                     argmax over the whole row (what the pinned sampler returns), including
                     ties inside and across shards, signed zeros and NaN; the packed block
                     reads the combined ids, and the audit compares them with the sampler's.

The flag-off tests hold every path to the calls it made before T1 existed. The device
properties the fakes cannot show (ArgMax's first occurrence, ttnn.max's value) are G0
(verify_t1_device_compare.py) and the in-model audit (QWEN_FAST_VERIFY_T1_AUDIT=1).
"""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import gdn_user_batch as batch
import packed_verifier
import verify_t1_device_compare as g0
import verify_trace_t1 as t1
from test_gdn_packed_segments import PackedFixture
from test_gdn_records import M3_SEGMENTS, packed_block
from test_gdn_user_batch import FakeTTNN as ProgramTTNN, KERNELS, mesh, user_inputs
# The fixture's own address counter, so the per-chip outputs never share an address with a
# buffer the fake block allocated.
from test_packed_verifier import FakeModelBatch, FakeShard, FourUserFixture, TAPS, _addresses

HERE = Path(__file__).parent
ON = {'QWEN_FAST_VERIFY_T1': '1'}


class FlagTests(unittest.TestCase):
    def test_the_flag_is_on_only_for_exactly_one(self):
        for environ, expected in (({}, False), ({'QWEN_FAST_VERIFY_T1': '0'}, False),
                                  ({'QWEN_FAST_VERIFY_T1': 'true'}, False), (ON, True)):
            with patch.dict('os.environ', environ, clear=True):
                self.assertIs(t1.enabled(), expected)

    def test_the_audit_needs_the_flag(self):
        for environ, expected in (({'QWEN_FAST_VERIFY_T1_AUDIT': '1'}, False),
                                  (dict(ON, QWEN_FAST_VERIFY_T1_AUDIT='1'), True), (ON, False)):
            with patch.dict('os.environ', environ, clear=True):
                self.assertIs(t1.audit_enabled(), expected)

    def test_counts_are_taken_once(self):
        t1.take()
        t1.note('direct_carry')
        t1.note('direct_carry', 2)
        self.assertEqual(t1.take(), {'direct_carry': 3})
        self.assertEqual(t1.take(), {})

    def test_the_skip_list_is_read_only_while_the_flag_is_on(self):
        skip = {'QWEN_FAST_VERIFY_T1_SKIP': 'mask_once, last_carry'}
        with patch.dict('os.environ', dict(ON, **skip), clear=True):
            self.assertEqual(t1.skipped(), {'mask_once', 'last_carry'})
            self.assertEqual([name for name in t1.CUTS if t1.cut(name)],
                             ['matmul_configs', 'direct_carry', 'coalesce', 'shard_argmax'])
        with patch.dict('os.environ', ON, clear=True):
            self.assertEqual(t1.skipped(), frozenset())
            self.assertTrue(all(t1.cut(name) for name in t1.CUTS))
        # flag off: nothing is read beyond the flag, not even a bad skip list
        with patch.dict('os.environ', {'QWEN_FAST_VERIFY_T1_SKIP': 'bogus'}, clear=True):
            self.assertEqual(t1.skipped(), frozenset())
            self.assertFalse(any(t1.cut(name) for name in t1.CUTS))

    def test_an_unknown_cut_name_raises(self):
        with patch.dict('os.environ', dict(ON, QWEN_FAST_VERIFY_T1_SKIP='mask_once,shard'), clear=True):
            with self.assertRaisesRegex(ValueError, 'names no cut: shard'):
                t1.skipped()
        with self.assertRaises(ValueError):
            t1.cut('masks')
        self.assertTrue(set(t1.WAVE2_CUTS) < set(t1.CUTS))

    def test_the_module_imports_only_the_standard_library_at_import(self):
        tree = ast.parse((HERE / 'verify_trace_t1.py').read_text(encoding='utf-8'))
        imported = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        imported += [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
        self.assertEqual(imported, ['os'])

    def test_it_reaches_the_image_with_every_module_that_imports_it(self):
        from test_serving_image_copy_closure import copied_modules, dockerfile_text
        shipped = copied_modules(dockerfile_text())
        for name in ('verify_trace_t1.py', 'packed_verifier.py', 'gdn_device_loop_state.py', 'gdn_user_batch.py'):
            with self.subTest(module=name):
                self.assertIn(name, shipped)


# ---------------------------------------------------------------------------------------
# #10: one mask refresh per forward.
# ---------------------------------------------------------------------------------------

class SharedMaskTests(FourUserFixture):
    """The real per-user replay readers the packed block builds, driven through sixteen
    attention layers with a mask refresh that writes a function of the positions word."""

    def setUp(self):
        super().setUp()
        patcher = patch('attention_replay.prepare', side_effect=lambda mesh, positions, mask, **geometry: geometry)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def refresh(positions, mask, program):
        mask.value = torch.tensor([int(positions.value[0]), program['offset'], program['rows'], program['batches']])

    def forward(self, packed, shared):
        """Sixteen layers, each calling every user's reader once in segment order (the packed
        reader's own order); returns every mask each SDPA call was handed, call by call."""
        seen = []

        def execute(mesh, operations, query, keys, values, metadata, owned, *, scale, memory_config):
            seen.append([entry[2].value.clone() for entry in metadata])
            return self.ttnn.allocate((1, 16, 12, 256))

        query = self.ttnn.allocate((1, 16, 12, 256))
        keys, values = self.ttnn.allocate((1,)), self.ttnn.allocate((2,))
        with patch('attention_replay.refresh_mask', side_effect=self.refresh), \
                patch('attention_replay.execute', side_effect=execute):
            scope = packed.shared_masks(16) if shared else nullcontext()
            before = packed.refresh_calls
            with scope:
                for layer in range(16):
                    for reader in packed.readers:
                        reader(query, keys, values, scale=0.0625, memory_config='dram')
        return seen, packed.refresh_calls - before

    def test_every_sdpa_call_reads_the_same_mask_bytes_either_way(self):
        block = self.build()
        block.stage_packed_inputs(self.four())
        packed = block.fixture.replay_reader
        per_call, per_call_refreshes = self.forward(packed, shared=False)
        once, once_refreshes = self.forward(packed, shared=True)
        self.assertEqual(len(per_call), 64)
        self.assertEqual(len(once), 64)
        for layer, (mine, theirs) in enumerate(zip(once, per_call, strict=True)):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(mine, theirs, strict=True)), layer)
        # each user's mask is its own word's, not its neighbour's
        self.assertEqual([masks[0][0].item() for masks in once[:4]], list(self.POSITIONS))
        entries = len(packed.metadata)
        self.assertEqual((per_call_refreshes, once_refreshes), (16 * entries, entries))

    def test_a_shared_mask_forward_refuses_a_restage_of_the_word_and_a_short_budget(self):
        """The reader's own guards around the one refresh: the word it was computed from cannot
        be restaged inside the forward, and a forward that skips a layer's call fails loudly."""
        block = self.build()
        block.stage_packed_inputs(self.four())
        first, second = block.fixture.replay_reader.readers[:2]
        with patch('attention_replay.refresh_mask'):
            with self.assertRaisesRegex(RuntimeError, 'shared-mask forward'):
                with first.shared_masks(16):
                    first.stage(4100)
            with self.assertRaisesRegex(AssertionError, 'exact attention call budget'):
                with second.shared_masks(16):
                    pass
        self.assertTrue(first.failed and second.failed)


# ---------------------------------------------------------------------------------------
# #3: the batched recurrence reads each carry in place.
# ---------------------------------------------------------------------------------------

class DirectCarryTests(PackedFixture):
    spans = M3_SEGMENTS
    slots = [['%s%d' % (user, index) for index in range(5)] for user in 'ABCD']
    checkpoints = [['ck' + user] for user in 'ABCD']

    def run_block(self, environ):
        """One deferred packed decode over value-carrying fakes: device memory maps a buffer
        name to what it holds; copy_compact / restore / save move values, never compute."""
        memory = {name: 'value(%s)' % name for slot in self.slots for name in slot}
        native = {'row0': ['stale-native-%d' % index for index in range(5)]}
        with patch.dict('os.environ', environ, clear=True):
            state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=True)

            def restore(source):
                calls.append(('restore', source[0]))
                native['row0'] = [memory[name] for name in source]

            def save(destination):
                calls.append(('save', destination[0]))
                for name, value in zip(destination, native['row0'], strict=True):
                    memory[name] = value

            def copy_compact(source, destination):
                calls.append(('copy_compact', source[0], destination[0]))
                for name, value in zip(destination, [memory[item] for item in source], strict=True):
                    memory[name] = value

            def launch(mesh, users, *args, **options):
                read = [(piece.name, memory[initial], tuple(memory[name] for name in history))
                        for piece, initial, history in users]
                calls.append(('launch', read, [(initial, tuple(history)) for piece, initial, history in users]))
                return [dict(output=SimpleNamespace(shape=(1, piece.shape[1], 3072), name='out%d' % index),
                             states=SimpleNamespace(shape=(piece.shape[1], 24, 128, 128)),
                             conv_prefixes=[None] * piece.shape[1], owned=[], packed_conv_states=[],
                             packed_checkpoints=True, deferred_conv_publication=True, norm_batch=False,
                             prefix_zero_reuse=False, user_batched=True)
                        for index, (piece, initial, history) in enumerate(users)]

            active.restore.side_effect = restore
            active.save.side_effect = save
            packed = SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1')
            t1.take()
            with patch('gdn_device_loop_state.run_user_batched_projected', side_effect=launch), \
                    patch('gdn_device_loop_state.copy_compact', side_effect=copy_compact), \
                    patch('gdn_device_loop_state.copy_compact_batch') as batched, \
                    patch('gdn_device_loop_state.release_owned'):
                result = state.decode(packed, self.checkpoints, [0] * 4, segments=self.spans,
                                      slots=self.slots, deferred=True)
        launch_call = next(entry for entry in calls if entry[0] == 'launch')
        moves = [entry for entry in calls if entry[0] in ('copy_compact', 'restore', 'save')]
        return dict(state=state, result=result, read=launch_call[1], bound=launch_call[2], moves=moves,
                    native=native['row0'], memory=memory, counts=t1.take(), batched=batched)

    def test_flag_off_the_state_moves_are_exactly_todays(self):
        run = self.run_block({})
        entries = run['state'].segment_entries
        self.assertEqual(run['moves'], [('copy_compact', 'A0', entries[0][0]), ('copy_compact', 'B0', entries[1][0]),
                                        ('copy_compact', 'C0', entries[2][0]), ('restore', 'D0'),
                                        ('save', entries[3][0])])
        self.assertEqual(run['bound'], [(entry[0], tuple(entry[1:])) for entry in entries])
        self.assertEqual(run['native'], [run['memory'][name] for name in self.slots[3]])
        self.assertEqual(run['counts'], {})

    def test_flag_on_no_state_moves_and_the_launch_binds_every_carry(self):
        run = self.run_block(ON)
        self.assertEqual(run['moves'], [])
        run['batched'].assert_not_called()
        self.assertEqual(run['bound'], [(slot[0], tuple(slot[1:])) for slot in self.slots])
        self.assertEqual(run['counts'], {'direct_carry': 1, 'last_carry': 1})
        # the entries stay allocated and untouched (still what the commit DMA's list names)
        self.assertEqual([list(entry) for entry in run['state'].segment_entries],
                         [['E%d.%d' % (user, index) for index in range(5)] for user in range(4)])

    def test_skipping_last_carry_keeps_only_the_last_users_restore_and_save(self):
        off = self.run_block({})
        run = self.run_block(dict(ON, QWEN_FAST_VERIFY_T1_SKIP='last_carry'))
        entries = run['state'].segment_entries
        self.assertEqual(run['moves'], [('restore', 'D0'), ('save', entries[3][0])])
        self.assertEqual(run['bound'], [(slot[0], tuple(slot[1:])) for slot in self.slots[:3]]
                         + [(entries[3][0], tuple(entries[3][1:]))])
        self.assertEqual(run['native'], off['native'], 'native slot 0 left holding the last user, as today')
        self.assertEqual(run['read'], off['read'])
        self.assertEqual(run['counts'], {'direct_carry': 1})

    def test_skipping_direct_carry_is_todays_moves(self):
        off = self.run_block({})
        for skip in ('direct_carry', 'direct_carry,last_carry'):
            with self.subTest(skip=skip):
                run = self.run_block(dict(ON, QWEN_FAST_VERIFY_T1_SKIP=skip))
                entries = run['state'].segment_entries
                self.assertEqual(run['moves'], [('copy_compact', 'A0', entries[0][0]),
                                                ('copy_compact', 'B0', entries[1][0]),
                                                ('copy_compact', 'C0', entries[2][0]), ('restore', 'D0'),
                                                ('save', entries[3][0])])
                self.assertEqual(run['read'], off['read'])
                self.assertEqual(run['counts'], {})

    def test_the_launch_reads_the_same_values_either_way(self):
        off, on = self.run_block({}), self.run_block(ON)
        self.assertEqual(on['read'], off['read'])
        self.assertEqual(on['read'][3][1], 'value(D0)')
        self.assertEqual(on['result']['segments'], off['result']['segments'])
        self.assertEqual([piece['states'].shape for piece in on['result']['segment_results']],
                         [piece['states'].shape for piece in off['result']['segment_results']])

    def test_only_native_slot_0_differs_and_the_first_accepted_commit_rewrites_it(self):
        """Flag off the trace leaves the last user in native slot 0; on, it is left as it was.
        Emulating gdn_commit_dma.cpp: every commit at prefix >= 1 writes all of slot 0 (and the
        carry) from that user's history, never reading the entry or slot 0, so after any round
        in which some user accepted a token the two are identical; a round of prefix-0 commits
        writes nothing, and no reader trusts slot 0 then (verifier_engine.note_packed_step)."""
        off, on = self.run_block({}), self.run_block(ON)
        self.assertNotEqual(off['native'], on['native'])

        def commit(native, carry, entry, history, prefix):
            source = entry if prefix == 0 else history[prefix - 1]
            return list(source), list(source)

        history = [['h%d.%d' % (token, index) for index in range(5)] for token in range(16)]
        for prefix in range(1, 17):
            after_off = commit(off['native'], None, ['entry-copy'] * 5, history, prefix)
            after_on = commit(on['native'], None, ['stale-entry'] * 5, history, prefix)
            self.assertEqual(after_off, after_on)

    def test_the_commit_dma_reads_an_entry_only_at_prefix_zero_and_never_reads_native(self):
        lines = [line.strip() for line in (HERE / 'gdn_commit_dma.cpp').read_text(encoding='utf-8').splitlines()]
        reads = [index for index, line in enumerate(lines) if 'noc_async_read_tile(' in line]
        entry_reads = [index for index in reads if 'entry' in lines[index]]
        self.assertEqual(len(entry_reads), 2, 'one recurrent and one convolution entry read')
        for index in entry_reads:
            self.assertEqual(lines[index - 1], 'if (prefix == 0) {')
        self.assertFalse([index for index in reads if 'native' in lines[index]])
        self.assertTrue(any('noc_async_write_tile(page, native_rec' in line for line in lines))

    def test_prefix_zero_never_publishes_and_the_block_captures_no_prefix_zero_trace(self):
        block = packed_block(segments=M3_SEGMENTS)
        publication = Mock()
        with patch('gdn_commit_dma.publish') as publish, patch('gdn_records.restore_prefix') as restore:
            for segment in range(4):
                block.commit_user(segment, 0, dma=True, publication=publication)
        publication.assert_not_called()
        publish.assert_not_called()
        restore.assert_not_called()
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertIn('for prefix in range(1, shape.rows_per_user + 1)}', source)

    def test_the_per_user_path_is_untouched_by_the_flag(self):
        """Without the batched launch (QWEN_FAST_GDN_USER_BATCH off) nothing changes."""
        calls_by_flag = []
        for environ in ({}, ON):
            with patch.dict('os.environ', environ, clear=True):
                state, operations, layer, active, calls = self.build(commit_only=True, users=4, user_batch=False)
                with patch('gdn_device_loop_state.run_batched_projected', side_effect=self.recurrence(calls)), \
                        patch('gdn_device_loop_state.copy_compact',
                              side_effect=lambda source, destination: calls.append(('copy_compact', source[0]))), \
                        patch('gdn_device_loop_state.restore_prefix'), patch('gdn_device_loop_state.release_owned'):
                    state.decode(SimpleNamespace(shape=(1, 64, 5120), name='packed', memory_config=lambda: 'l1'),
                                 self.checkpoints, [0] * 4, segments=self.spans, slots=self.slots, deferred=True)
            calls_by_flag.append(calls)
        self.assertEqual(calls_by_flag[0], calls_by_flag[1])


# ---------------------------------------------------------------------------------------
# #12: rectangle ranges and one descriptor per role.
# ---------------------------------------------------------------------------------------

class InterleavedTTNN(ProgramTTNN):
    """Accessor compile args as ttnn gives them for interleaved buffers: no address."""

    @staticmethod
    def TensorAccessorArgs(value):
        return SimpleNamespace(get_compile_time_args=lambda: [7, value.memory_config()])


def expand(ranges):
    return [(x, y) for (x0, y0), (x1, y1) in ranges for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def per_core(program):
    """{(chip, core, kernel source): (compile args, config, runtime args)} and the per-chip CBs
    (each CB's core ranges expanded to the cores they cover)."""
    cores, cbs = {}, {}
    for chip, descriptor in program.items():
        cbs[chip] = [(kind, size, sorted(expand(ranges)), formats) for kind, size, ranges, formats in descriptor.cbs]
        for kernel in descriptor.kernels:
            covered = expand(kernel.core_ranges)
            runtime = {(x, y): args for x, y, args in kernel.runtime_args.flattened()}
            if sorted(covered) != sorted(runtime):
                raise AssertionError('a descriptor has runtime args for cores it does not cover, or the reverse')
            for core in covered:
                key = (chip, core, kernel.kernel_source)
                if key in cores:
                    raise AssertionError('two descriptors of one role on one core')
                cores[key] = (tuple(kernel.compile_time_args), kernel.config, runtime[core])
    return cores, cbs


class CoalescingTests(unittest.TestCase):
    def build(self, fake, environ, rows=(16, 16, 16, 16)):
        groups = [user_inputs(index, width) for index, width in enumerate(rows)]
        with patch.dict('os.environ', environ, clear=True), patch('gdn_multitoken.validate_handoff_runtime'):
            t1.take()
            batch.execute(mesh(), groups, KERNELS, fake, output_memory=fake.L1_MEMORY_CONFIG)
            counts = t1.take()
        return fake.launches[-1][1], counts

    def test_rectangles_cover_exactly_the_points(self):
        shares = batch.core_shares(11, 10, 4)
        self.assertEqual(t1.rectangles(shares[0]), [(0, 0, 1, 9), (2, 0, 2, 3)])
        self.assertEqual(t1.rectangles(shares[1]), [(2, 4, 2, 9), (3, 0, 3, 9), (4, 0, 4, 7)])
        self.assertEqual(t1.rectangles([point for share in shares for point in share]), [(0, 0, 8, 9), (9, 0, 9, 5)])
        self.assertEqual(t1.rectangles([(0, 0), (0, 2), (1, 0), (1, 2)]), [(0, 0, 1, 0), (0, 2, 1, 2)])
        for points in ([], [(0, 0), (0, 0)]):
            with self.assertRaises(ValueError):
                t1.rectangles(points)

    def test_one_descriptor_per_role_and_every_core_unchanged(self):
        before, unused = self.build(InterleavedTTNN(), {})
        after, counts = self.build(InterleavedTTNN(), ON)
        self.assertEqual(counts, {'coalesced': 1})
        self.assertEqual(per_core(after), per_core(before))
        for chip, descriptor in after.items():
            self.assertEqual(len(before[chip].kernels), 12)
            self.assertEqual([kernel.kernel_source for kernel in descriptor.kernels],
                             ['READER-SOURCE', 'WRITER-SOURCE', 'COMPUTE-SOURCE'])
            for kernel in descriptor.kernels:
                self.assertEqual(kernel.core_ranges, (((0, 0), (8, 9)), ((9, 0), (9, 5))))
                self.assertEqual(len(expand(kernel.core_ranges)), 96)

    def test_users_that_compile_differently_keep_their_own_descriptors_on_rectangles(self):
        for fake, rows in ((ProgramTTNN(), (16, 16, 16, 16)), (InterleavedTTNN(), (16, 8, 16, 16))):
            with self.subTest(rows=rows, fake=type(fake).__name__):
                before, unused = self.build(type(fake)(), {}, rows)
                after, counts = self.build(fake, ON, rows)
                self.assertEqual(counts, {'coalesce_fallback': 1})
                self.assertEqual(per_core(after), per_core(before))
                for chip, descriptor in after.items():
                    self.assertEqual(len(descriptor.kernels), 12)
                    self.assertTrue(all(len(kernel.core_ranges) <= 3 for kernel in descriptor.kernels))

    def test_flag_off_the_program_is_todays(self):
        before, counts = self.build(InterleavedTTNN(), {})
        self.assertEqual(counts, {})
        for descriptor in before.values():
            self.assertTrue(all(len(kernel.core_ranges) == 24 for kernel in descriptor.kernels))

    def test_skipping_coalesce_is_todays_program(self):
        before, unused = self.build(InterleavedTTNN(), {})
        skipped, counts = self.build(InterleavedTTNN(), dict(ON, QWEN_FAST_VERIFY_T1_SKIP='coalesce'))
        self.assertEqual(counts, {})
        for chip, descriptor in skipped.items():
            self.assertEqual([(kernel.kernel_source, kernel.core_ranges, tuple(kernel.compile_time_args),
                               sorted(kernel.runtime_args.flattened())) for kernel in descriptor.kernels],
                             [(kernel.kernel_source, kernel.core_ranges, tuple(kernel.compile_time_args),
                               sorted(kernel.runtime_args.flattened())) for kernel in before[chip].kernels])


# ---------------------------------------------------------------------------------------
# #8a: per-shard argmax and the host combine.
# ---------------------------------------------------------------------------------------

def shard_emulation(logits, width):
    """What each chip reports: its shard's first-occurrence argmax (torch.argmax) and max."""
    rows = logits.shape[-2]
    flat = logits.reshape(rows, -1)
    ids, values = [], []
    for shard in range(2):
        part = flat[:, shard * width:(shard + 1) * width]
        ids.append(part.float().argmax(dim=-1).to(torch.int32))
        values.append(part.amax(dim=-1))
    return ids, values


class CombineTests(unittest.TestCase):
    def test_every_g0_case_at_the_real_width_matches_the_first_occurrence_over_the_whole_row(self):
        for kind in g0.ARGMAX_KINDS:
            with self.subTest(kind=kind):
                logits = g0.argmax_case(kind)
                ids, values = shard_emulation(logits, t1.SHARD_WIDTH)
                combined = t1.combine_shards(ids, values)
                self.assertTrue(torch.equal(combined, g0.reference_ids(logits)))
                self.assertTrue(torch.equal(combined, logits.reshape(64, -1).argmax(dim=-1)))

    def test_ties_signed_zeros_and_nan_on_a_small_vocabulary(self):
        generator = torch.Generator().manual_seed(7)
        palette = torch.tensor([-1.0, -0.0, 0.0, 0.5, 1.0, 2.0, float('nan'), float('-inf'), float('inf')])
        for trial in range(40):
            width = 1 + trial % 9
            choice = torch.randint(0, len(palette), (256, 2 * width), generator=generator)
            logits = palette[choice].to(torch.bfloat16).reshape(1, 1, 256, 2 * width)
            ids, values = shard_emulation(logits, width)
            combined = t1.combine_shards(ids, values, shard_width=width)
            self.assertTrue(torch.equal(combined, logits.reshape(256, -1).argmax(dim=-1)), trial)

    def test_shard_one_wins_only_when_strictly_greater(self):
        ids = [torch.tensor([3, 3, 3]), torch.tensor([4, 4, 4])]
        values = [torch.tensor([1.0, 1.0, 2.0]).bfloat16(), torch.tensor([1.0078125, 1.0, 1.0]).bfloat16()]
        self.assertEqual(t1.combine_shards(ids, values, shard_width=10).tolist(), [14, 3, 3])

    def test_malformed_shard_reports_are_refused(self):
        good = [torch.tensor([0, 1]), torch.tensor([1, 0])], [torch.zeros(2), torch.zeros(2)]
        self.assertEqual(t1.combine_shards(*good, shard_width=2).tolist(), [0, 1])
        for ids, values in (([torch.tensor([0, 2]), torch.tensor([0, 0])], good[1]),
                            ([torch.tensor([0, -1]), torch.tensor([0, 0])], good[1]),
                            ([torch.tensor([0]), torch.tensor([0, 0])], good[1]),
                            (good[0][:1], good[1]), ([torch.tensor([])] * 2, [torch.tensor([])] * 2)):
            with self.assertRaises(ValueError):
                t1.combine_shards(ids, values, shard_width=2)


def greedy_sampler(**changes):
    options = dict(force_argmax_sampling=True, vocab_size=248320, padded_vocab_size=248320)
    sampler = dict(_penalties_active=False, _log_probs_active=False, seed=False)
    for key, value in changes.items():
        (options if key in options else sampler)[key] = value
    return SimpleNamespace(tt_sampling=SimpleNamespace(**{k: v for k, v in options.items() if v is not None}),
                           _penalties_active=sampler['_penalties_active'],
                           _log_probs_active=sampler['_log_probs_active'],
                           seed_manager=SimpleNamespace(has_active_request_seed=lambda: sampler['seed']))


class ShardSamplingTests(unittest.TestCase):
    def test_only_the_plain_greedy_sampler_over_the_unpadded_vocabulary_qualifies(self):
        self.assertIsNone(t1.shard_sampling_problem(greedy_sampler()))
        for changes in (dict(force_argmax_sampling=False), dict(vocab_size=248320 + 64),
                        dict(padded_vocab_size=248384), dict(_penalties_active=True),
                        dict(_log_probs_active=True), dict(seed=True)):
            with self.subTest(changes=changes):
                self.assertIsNotNone(t1.shard_sampling_problem(greedy_sampler(**changes)))
        self.assertIsNotNone(t1.shard_sampling_problem('sampler'))

    def fake(self):
        calls = []
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', ROW_MAJOR_LAYOUT='row_major', TILE_LAYOUT='tile',
                                     bfloat16='bf16')
        operations.to_layout = Mock(side_effect=lambda value, layout, memory_config=None: (
            calls.append(('to_layout', layout, memory_config)) or SimpleNamespace(name='row-major', shape=value.shape)))
        operations.argmax = Mock(side_effect=lambda value, **options: (
            calls.append(('argmax', value.name, tuple(sorted(options.items())))) or 'ids'))
        operations.max = Mock(side_effect=lambda value, **options: (
            calls.append(('max', value.name, tuple(sorted(options.items())))) or 'values'))
        operations.deallocate = Mock(side_effect=lambda value: calls.append(('free', value.name)))
        return operations, calls

    def test_each_chip_argmaxes_its_row_major_shard_and_maxes_the_tiled_one(self):
        operations, calls = self.fake()
        logits = SimpleNamespace(name='logits', shape=(1, 1, 64, 124160), dtype='bf16', layout='tile')
        self.assertEqual(t1.sample_shards(operations, logits, 64), ('ids', 'values'))
        self.assertEqual(calls, [
            ('to_layout', 'row_major', 'dram'),
            ('argmax', 'row-major', (('dim', 3), ('keepdim', True), ('memory_config', 'dram'))),
            ('max', 'logits', (('dim', 3), ('keepdim', True), ('memory_config', 'dram'))),
            ('free', 'row-major')])

    def test_anything_but_the_pre_gather_shard_is_refused_before_any_op(self):
        operations, calls = self.fake()
        for shape in ((1, 1, 64, 248320), (1, 1, 32, 124160), (64, 124160)):
            with self.assertRaisesRegex(ValueError, 'pre-gather'):
                t1.sample_shards(operations, SimpleNamespace(name='logits', shape=shape, dtype='bf16', layout='tile'),
                                 64)
        self.assertEqual(calls, [])

    def test_only_bf16_tile_logits_are_sampled(self):
        """A block-float shard's max would come back sharing an exponent with its padding."""
        operations, calls = self.fake()
        for dtype, layout in (('bf8_b', 'tile'), ('bf16', 'row_major'), (None, None)):
            with self.subTest(dtype=dtype, layout=layout):
                with self.assertRaisesRegex(ValueError, 'bf16 TILE'):
                    t1.sample_shards(operations, SimpleNamespace(name='logits', shape=(1, 1, 64, 124160),
                                                                 dtype=dtype, layout=layout), 64)
        self.assertEqual(calls, [])

    def test_a_failing_max_frees_the_ids_and_the_temporary(self):
        operations, calls = self.fake()
        operations.max = Mock(side_effect=RuntimeError('device'))
        operations.argmax = Mock(return_value=SimpleNamespace(name='ids'))
        with self.assertRaises(RuntimeError):
            t1.sample_shards(operations, SimpleNamespace(name='logits', shape=(1, 1, 64, 124160), dtype='bf16',
                                                         layout='tile'), 64)
        self.assertEqual([entry for entry in calls if entry[0] == 'free'], [('free', 'ids'), ('free', 'row-major')])

    def test_the_audit_logs_every_round_and_raises_on_the_first_difference(self):
        lines = []
        t1._AUDIT.update(rounds=0, rows=0)
        with patch.object(t1, 'log_line', side_effect=lines.append):
            self.assertTrue(t1.audit_round([1, 2, 3], [1, 2, 3]))
            self.assertTrue(t1.audit_round([4], [4]))
            with self.assertRaises(AssertionError):
                t1.audit_round([1, 9, 3], [1, 2, 3])
        self.assertEqual(lines[:2], [t1.AUDIT_MARKER + ' 1 exact=True rows=3', t1.AUDIT_MARKER + ' 2 exact=True rows=4'])
        self.assertTrue(lines[2].startswith(t1.AUDIT_MISMATCH + ' round=3 rows=[1] shard=[9] sampler=[2]'))


class DeviceComparePlanTests(unittest.TestCase):
    """G0's comparisons are the configs the graft builds, before and after section A2."""

    def test_the_plan_builds_the_graft_configs(self):
        from test_verify_trace_t1_graft import GRAFTED, build_configs, create_matmul_1d_decode_progcfg, without_t1
        before, unused = build_configs(without_t1(GRAFTED), {})
        after, unused = build_configs(GRAFTED, ON)
        for entry in g0.plan(11):
            silu = dict(fused_activation='silu') if entry['silu'] else {}
            name = entry['name'] + '_decode_1d_progcfg_64'
            for side, args in (('before', before), ('after', after)):
                with self.subTest(projection=entry['name'], side=side):
                    built = create_matmul_1d_decode_progcfg(entry['m'], entry['k'], entry['n'],
                                                            **dict(entry[side], **silu))
                    self.assertEqual(vars(built), vars(getattr(args, name)))
                    self.assertEqual(g0.config_shape(built), g0.PLAN_SHAPES[entry['name']][side],
                                     'the shape G0 insists on is the one the graft builds')

    def test_the_grid_must_be_the_models(self):
        self.assertEqual(g0.model_grid_w(SimpleNamespace(x=11, y=10)), 11)
        for x, y in ((13, 10), (8, 8), (11, 7)):
            with self.subTest(grid=(x, y)):
                with self.assertRaisesRegex(ValueError, 'decode_grid_w 11'):
                    g0.model_grid_w(SimpleNamespace(x=x, y=y))

    def test_a_coordinate_grid_reads_like_a_tuple(self):
        config = SimpleNamespace(compute_with_storage_grid_size=SimpleNamespace(x=11, y=4), per_core_N=6,
                                 out_subblock_h=1, out_subblock_w=3)
        self.assertEqual(g0.config_shape(config), ((11, 4), 6, (1, 3)))

    def test_the_compute_configs_are_the_call_sites(self):
        made = []
        ttnn = SimpleNamespace(MathFidelity=SimpleNamespace(LoFi='lofi'),
                               WormholeComputeKernelConfig=lambda **options: made.append(options) or 'lofi-config')
        configs = g0.compute_configs(ttnn, SimpleNamespace(COMPUTE_HIFI2='hifi2-config'))
        self.assertEqual(configs, dict(hifi2='hifi2-config', lofi_decode='lofi-config'))
        self.assertEqual(made, [dict(math_fidelity='lofi', math_approx_mode=True, fp32_dest_acc_en=True,
                                     packer_l1_acc=True)])
        self.assertEqual({entry['name']: (entry['compute'], entry['input']) for entry in g0.plan()},
                         dict(attn_qkv=('hifi2', 'dram'), mlp_w1=('lofi_decode', 'l1')))

    def test_signed_zeros_are_one_value_and_nothing_else_is(self):
        zero, negative = torch.tensor([0.0, 1.5]).bfloat16(), torch.tensor([-0.0, 1.5]).bfloat16()
        self.assertTrue(g0.same_values(zero, negative))
        self.assertTrue(g0.same_values(negative, torch.tensor([-0.0, 1.5])))
        self.assertFalse(g0.same_values(torch.tensor([0.0, 1.5]), torch.tensor([0.0, 1.5078125])))
        self.assertFalse(g0.same_values(torch.tensor([1.0]), torch.tensor([1.0, 1.0])))


class RigRunnerTests(unittest.TestCase):
    """The device checks can run on an image built before T1: every checkout module they import
    is mounted, and the explicit allocation crosses into the container."""

    @staticmethod
    def local_imports(name):
        tree = ast.parse((HERE / name).read_text(encoding='utf-8'))
        names = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        names += [node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module]
        return {module + '.py' for module in names if (HERE / (module + '.py')).is_file()}

    def test_the_user_batch_runner_mounts_every_module_its_scripts_import(self):
        text = (HERE / 'gdn-user-batch-rig.sh').read_text(encoding='utf-8')
        loop = next(line for line in text.splitlines() if line.startswith('for name in gdn_user_batch_device_test.py'))
        mounted = set(loop.split(' in ', 1)[1].split(';', 1)[0].split())
        needed = self.local_imports('gdn_user_batch_device_test.py') | self.local_imports('gdn_user_batch.py')
        self.assertIn('verify_trace_t1.py', needed)
        self.assertLessEqual(needed, mounted)
        self.assertIn('${GDN_USER_BATCH_VERIFY_T1:+-e QWEN_FAST_VERIFY_T1=1} ' + chr(92), text)

    def test_the_g0_runner_mounts_the_checkout_and_passes_the_allocation_inside(self):
        text = (HERE / 'verify-t1-g0-rig.sh').read_text(encoding='utf-8')
        self.assertIn('dst=/checkout/scripts/ci,readonly', text)
        self.assertIn('-e QWEN_HARDWARE_TESTS=1 -e QWEN_CARDS_ALLOCATED=1', text)
        self.assertIn('/checkout/scripts/ci/verify_t1_device_compare.py', text)
        checked = set(next(line for line in text.splitlines() if line.startswith('for name in verify_t1_device_compare.py'))
                      .split(' in ', 1)[1].split(';', 1)[0].split())
        needed = self.local_imports('verify_t1_device_compare.py') | self.local_imports('lever_n_m3native_patch.py')
        self.assertLessEqual(needed, checked)
        self.assertIn('Refusing G0: running container can reach a card', text)


# ---------------------------------------------------------------------------------------
# The packed block: masks shared, shards sampled, the marker, the audit.
# ---------------------------------------------------------------------------------------

class MaskAwareModelBatch(FakeModelBatch):
    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.attention_mask_once = bool(options.get('attention_mask_once') and options.get('attention_replay'))


def chip_tensor(values, dtype):
    tensor = SimpleNamespace(shape=(1, 1, 64, 1), dtype=dtype, layout='row_major')
    tensor.shards = [FakeShard(SimpleNamespace(value=value), next(_addresses)) for value in values]
    return tensor


class PackedBlockTests(FourUserFixture):
    ROWS = torch.arange(64)
    # shard 1 strictly greater on even rows, tied on odd rows (the tie keeps shard 0)
    CHIP_IDS = (ROWS.to(torch.int32), (ROWS + 500).to(torch.int32))
    CHIP_VALUES = (torch.ones(64).bfloat16(), torch.where(ROWS % 2 == 0, 2.0, 1.0).bfloat16())
    COMBINED = torch.where(ROWS % 2 == 0, ROWS + 500 + 124160, ROWS).tolist()

    def setUp(self):
        super().setUp()
        for target, value in (('ModelBatch', MaskAwareModelBatch),):
            patcher = patch.object(packed_verifier, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.lines = []
        patcher = patch.object(packed_verifier, 'diagnostic', Mock(side_effect=self.lines.append))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.shard_calls = []

        def sample_shards(operations, logits, rows):
            self.shard_calls.append((tuple(logits.shape), rows))
            return chip_tensor(self.CHIP_IDS, 'uint32'), chip_tensor(self.CHIP_VALUES, 'bf16')

        patcher = patch.object(t1, 'sample_shards', side_effect=sample_shards)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build_with(self, environ, sampler=None):
        with patch.dict('os.environ', environ, clear=True):
            return packed_verifier.PackedVerifierEngine(self.ttnn, self.model, self.helpers,
                greedy_sampler() if sampler is None else sampler, pool=self.pool, shared_weights=self.weights,
                shape=self.shape(), feature_taps=TAPS)

    def verify(self, block, environ):
        with patch.dict('os.environ', environ, clear=True):
            return block.verify(self.four(order=(0, 1, 2, 3)))

    def t1_lines(self):
        return [line for line in self.lines if 'verify t1' in line]

    def test_flag_off_the_block_is_todays(self):
        block = self.build_with({})
        warm, captured = MaskAwareModelBatch.instances
        self.assertEqual([fixture.options['attention_mask_once'] for fixture in (warm, captured)], [False, False])
        self.assertEqual(packed_verifier.sample_rows.call_count, 2, 'warm forward and capture, as before')
        self.assertEqual(self.shard_calls, [])
        self.assertEqual(len(block.output), 2)
        predictions, metrics = self.verify(block, {})
        self.assertEqual(predictions, [list(range(1000 + 16 * user, 1016 + 16 * user)) for user in range(4)])
        self.assertEqual(self.t1_lines(), [])

    def test_flag_on_masks_are_shared_and_the_ids_come_from_the_shard_combine(self):
        block = self.build_with(ON)
        warm, captured = MaskAwareModelBatch.instances
        self.assertEqual([fixture.options['attention_mask_once'] for fixture in (warm, captured)], [True, True])
        packed_verifier.sample_rows.assert_not_called()
        self.assertEqual(self.shard_calls, [((1, 1, 64, 124160), 64)] * 2)
        self.assertEqual(len(block.output), 3)
        predictions, metrics = self.verify(block, ON)
        self.assertEqual(predictions, [self.COMBINED[16 * user:16 * user + 16] for user in range(4)])
        self.assertEqual(self.t1_lines(), ['[PINDIAG] verify t1 engaged site=packed_verify audit=0 coalesce_fallback=0 '
                                           'coalesced=0 direct_carry=0 last_carry=0 mask_once=1 shard_argmax=1'])

    def test_the_marker_counts_only_what_the_captured_forward_engaged(self):
        def forward(fixture_self, *, sharded_logits):
            t1.note('direct_carry', 48)
            t1.note('last_carry', 48)
            t1.note('coalesced', 48)
            return FakeModelBatch.forward(fixture_self, sharded_logits=sharded_logits)

        with patch.object(MaskAwareModelBatch, 'forward', forward):
            self.build_with(ON)
        self.assertEqual(self.t1_lines(), ['[PINDIAG] verify t1 engaged site=packed_verify audit=0 coalesce_fallback=0 '
                                           'coalesced=48 direct_carry=48 last_carry=48 mask_once=1 shard_argmax=1'])

    def test_a_sampler_that_is_not_plain_greedy_keeps_the_pinned_path_and_says_why(self):
        block = self.build_with(ON, sampler=greedy_sampler(_penalties_active=True))
        self.assertEqual(self.shard_calls, [])
        self.assertEqual(packed_verifier.sample_rows.call_count, 2)
        self.assertEqual(len(block.output), 2)
        predictions, metrics = self.verify(block, ON)
        self.assertEqual(predictions[0], list(range(1000, 1016)))
        self.assertEqual(self.t1_lines(), [
            '[PINDIAG] verify t1 engaged site=packed_verify audit=0 coalesce_fallback=0 coalesced=0 direct_carry=0 '
            'last_carry=0 mask_once=1 shard_argmax=0',
            '[PINDIAG] verify t1 kept the pinned sampler: penalties or log-probabilities are active'])

    def test_skipped_cuts_keep_todays_mask_and_sampler_and_say_nothing_about_it(self):
        skip = dict(ON, QWEN_FAST_VERIFY_T1_SKIP='mask_once,shard_argmax')
        block = self.build_with(skip)
        warm, captured = MaskAwareModelBatch.instances
        self.assertEqual([fixture.options['attention_mask_once'] for fixture in (warm, captured)], [False, False])
        self.assertEqual(self.shard_calls, [])
        self.assertEqual(packed_verifier.sample_rows.call_count, 2)
        self.assertEqual(len(block.output), 2)
        predictions, metrics = self.verify(block, skip)
        self.assertEqual(predictions, [list(range(1000 + 16 * user, 1016 + 16 * user)) for user in range(4)])
        self.assertEqual(self.t1_lines(), ['[PINDIAG] verify t1 engaged site=packed_verify audit=0 coalesce_fallback=0 '
                                           'coalesced=0 direct_carry=0 last_carry=0 mask_once=0 shard_argmax=0'])

    def test_the_audit_runs_the_pinned_sampler_beside_it_and_compares_every_row(self):
        audit = dict(ON, QWEN_FAST_VERIFY_T1_AUDIT='1')
        self.ids.value = torch.tensor(self.COMBINED, dtype=torch.int64)
        block = self.build_with(audit)
        self.assertEqual(len(block.output), 4)
        self.assertEqual(packed_verifier.sample_rows.call_count, 2)
        logged = []
        t1._AUDIT.update(rounds=0, rows=0)
        with patch.object(t1, 'log_line', side_effect=logged.append):
            predictions, metrics = self.verify(block, audit)
        self.assertEqual(predictions[1], self.COMBINED[16:32])
        self.assertEqual(logged, [t1.AUDIT_MARKER + ' 1 exact=True rows=64'])

    def test_an_audit_mismatch_fails_the_round(self):
        audit = dict(ON, QWEN_FAST_VERIFY_T1_AUDIT='1')
        block = self.build_with(audit)
        entries = self.four(order=(0, 1, 2, 3))
        logged = []
        with patch.dict('os.environ', audit, clear=True), patch.object(t1, 'log_line', side_effect=logged.append):
            with self.assertRaises(AssertionError):
                block.verify(entries)
        self.assertEqual(block.phase, 'failed')
        self.assertTrue(logged[0].startswith(t1.AUDIT_MISMATCH))
        for entry in entries:
            entry['request'].session.fail_verification.assert_called_once()

    def test_nobody_trusts_native_slot_0_across_the_round(self):
        """#3's premise for dropping the last user's restore/save: the block clears
        verifier_engine's residency before the verify trace and after every commit, so any
        sequential step afterwards restores its own carry instead of reading slot 0."""
        import verifier_engine

        block = self.build_with(ON)
        sentinel, seen = object(), []
        original = self.ttnn.execute_trace

        def execute(mesh, trace, cq_id=0, blocking=True):
            seen.append(verifier_engine._resident)
            verifier_engine._resident = sentinel  # as if something claimed slot 0 mid-round
            return original(mesh, trace, cq_id=cq_id, blocking=blocking)

        verifier_engine._resident = sentinel
        with patch.object(self.ttnn, 'execute_trace', side_effect=execute):
            predictions, metrics = self.verify(block, ON)
            for segment, prefix in zip(metrics['segments'], (3, 0, 16, 1)):
                block.commit_user(segment, prefix)
                self.assertIsNone(verifier_engine._resident)
        self.assertIsNone(seen[0], 'cleared before the verify trace ran')
        self.assertEqual(len(seen), 4, 'the verify trace and three commit traces (prefix 0 runs none)')

    def test_close_releases_every_output(self):
        block = self.build_with(dict(ON, QWEN_FAST_VERIFY_T1_AUDIT='1'))
        outputs = list(block.output)
        with patch('packed_verifier.release_owned') as release:
            block.close()
        released = [value for call in release.call_args_list for value in call.args[1]]
        self.assertTrue(all(any(value is item for item in released) for value in outputs))


if __name__ == '__main__':
    unittest.main()
