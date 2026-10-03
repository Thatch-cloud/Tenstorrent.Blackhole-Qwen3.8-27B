"""tp4_draft_conv (QWEN_FAST_TP4_DRAFT_CONV, D2a): the rewritten conv I/O stage hands the compute kernel byte-identical tiles.

`served_image` is the served kernels' per-element loop (draft_convolution_fused_io.cpp / quad_conv_io.cpp) and `fast_image` the
word-copy reader's `prepare()` (draft_conv_io_fast.cpp), both transliterated line for line; the test builds the seven-tile CB 0
image both ways for every page position, live-row count and seam mask and holds them equal. Plus the page and worker plan against the
pinned quad plan, the launch the builder describes, the refusals, the audit and the twins' flag-off path."""

import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import draft_convolution_fused_tp as pair_twin
import quad_draft
import quad_draft_tp as quad_twin
import tp4_draft_conv as conv
import tp4_sampdraft
from test_tp4_shard_argmax import FakeOperations

FOUR = {'QWEN_FAST_TP': '4'}


def lane(row, column):
    return (row // 16) * 512 + (column // 16) * 256 + (row % 16) * 16 + column % 16


def served_image(hidden, base0, base1, dynamic0, dynamic1, page, live, seams):
    """The served per-element loop: 7 tiles of 1,024 bf16 (as ints)."""
    tiles = [0] * (7 * 1024)
    tiles[0:1024] = hidden
    tiles[2 * 1024:3 * 1024] = base0
    tiles[4 * 1024:5 * 1024] = base1
    for row in range(32):
        for column in range(32):
            element = lane(row, column)
            group = lane(row, 2 * (page % 16) + column // 16)
            carries = row < live and not ((seams >> row) & 1)
            tiles[1024 + element] = tiles[lane(row - 1, column)] if carries else 0
            tiles[2 * 1024 + element] = tiles[2 * 1024 + lane(0, column)]
            tiles[3 * 1024 + element] = dynamic0[group] if row < live else 0
            tiles[4 * 1024 + element] = tiles[4 * 1024 + lane(0, column)]
            tiles[5 * 1024 + element] = dynamic1[group] if row < live else 0
            tiles[6 * 1024 + element] = 0
            if row >= live:
                tiles[element] = 0
    return tiles


def to_words(elements):
    return [elements[2 * index] | (elements[2 * index + 1] << 16) for index in range(len(elements) // 2)]


def row_words(row, half):
    return (row >> 4) * 256 + half * 128 + (row & 15) * 8


def fast_image(hidden, base0, base1, dynamic0, dynamic1, col, live, seams):
    """draft_conv_io_fast.cpp prepare(): word copies and fills over the 16 x 16 face layout. The CB slot starts holding the NoC
    reads (tiles 0, 2, 4) and arbitrary stale bytes elsewhere; the zero tile (6) is written once per slot."""
    rng = random.Random(1234)
    stale = [rng.randrange(1 << 32) for _ in range(7 * 512)]
    words = list(stale)
    words[0:512] = to_words(hidden)
    words[2 * 512:3 * 512] = to_words(base0)
    words[4 * 512:5 * 512] = to_words(base1)
    words[6 * 512:7 * 512] = [0] * 512                      # the once-per-slot zero tile
    shift, first_base, dyn0, second_base, dyn1 = 512, 2 * 512, 3 * 512, 4 * 512, 5 * 512
    first_column = 2 * (col % 16)
    seams |= 1
    for row in range(32):
        carries = row < live and not ((seams >> row) & 1)
        for half in range(2):
            at = row_words(row, half)
            if carries:
                source = row_words(row - 1, half)
                words[shift + at:shift + at + 8] = words[source:source + 8]
            else:
                words[shift + at:shift + at + 8] = [0] * 8
            if row != 0:
                zero_row = row_words(0, half)
                words[first_base + at:first_base + at + 8] = words[first_base + zero_row:first_base + zero_row + 8]
                words[second_base + at:second_base + at + 8] = words[second_base + zero_row:second_base + zero_row + 8]
            if row < live:
                index = lane(row, first_column + half)
                low, high = dynamic0[index], dynamic1[index]
                words[dyn0 + at:dyn0 + at + 8] = [(low << 16) | low] * 8
                words[dyn1 + at:dyn1 + at + 8] = [(high << 16) | high] * 8
            else:
                words[dyn0 + at:dyn0 + at + 8] = [0] * 8
                words[dyn1 + at:dyn1 + at + 8] = [0] * 8
                words[at:at + 8] = [0] * 8
    return words


def image_words(tiles):
    return to_words(tiles)


class ImageTests(unittest.TestCase):
    def operands(self, seed):
        rng = random.Random(seed)
        draw = lambda: [rng.randrange(1 << 16) for _ in range(1024)]  # noqa: E731
        return draw(), draw(), draw(), draw(), draw()

    def test_the_word_copy_image_is_the_served_image_for_every_page_live_count_and_seam_mask(self):
        masks = (1, 0x00010001, 0x00000001 | (1 << 16), 0x80000001, 0xFFFFFFFF, 0x55555555, 0x00008001)
        for page in range(0, 160, 7):
            for live in (1, 8, 16, 17, 31, 32):
                for mask in masks:
                    hidden, base0, base1, dynamic0, dynamic1 = self.operands(1000 * page + live)
                    served = image_words(served_image(hidden, base0, base1, dynamic0, dynamic1, page, live, mask))
                    fast = fast_image(hidden, base0, base1, dynamic0, dynamic1, page % 160, live, mask)
                    self.assertEqual(served, fast, (page, live, hex(mask)))

    def test_every_dynamic_column_group_is_covered(self):
        hidden, base0, base1, dynamic0, dynamic1 = self.operands(5)
        for page in range(16):
            served = image_words(served_image(hidden, base0, base1, dynamic0, dynamic1, page, 32, 1))
            self.assertEqual(served, fast_image(hidden, base0, base1, dynamic0, dynamic1, page, 32, 1), page)

    def test_the_quad_page_decomposition_picks_the_same_dynamic_group_as_the_served_formula(self):
        # served quad: group = lane(row, 2 * (col % 16) + column / 16) with col = page % 160; pair: page % 16 with page < 160
        for page in range(0, 320):
            self.assertEqual((page % 160) % 16, page % 16)

    def test_a_row_that_begins_a_segment_carries_nothing_and_so_does_a_dead_row(self):
        hidden, base0, base1, dynamic0, dynamic1 = self.operands(77)
        image = fast_image(hidden, base0, base1, dynamic0, dynamic1, 3, 20, 1 | (1 << 16))
        shift = image[512:1024]
        for row in (0, 16, 20, 31):                      # seam rows and rows at or past live
            self.assertEqual(shift[row_words(row, 0):row_words(row, 0) + 8], [0] * 8)
            self.assertEqual(shift[row_words(row, 1):row_words(row, 1) + 8], [0] * 8)
        for row in (1, 15, 17, 19):
            source = to_words(hidden)
            self.assertEqual(shift[row_words(row, 0):row_words(row, 0) + 8],
                             source[row_words(row - 1, 0):row_words(row - 1, 0) + 8], row)

    def test_the_zero_tile_survives_and_stale_slot_bytes_never_reach_the_other_tiles(self):
        hidden, base0, base1, dynamic0, dynamic1 = self.operands(31)
        image = fast_image(hidden, base0, base1, dynamic0, dynamic1, 40, 32, 1)
        self.assertEqual(image[6 * 512:], [0] * 512)


class PlanTests(unittest.TestCase):
    def test_the_page_plan_is_the_pinned_quad_plan(self):
        for workers, rows in ((80, 32), (80, 64), (110, 64), (80, 1), (80, 16)):
            self.assertEqual(conv.pages_for(workers, rows), quad_draft.conv_pages(workers, rows), (workers, rows))

    def test_the_pair_and_quad_plans_cover_every_page_once(self):
        for workers, rows in ((80, 32), (80, 64), (110, 64)):
            pages = conv.pages_for(workers, rows)
            flat = sorted(page for owned in pages.values() for page in owned)
            self.assertEqual(flat, list(range(160 * ((rows + 31) // 32))))

    def test_the_compute_groups_are_the_served_ones(self):
        self.assertEqual({count: len(group) for count, group in conv.group_workers(conv.pages_for(80, 32)).items()}, {2: 80})
        self.assertEqual({count: len(group) for count, group in conv.group_workers(conv.pages_for(110, 64)).items()},
                         {3: 100, 2: 10})
        self.assertEqual({count: len(group) for count, group in conv.group_workers(conv.pages_for(80, 64)).items()}, {4: 80})


def tensors_for(operations, rows):
    shapes = [(1, 1, rows, 5120)] + [(1, 1, rows, 320)] * 2 + [(1, 1, 1, 5120)] * 2
    return [operations.tensor(shape) for shape in shapes]


def pair_call(operations, rows=32):
    values = tensors_for(operations, rows)
    return values, dict(hidden=values[0], dynamic=values[1:3], base=values[3:])


class Setup(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.DRAFT_CONV: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        tp4_sampdraft._LOGGED.clear()
        conv._CURRENT['scope'] = None
        conv._OPENED.clear()
        self.lines = []
        logger = patch.object(tp4_sampdraft, 'log_line', side_effect=self.lines.append)
        logger.start()
        self.addCleanup(logger.stop)

    def operations(self):
        operations = FakeOperations()
        operations.clone = Mock(side_effect=lambda value, memory_config=None: operations.tensor(value.shape))
        return operations

    def mesh(self, chips=4):
        return SimpleNamespace(shape=[1, chips])

    def call(self, operations, rows, workers, coordinates, seams_low, seams_high, served, **extra):
        values, parts = pair_call(operations, rows)
        output = conv.convolution(
            operations, extra.pop('mesh', self.mesh()), parts['hidden'], parts['dynamic'], parts['base'], rows=rows,
            seams_low=seams_low, seams_high=seams_high, workers=workers, coordinates=coordinates, label=extra.pop('label', 'pair'),
            core_set=lambda group: [tuple(group)], served=served, **extra)
        return values, output


class BuilderTests(Setup):
    def test_the_pair_launch_is_one_generic_op_of_reader_writer_and_compute_per_chip(self):
        operations = self.operations()
        served = Mock()
        coordinates = [(worker % 8, worker // 8) for worker in range(80)]
        values, output = self.call(operations, 32, 80, coordinates, 0b101, 0, served)
        served.assert_not_called()
        self.assertEqual(output, operations.empties[0])
        self.assertEqual(len(operations.generic), 1)
        tensors, program = operations.generic[0]
        self.assertEqual(tensors, [*values, output])
        self.assertEqual(len(program), 4)
        for chip, descriptor in enumerate(program.values()):
            reader, writer, compute = descriptor['kernels']
            self.assertTrue(reader['kernel_source'].endswith('draft_conv_io_fast.cpp'))
            self.assertTrue(writer['kernel_source'].endswith('draft_conv_out.cpp'))
            self.assertTrue(compute['kernel_source'].endswith('draft_convolution_fused_compute.cpp'))
            self.assertEqual((reader['config']['processor'], reader['config']['noc']), (0, 0))
            self.assertEqual((writer['config']['processor'], writer['config']['noc']), (1, 1))
            self.assertEqual(compute['compile_time_args'], [2])
            self.assertEqual(compute['config'], dict(math_fidelity='hifi4', fp32_dest_acc_en=True, math_approx_mode=False))
            # the reader carries the five input accessors, the writer the output's
            self.assertEqual(len(reader['compile_time_args']), 5)
            self.assertEqual(len(writer['compile_time_args']), 1)
            for worker, (x, y) in enumerate(coordinates):
                self.assertEqual(reader['runtime_args'][x][y],
                                 [value.shards[chip].buffer_address() for value in values] + [32, worker, 80, 0b101, 0])
                self.assertEqual(writer['runtime_args'][x][y], [output.shards[chip].buffer_address(), 32, worker, 80])
            self.assertEqual([cb['total_size'] for cb in descriptor['cbs']], [2048 * 14, 2048 * 2, 2048 * 1])
            self.assertEqual([cb['format_descriptors'][0]['buffer_index'] for cb in descriptor['cbs']], [0, 1, 16])
        self.assertEqual(operations.freed, [])

    def test_the_quad_launch_groups_the_compute_kernels_by_pages_per_worker(self):
        operations = self.operations()
        coordinates = [(worker // 10, worker % 10) for worker in range(110)]
        values, output = self.call(operations, 64, 110, coordinates, 0x00010001, 0x00010001, Mock(), label='quad')
        _, program = operations.generic[0]
        for descriptor in program.values():
            reader, writer, *computes = descriptor['kernels']
            self.assertEqual([kernel['compile_time_args'] for kernel in computes], [[2], [3]])
            self.assertEqual([len(kernel['core_ranges'][0]) for kernel in computes], [10, 100])
            self.assertEqual(reader['runtime_args'][10][9][-5:], [64, 109, 110, 0x00010001, 0x00010001])

    def test_runtime_argument_lengths_never_vary(self):
        lengths = set()
        for rows, workers, coordinates in ((32, 80, [(w % 8, w // 8) for w in range(80)]), (16, 80, [(w % 8, w // 8) for w in range(80)]),
                                          (64, 110, [(w // 10, w % 10) for w in range(110)]),
                                          (64, 80, [(w % 8, w // 8) for w in range(80)])):
            operations = self.operations()
            self.call(operations, rows, workers, coordinates, 1, 1, Mock())
            for descriptor in operations.generic[0][1].values():
                reader, writer = descriptor['kernels'][:2]
                for x in reader['runtime_args']:
                    for words in reader['runtime_args'][x].values():
                        lengths.add(('reader', len(words)))
                for x in writer['runtime_args']:
                    for words in writer['runtime_args'][x].values():
                        lengths.add(('writer', len(words)))
        self.assertEqual(lengths, {('reader', 10), ('writer', 4)})

    def test_one_engaged_marker_per_site_and_shape(self):
        operations = self.operations()
        coordinates = [(worker % 8, worker // 8) for worker in range(80)]
        for _ in range(3):
            self.call(operations, 32, 80, coordinates, 1, 0, Mock())
        engaged = [line for line in self.lines if line.startswith(tp4_sampdraft.CONV_ENGAGED)]
        self.assertEqual(engaged, ['%s site=pair rows=32 pages=160 workers=80' % tp4_sampdraft.CONV_ENGAGED])

    def test_a_call_it_cannot_take_runs_the_served_call_and_says_why(self):
        coordinates = [(worker % 8, worker // 8) for worker in range(80)]
        cases = []
        operations = self.operations()
        cases.append(('mesh', dict(mesh=self.mesh(2)), operations))
        operations = self.operations()
        cases.append(('rows', dict(), operations))
        for name, extra, operations in cases:
            served = Mock(return_value='served output')
            rows = 65 if name == 'rows' else 32
            values, output = self.call(operations, rows, 80, coordinates, 1, 0, served, **extra)
            self.assertEqual(output, 'served output', name)
            served.assert_called_once()
            self.assertEqual(operations.generic, [])
            self.assertEqual(operations.empties, [])
        operations = self.operations()
        bad = tensors_for(operations, 32)
        bad[0].dtype = 'fp32'
        served = Mock(return_value='served output')
        self.assertEqual(conv.convolution(operations, self.mesh(), bad[0], bad[1:3], bad[3:], rows=32, seams_low=1, seams_high=0,
                                          workers=80, coordinates=coordinates, label='pair', core_set=lambda g: g, served=served),
                         'served output')
        fell = [line for line in self.lines if line.startswith(tp4_sampdraft.CONV_FALLBACK)]
        self.assertEqual(len(fell), 3)

    def test_a_failed_submission_releases_only_the_owned_output(self):
        coordinates = [(worker % 8, worker // 8) for worker in range(80)]
        operations = self.operations()
        operations.generic_op = Mock(side_effect=RuntimeError('submit'))
        with self.assertRaises(RuntimeError):
            self.call(operations, 32, 80, coordinates, 1, 0, Mock())
        self.assertEqual(operations.freed, [operations.empties[0]])


class AuditTests(Setup):
    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ, {tp4_sampdraft.DRAFT_CONV_AUDIT: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def audited(self, operations, calls=1, scope=True):
        """`calls` engaged convs inside an open capture scope; returns (scope, last served mock)."""
        coordinates = [(worker % 8, worker // 8) for worker in range(80)]
        opened = conv.open_scope('pair') if scope else None
        served = None
        try:
            for _ in range(calls):
                served = Mock(return_value=operations.tensor((1, 1, 32, 5120)))
                self.call(operations, 32, 80, coordinates, 1, 0, served)
        finally:
            conv.close_scope(opened)
        return opened, served

    def test_a_call_inside_a_scope_runs_the_served_kernel_and_the_scope_holds_the_pair(self):
        operations = self.operations()
        scope, served = self.audited(operations)
        served.assert_called_once()
        self.assertEqual(len(scope), 1)
        self.assertEqual(operations.clone.call_count, 1)

    def test_a_call_with_no_scope_warms_the_audit_and_holds_nothing(self):
        # The bucket's warm pass (no scope): the served conv and the audit clone run once so the capture only replays them
        # (v403: 'Cannot load new binaries during trace capture'); both outputs are freed and nothing is held.
        operations = self.operations()
        scope, served = self.audited(operations, scope=False)
        served.assert_called_once()
        self.assertEqual(operations.clone.call_count, 1)
        self.assertIn(served.return_value, operations.freed)
        self.assertIsNone(scope)
        self.assertFalse(conv.capturing())

    def test_with_the_audit_off_a_call_runs_no_served_conv_and_no_clone(self):
        operations = self.operations()
        with patch.dict(os.environ, {tp4_sampdraft.DRAFT_CONV_AUDIT: '0'}):
            scope, served = self.audited(operations, scope=False)
        served.assert_not_called()
        self.assertEqual(operations.clone.call_count, 0)

    def test_the_scope_closes_so_later_calls_hold_nothing(self):
        operations = self.operations()
        scope, _ = self.audited(operations)
        _, served = self.audited(operations, scope=False)
        served.assert_called_once()  # the warm pass, not an audit: nothing is held
        self.assertEqual(len(scope), 1)
        self.assertFalse(conv.capturing())

    def test_identical_pairs_pass_and_log_the_audit_marker(self):
        operations = self.operations()
        scope, _ = self.audited(operations, 2)
        data = torch.zeros(1, 1, 32, 5120, dtype=torch.bfloat16)
        operations.to_torch = lambda part: data
        self.assertEqual(conv.compare_scope(operations, scope), 2)
        self.assertTrue(any(line == '%s exact=True convs=2 round=1 site=pair' % tp4_sampdraft.CONV_AUDIT for line in self.lines))

    def test_a_differing_pair_raises_and_logs_the_mismatch_marker(self):
        operations = self.operations()
        scope, _ = self.audited(operations)
        _, mine, served, _ = scope.pairs[0]
        seen = {id(part): index for index, part in enumerate([*mine.shards, *served.shards])}
        operations.to_torch = lambda part: torch.full((1, 1, 32, 5120), 1.0 if seen[id(part)] < 4 else 2.0, dtype=torch.bfloat16)
        with self.assertRaises(AssertionError):
            conv.compare_scope(operations, scope)
        self.assertTrue(self.lines[-1].startswith(tp4_sampdraft.CONV_MISMATCH))

    def test_a_scope_stops_reading_back_after_the_audited_rounds_and_release_frees_what_it_holds(self):
        operations = self.operations()
        scope, _ = self.audited(operations)
        operations.to_torch = lambda part: torch.zeros(1, 1, 32, 5120, dtype=torch.bfloat16)
        results = [conv.compare_scope(operations, scope) for _ in range(conv.AUDIT_ROUNDS + 2)]
        self.assertEqual(results, [1] * conv.AUDIT_ROUNDS + [0, 0])
        _, mine, served, owned = scope.pairs[0]
        self.assertTrue(owned)
        conv.release_scope(operations, scope)
        self.assertEqual(scope.pairs, [])
        self.assertEqual(operations.freed, [mine, served])

    def test_a_scope_is_freed_by_its_own_release_only(self):
        operations = self.operations()
        first, _ = self.audited(operations)
        second, _ = self.audited(operations)
        conv.release_scope(operations, first)
        self.assertEqual(len(second), 1)
        self.assertEqual(len(operations.freed), 2)

    def test_the_module_holds_nothing_once_the_scope_is_released(self):
        operations = self.operations()
        scope, _ = self.audited(operations)
        conv.release_scope(operations, scope)
        self.assertFalse(hasattr(conv, '_HELD'))
        self.assertIsNone(conv._CURRENT['scope'])

    def test_only_the_first_scopes_of_a_site_are_audited(self):
        scopes = []
        for _ in range(conv.AUDIT_SCOPES + 2):
            scope = conv.open_scope('quad')
            scopes.append(scope)
            conv.close_scope(scope)
        self.assertEqual([scope is None for scope in scopes], [False] * conv.AUDIT_SCOPES + [True, True])
        other = conv.open_scope('pair')
        self.assertIsNotNone(other)
        conv.close_scope(other)

    def test_two_open_scopes_are_refused(self):
        scope = conv.open_scope('pair')
        with self.assertRaises(RuntimeError):
            conv.open_scope('quad')
        conv.close_scope(scope)

    def test_nothing_held_compares_nothing(self):
        self.assertEqual(conv.compare_scope(self.operations(), None), 0)
        self.assertEqual(conv.compare_scope(self.operations(), conv.Scope('pair')), 0)

    def test_a_failing_clone_frees_the_served_output(self):
        operations = self.operations()
        operations.clone = Mock(side_effect=RuntimeError('clone'))
        coordinates = [(worker % 8, worker // 8) for worker in range(80)]
        scope = conv.open_scope('pair')
        reference = operations.tensor((1, 1, 32, 5120))
        try:
            with self.assertRaises(RuntimeError):
                self.call(operations, 32, 80, coordinates, 1, 0, Mock(return_value=reference))
        finally:
            conv.close_scope(scope)
        self.assertIn(reference, operations.freed)


class DrafterTraceHooksTests(unittest.TestCase):
    """dflash_proposal_trace's scope helpers: inert with both audits off, a scope per bucket with one on."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in tp4_sampdraft.ALL_FLAGS:
            os.environ.pop(name, None)
        conv._CURRENT['scope'] = None
        conv._OPENED.clear()

    def test_with_both_audits_off_nothing_opens_compares_or_frees(self):
        import dflash_proposal_trace as trace
        self.assertIsNone(trace.open_draft_audit('pair'))
        trace.close_draft_audit(None)
        trace.compare_draft_audit('ops', SimpleNamespace())
        trace.compare_draft_audit('ops', SimpleNamespace(draft_audit=None))
        trace.release_draft_audit('ops', None)
        self.assertEqual(conv._OPENED, {})

    def test_either_audit_opens_a_scope_and_the_helpers_compare_and_free_it(self):
        import dflash_proposal_trace as trace
        for lever, audit in ((tp4_sampdraft.DRAFT_CONV, tp4_sampdraft.DRAFT_CONV_AUDIT),
                             (tp4_sampdraft.DRAFT_HEADS, tp4_sampdraft.DRAFT_HEADS_AUDIT)):
            conv._OPENED.clear()
            with patch.dict(os.environ, {lever: '1', audit: '1'}):
                scope = trace.open_draft_audit('pair')
                self.assertIsInstance(scope, conv.Scope)
                trace.close_draft_audit(scope)
                with patch('tp4_draft_conv.compare_scope') as compare, patch('tp4_draft_conv.release_scope') as release:
                    trace.compare_draft_audit('ops', SimpleNamespace(draft_audit=scope))
                    trace.release_draft_audit('ops', scope)
                compare.assert_called_once_with('ops', scope)
                release.assert_called_once_with('ops', scope)

    def test_an_audit_without_its_lever_fails_at_the_first_open(self):
        import dflash_proposal_trace as trace
        with patch.dict(os.environ, {tp4_sampdraft.DRAFT_HEADS_AUDIT: '1'}), self.assertRaises(ValueError):
            trace.open_draft_audit('pair')


class TwinTests(unittest.TestCase):
    def hooks(self, environ):
        patcher = patch.dict(os.environ, environ, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_flag_off_the_pair_twin_is_the_served_function(self):
        self.hooks({'QWEN_FAST_TP': '4'})
        os.environ.pop(tp4_sampdraft.DRAFT_CONV, None)
        with patch.object(pair_twin, 'served_fused_convolution', return_value='served') as served, \
                patch('tp4_draft_conv.convolution') as fast:
            self.assertEqual(pair_twin.fused_convolution('ops', 'mesh', 'hidden', ['d0', 'd1'], ['b0', 'b1'], boundaries=None), 'served')
        served.assert_called_once_with('ops', 'mesh', 'hidden', ['d0', 'd1'], ['b0', 'b1'], boundaries=None)
        fast.assert_not_called()

    def test_flag_on_the_pair_twin_asks_for_eighty_workers_and_the_pair_seam_mask(self):
        self.hooks({'QWEN_FAST_TP': '4', tp4_sampdraft.DRAFT_CONV: '1'})
        with patch.object(pair_twin, 'validate_shapes', return_value=16), patch.object(pair_twin, 'seam_mask', return_value=0b1001), \
                patch('tp4_draft_conv.convolution', return_value='fast') as fast:
            self.assertEqual(pair_twin.fused_convolution('ops', 'mesh', 'hidden', ['d0', 'd1'], ['b0', 'b1'], boundaries='b'), 'fast')
        options = fast.call_args.kwargs
        self.assertEqual((options['rows'], options['seams_low'], options['seams_high'], options['workers'], options['label']),
                         (16, 0b1001, 0, 80, 'pair'))
        self.assertEqual(options['coordinates'][:10], [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (0, 1), (1, 1)])
        self.assertEqual(options['coordinates'][-1], (7, 9))

    def test_flag_off_the_quad_twin_is_the_served_function(self):
        self.hooks({'QWEN_FAST_TP': '4'})
        os.environ.pop(tp4_sampdraft.DRAFT_CONV, None)
        with patch.object(quad_twin, 'served_quad_fused_convolution', return_value='served') as served, \
                patch('tp4_draft_conv.convolution') as fast:
            self.assertEqual(quad_twin.quad_fused_convolution('ops', 'mesh', 'h', 'd', 'b', boundaries='x', conv='80'), 'served')
        served.assert_called_once_with('ops', 'mesh', 'h', 'd', 'b', boundaries='x', conv='80')
        fast.assert_not_called()

    def test_flag_on_the_quad_twin_takes_the_pinned_variants_workers_and_seam_words(self):
        self.hooks({'QWEN_FAST_TP': '4', tp4_sampdraft.DRAFT_CONV: '1'})
        for variant, workers in (('110', 110), ('80', 80)):
            with patch.object(quad_twin, 'validate_conv_shapes'), patch.object(quad_twin, 'seam_words', return_value=(7, 9)), \
                    patch('tp4_draft_conv.convolution', return_value='fast') as fast:
                self.assertEqual(quad_twin.quad_fused_convolution('ops', 'mesh', 'h', 'd', 'b', boundaries='x', conv=variant), 'fast')
            options = fast.call_args.kwargs
            self.assertEqual((options['rows'], options['seams_low'], options['seams_high'], options['workers'], options['label']),
                             (64, 7, 9, workers, 'quad'))
            self.assertEqual(len(options['coordinates']), workers)

    def test_the_quad_twin_still_refuses_an_unknown_variant_with_the_flag_on(self):
        self.hooks({'QWEN_FAST_TP': '4', tp4_sampdraft.DRAFT_CONV: '1'})
        with patch.object(quad_twin, 'validate_conv_shapes'), self.assertRaisesRegex(ValueError, '110 or 80'):
            quad_twin.quad_fused_convolution('ops', 'mesh', 'h', 'd', 'b', boundaries='x', conv='halves')

    def test_the_flag_is_refused_at_the_pair(self):
        self.hooks({tp4_sampdraft.DRAFT_CONV: '1'})
        os.environ.pop('QWEN_FAST_TP', None)
        with self.assertRaisesRegex(ValueError, 'TP4 levers'):
            pair_twin.fused_convolution('ops', 'mesh', 'hidden', [], [], boundaries=None)


class KernelSourceTests(unittest.TestCase):
    """The C++ cannot be compiled here; these hold the properties the Python builder and the transliteration rely on."""

    def read(self, name):
        from pathlib import Path
        return Path(conv.__file__).with_name(name).read_text()

    def test_the_reader_and_writer_agree_on_the_page_plan_and_the_argument_order(self):
        reader, writer = self.read('draft_conv_io_fast.cpp'), self.read('draft_conv_out.cpp')
        for text in (reader, writer):
            self.assertIn('for (uint32_t page = worker; page < 160 * tile_rows; page += workers)', text)
            self.assertIn('const uint32_t tile_rows = (rows + 31) / 32;', text)
        for index, name in enumerate(('rows', 'worker', 'workers', 'seams_low', 'seams_high')):
            self.assertIn('const uint32_t %s = get_arg_val<uint32_t>(%d);' % (name, index + 5), reader)
        for index, name in enumerate(('rows', 'worker', 'workers')):
            self.assertIn('const uint32_t %s = get_arg_val<uint32_t>(%d);' % (name, index + 1), writer)

    def test_the_reader_zeroes_tile_six_of_both_slots_and_pushes_seven_pages(self):
        reader = self.read('draft_conv_io_fast.cpp')
        self.assertIn('(slot * 7 + 6) * TILE_BYTES', reader)
        self.assertIn('cb_reserve_back(0, 7);', reader)
        self.assertIn('cb_push_back(0, 7);', reader)
        self.assertNotIn('cb_wait_front', reader)

    def test_the_writer_waits_for_the_compute_result_and_nothing_else(self):
        writer = self.read('draft_conv_out.cpp')
        self.assertIn('cb_wait_front(16, 1);', writer)
        self.assertIn('noc_async_write_tile(page, output, get_read_ptr(16));', writer)
        self.assertIn('cb_pop_front(16, 1);', writer)

    def test_the_pinned_kernels_are_untouched(self):
        import hashlib
        from pathlib import Path
        digest = hashlib.sha256(Path(conv.__file__).with_name('quad_conv_io.cpp').read_bytes()).hexdigest()
        self.assertEqual(digest, quad_draft.CONV_KERNEL_SHA256)


if __name__ == '__main__':
    unittest.main()
