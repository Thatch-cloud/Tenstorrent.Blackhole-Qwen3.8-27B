"""tp4_shard_argmax (QWEN_FAST_TP4_SHARD_ARGMAX, S1): the scan and fold kernels transliterated and held against torch.argmax, the
launch plan, the flag-off path being today's calls, the refusals and the audit.

The two C++ kernels (tp4_shard_argmax_scan.cpp, tp4_shard_argmax_fold.cpp) cannot run on this machine. `simulate_scan` and
`simulate_fold` below are their line-for-line Python transliterations over the same tile-face word layout, so the tie-breaking,
the fast path's preconditions, the slow path and the fold order are proved on the CPU; the card harness and the audited gate prove
the transliteration is the kernel."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import tp4_sampdraft
import tp4_shard_argmax as sarg
import verify_trace_t1 as t1

TILE_WORDS = 512
FOUR = {'QWEN_FAST_TP': '4'}


def bits_of(values):
    """bf16 bit patterns (ints 0..65535) of a float tensor."""
    return (values.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF).tolist()


def lane(row, column):
    return (row // 16) * 512 + (column // 16) * 256 + (row % 16) * 16 + column % 16


def tile_words(rows_bits, tile_row, tile_column):
    """The 512 uint32 words of tile (tile_row, tile_column) of a (rows, 32 * tiles) bf16 matrix given as rows of bit patterns:
    elements in the four-face order, two bf16 a word, the lower address (lower column) in the low half. Rows past the matrix are 0."""
    elements = [0] * 1024
    for row in range(32):
        source = 32 * tile_row + row
        if source >= len(rows_bits):
            continue
        for column in range(32):
            elements[lane(row, column)] = rows_bits[source][32 * tile_column + column]
    return [elements[2 * index] | (elements[2 * index + 1] << 16) for index in range(TILE_WORDS)]


def order_key(bits):
    magnitude = bits & 0x7FFF
    if magnitude > 0x7F80:
        return 0x10000
    if magnitude == 0:
        return 0x8000
    return 0x8000 - magnitude if bits & 0x8000 else 0x8000 + magnitude


def int16(value):
    return value - 65536 if value & 0x8000 else value


def simulate_scan(tiles, live, first):
    """tp4_shard_argmax_scan.cpp for one task: `tiles` is the list of tile word lists of the run, `first` its first tile column.
    Returns the 32 words of the partials page."""
    result = [0] * 32
    for row in range(live):
        row_offset = (row >> 4) * 256 + (row & 15) * 8
        best, best_at, nan_flags = -65536, 0, 0
        for tile, words in enumerate(tiles):
            for half in range(2):
                base = half * 128 + row_offset
                at = tile * 32 + half * 16
                for k in range(8):
                    word = words[base + k]
                    nan_flags |= ((word & 0x7FFF7FFF) + 0x007F007F) & 0xFFFFFFFF
                    low, high = int16(word & 0xFFFF), int16(word >> 16)
                    if low > best:
                        best, best_at = low, at + 2 * k
                    if high > best:
                        best, best_at = high, at + 2 * k + 1
        bits = best & 0xFFFF
        if nan_flags & 0x80008000 or best <= 0:
            best_key = 0
            for tile, words in enumerate(tiles):
                for half in range(2):
                    base = half * 128 + row_offset
                    at = tile * 32 + half * 16
                    for k in range(8):
                        word = words[base + k]
                        low_key = order_key(word & 0xFFFF)
                        if low_key > best_key:
                            best_key, best_at, bits = low_key, at + 2 * k, word & 0xFFFF
                        high_key = order_key(word >> 16)
                        if high_key > best_key:
                            best_key, best_at, bits = high_key, at + 2 * k + 1, word >> 16
        result[row] = ((first * 32 + best_at) << 16) | bits
    return result


def simulate_fold(pages, rows, per_tile_row):
    """tp4_shard_argmax_fold.cpp: `pages` the partials pages in task order. Returns (ids, values) lists of 64."""
    ids, values = [0] * 64, [0] * 64
    for row in range(rows):
        tile_row, word = row >> 5, row & 31
        best_key = 0
        for task in range(per_tile_row):
            record = pages[tile_row * per_tile_row + task][word]
            key = order_key(record & 0xFFFF)
            if key > best_key:
                best_key, ids[row], values[row] = key, record >> 16, record & 0xFFFF
    return ids, values


def run_kernels(rows_bits, tile_columns):
    """The scan tasks of plan() over `rows_bits` (a rows x 32 * tile_columns matrix of bit patterns), then the fold."""
    rows = len(rows_bits)
    tasks, per_tile_row, tile_rows = sarg.plan(rows, tile_columns)
    pages = [None] * len(tasks)
    cache = {}
    for task, worker, role, tile_row, first, last, live in tasks:
        tiles = []
        for column in range(first, last):
            if (tile_row, column) not in cache:
                cache[(tile_row, column)] = tile_words(rows_bits, tile_row, column)
            tiles.append(cache[(tile_row, column)])
        pages[task] = simulate_scan(tiles, live, first)
    return simulate_fold(pages, rows, per_tile_row)


def reference(rows_bits):
    """torch.argmax per row of the bf16 matrix, and the bit pattern of the element it picks."""
    matrix = torch.tensor(rows_bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    ids = torch.argmax(matrix, dim=1).tolist()
    return ids, [rows_bits[row][column] for row, column in enumerate(ids)]


class Matrix:
    """A small bf16 matrix builder: random rows, then edits by (row, column) -> value bits."""

    def __init__(self, rows, tile_columns, seed=0, scale=1.0):
        generator = torch.Generator().manual_seed(seed)
        self.rows = bits_of(torch.randn(rows, 32 * tile_columns, generator=generator) * scale)
        self.tile_columns = tile_columns

    def put(self, row, column, bits):
        self.rows[row][column] = bits
        return self


POSITIVE = lambda value: bits_of(torch.tensor([value]))[0]  # noqa: E731
NEGATIVE_ZERO, POSITIVE_ZERO, INFINITY, NAN = 0x8000, 0x0000, 0x7F80, 0x7FC0


class PlanTests(unittest.TestCase):
    def test_the_column_plan_covers_every_tile_column_once_in_ascending_order(self):
        for columns in (1940, 220, 110, 1980):
            runs = sarg.column_runs(columns)
            flat = [column for first, last in runs for column in range(first, last)]
            self.assertEqual(flat, list(range(columns)), columns)
            self.assertEqual(len(runs), 110)
        runs = sarg.column_runs(1940)
        self.assertEqual([last - first for first, last in runs].count(18), 70)
        self.assertEqual([last - first for first, last in runs].count(17), 40)

    def test_the_tasks_cover_each_tile_row_in_ascending_column_order_for_every_row_count(self):
        for rows in (1, 8, 16, 31, 32, 33, 48, 63, 64):
            tasks, per_tile_row, tile_rows = sarg.plan(rows, 1940)
            self.assertEqual(tile_rows, (rows + 31) // 32)
            self.assertEqual(len(tasks), sarg.TASKS)
            self.assertEqual(per_tile_row * tile_rows, sarg.TASKS)
            self.assertEqual(sorted(task[0] for task in tasks), list(range(sarg.TASKS)))
            for tile_row in range(tile_rows):
                mine = sorted((task for task in tasks if task[3] == tile_row), key=lambda task: task[0])
                columns = [column for task in mine for column in range(task[4], task[5])]
                self.assertEqual(columns, list(range(1940)), (rows, tile_row))
                self.assertEqual(len(mine), per_tile_row)
                self.assertTrue(all(task[6] == min(32, rows - 32 * tile_row) for task in mine))
            self.assertTrue(all(task[5] - task[4] <= sarg.MAX_TILES for task in tasks))
            # each worker carries exactly one task per RISC
            for worker in range(110):
                self.assertEqual(sorted(task[2] for task in tasks if task[1] == worker), [0, 1])

    def test_the_plan_refuses_what_the_kernels_cannot_hold(self):
        for rows in (0, 65, -1, 1.0, None):
            with self.assertRaises(ValueError):
                sarg.plan(rows, 1940)
        with self.assertRaises(ValueError):
            sarg.plan(64, 100)          # fewer tile columns than workers
        with self.assertRaises(ValueError):
            sarg.plan(64, 110 * 18 + 1)  # a run past the scan scratch
        with self.assertRaises(ValueError):
            sarg.plan(16, 110)           # up to 32 rows the RISC-Vs split a one-tile run: an empty half
        sarg.plan(64, 110)               # one tile row per RISC needs no split


class KernelTests(unittest.TestCase):
    """The transliterated kernels against torch.argmax: ids AND the winning element's bits."""

    def check(self, matrix):
        ids, values = run_kernels(matrix.rows, matrix.tile_columns)
        want_ids, want_bits = reference(matrix.rows)
        rows = len(matrix.rows)
        self.assertEqual(ids[:rows], want_ids)
        self.assertEqual(values[:rows], want_bits)
        self.assertEqual(ids[rows:], [0] * (64 - rows))
        self.assertEqual(values[rows:], [0] * (64 - rows))

    def test_random_rows_at_every_row_count_that_matters(self):
        for rows in (1, 16, 32, 33, 64):
            with self.subTest(rows=rows):
                self.check(Matrix(rows, 220, seed=rows))

    def test_the_real_shard_width_at_sixty_four_rows(self):
        self.check(Matrix(64, 1940, seed=7))

    def test_ties_break_to_the_lowest_column_everywhere_a_boundary_can_hide_one(self):
        top = POSITIVE(30.0)
        columns = 220
        # (first, second): inside a face row, across the face seam 15/16, across tiles 31/32, across a worker run (the first
        # run of 220 columns over 110 workers is 2 tile columns: columns 63/64), and far apart.
        pairs = [(3, 9), (15, 16), (31, 32), (63, 64), (100, 6000), (0, 7039), (6000, 6001)]
        for rows in (16, 64):
            for row in range(rows):
                matrix = Matrix(rows, columns, seed=100 + row)
                first, second = pairs[row % len(pairs)]
                matrix.put(row, first, top).put(row, second, top)
                for other in range(rows):
                    if other != row:
                        matrix.put(other, (other * 37) % (32 * columns), POSITIVE(20.0))
                ids, values = run_kernels(matrix.rows, columns)
                self.assertEqual(ids[row], first, (rows, row, first, second))
                self.assertEqual(values[row], top)

    def test_a_tie_across_the_tile_row_seam_belongs_to_each_row_alone(self):
        matrix = Matrix(64, 220, seed=5)
        top = POSITIVE(40.0)
        for row in (0, 31, 32, 63):
            matrix.put(row, 11, top).put(row, 7000, top)
        ids, _ = run_kernels(matrix.rows, 220)
        self.assertEqual([ids[row] for row in (0, 31, 32, 63)], [11] * 4)

    def test_all_negative_rows_take_the_slow_path_and_agree_with_torch(self):
        matrix = Matrix(16, 220, seed=3)
        for row in range(16):
            matrix.rows[row] = [bits | 0x8000 if (bits & 0x7FFF) else NEGATIVE_ZERO for bits in matrix.rows[row]]
            matrix.rows[row] = [bits if bits != NEGATIVE_ZERO else 0x8001 for bits in matrix.rows[row]]
        # the maximum of an all-negative row is its smallest magnitude; make it unique in some rows and tied in others
        matrix.put(2, 5000, 0x8001).put(2, 5001, 0x8001)
        matrix.put(5, 17, 0xBF80).put(5, 18, 0xBF80)
        self.check(matrix)

    def test_zero_rows_treat_minus_zero_and_plus_zero_as_equal_and_keep_the_first(self):
        for first, second, expected in ((NEGATIVE_ZERO, POSITIVE_ZERO, NEGATIVE_ZERO), (POSITIVE_ZERO, NEGATIVE_ZERO, POSITIVE_ZERO)):
            matrix = Matrix(2, 220, seed=9)
            for row in range(2):
                matrix.rows[row] = [0x8001] * len(matrix.rows[row])          # every value negative
                matrix.put(row, 9, first).put(row, 10, second)               # then the two zeros
            ids, values = run_kernels(matrix.rows, 220)
            self.assertEqual(ids[:2], [9, 9])
            self.assertEqual(values[:2], [expected, expected])
            self.check(matrix)

    def test_a_row_whose_maximum_is_positive_zero_with_negatives_around_it(self):
        matrix = Matrix(4, 220, seed=11)
        for row in range(4):
            matrix.rows[row] = [0x8001 + (column % 50) for column in range(len(matrix.rows[row]))]
            matrix.put(row, 1000 + row, POSITIVE_ZERO)
        self.check(matrix)

    def test_infinity_is_the_maximum_and_negative_infinity_the_minimum(self):
        matrix = Matrix(8, 220, seed=13)
        for row in range(8):
            matrix.put(row, 1234, INFINITY).put(row, 3000, 0xFF80)
        matrix.put(3, 77, INFINITY)
        ids, _ = run_kernels(matrix.rows, 220)
        self.assertEqual(ids[3], 77)
        self.check(matrix)
        ids, values = run_kernels([[0xFF80] * (32 * 220)] * 2, 220)
        self.assertEqual(ids[:2], [0, 0])

    def test_a_nan_wins_whether_it_comes_before_or_after_the_maximum_and_the_first_nan_wins(self):
        for nan_column, top_column in ((5, 6000), (6000, 5), (31, 32), (3, 4)):
            matrix = Matrix(16, 220, seed=17)
            top = POSITIVE(55.0)
            for row in range(16):
                matrix.put(row, top_column, top).put(row, nan_column, NAN)
            matrix.put(1, nan_column + 1, 0xFFC1)           # a second, negative NaN later: the first still wins
            matrix.put(2, 4000, NAN)                        # a NaN far after: the earlier one still wins
            ids, values = run_kernels(matrix.rows, 220)
            self.assertEqual(ids[:16], [min(nan_column, 4000) if row == 2 else nan_column for row in range(16)][:16])
            self.assertTrue(all((value & 0x7F80) == 0x7F80 and value & 0x7F for value in values[:16]))
            self.check(matrix)

    def test_a_nan_only_in_one_task_still_moves_only_its_own_row(self):
        matrix = Matrix(64, 220, seed=19)
        matrix.put(40, 6500, NAN)
        ids, _ = run_kernels(matrix.rows, 220)
        want, _ = reference(matrix.rows)
        self.assertEqual(ids[40], 6500)
        self.assertEqual([ids[row] for row in range(64) if row != 40], [want[row] for row in range(64) if row != 40])

    def test_denormals_and_tiny_values_order_by_magnitude(self):
        matrix = Matrix(4, 220, seed=23)
        for row in range(4):
            matrix.rows[row] = [0x8000 | (1 + column % 3) for column in range(len(matrix.rows[row]))]
            matrix.put(row, 400 + row, 0x0003).put(row, 401 + row, 0x0002)
        self.check(matrix)


class FakeOperations:
    """A recording fake for the launch builder: tensors know their shape and shards, generic_op is recorded."""

    bfloat16, uint32 = 'bf16', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'tile', 'row_major'
    DRAM_MEMORY_CONFIG = 'dram'

    def __init__(self, chips=4):
        self.chips = chips
        self.generic = []
        self.freed = []
        self.empties = []
        self.counter = 0
        self.NOC = SimpleNamespace(RISCV_0_default=0, RISCV_1_default=1)
        self.DataMovementProcessor = SimpleNamespace(RISCV_0=0, RISCV_1=1)
        self.MathFidelity = SimpleNamespace(HiFi4='hifi4')

    def tensor(self, shape, dtype='bf16', layout='tile'):
        self.counter += 1
        outer = SimpleNamespace(shape=shape, dtype=dtype, layout=layout, memory_config=lambda: 'dram', name='t%d' % self.counter,
                                device=lambda: 'mesh')
        outer.shards = [SimpleNamespace(buffer_address=lambda chip=chip, base=self.counter: 4096 * base + (1 << 20) * chip)
                        for chip in range(self.chips)]
        return outer

    def empty(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        made = self.tensor(tuple(shape), dtype, layout)
        self.empties.append(made)
        return made

    def get_device_tensors(self, value):
        return value.shards

    def deallocate(self, value):
        self.freed.append(value)

    def generic_op(self, tensors, program):
        self.generic.append((tensors, program))

    CoreCoord = staticmethod(lambda *args: args)
    CoreRange = staticmethod(lambda *args: args)
    CoreRangeSet = staticmethod(lambda args: list(args))
    Tile = staticmethod(lambda args: args)
    TileDescriptor = staticmethod(lambda args: args)
    CBDescriptor = staticmethod(lambda **kwargs: kwargs)
    CBFormatDescriptor = staticmethod(lambda **kwargs: kwargs)
    KernelDescriptor = staticmethod(lambda **kwargs: kwargs)
    DataMovementConfigDescriptor = staticmethod(lambda **kwargs: kwargs)
    ComputeConfigDescriptor = staticmethod(lambda **kwargs: kwargs)
    ProgramDescriptor = staticmethod(lambda **kwargs: kwargs)
    MeshProgramDescriptor = dict
    MeshCoordinate = staticmethod(lambda *args: args)
    MeshCoordinateRange = staticmethod(lambda *args: args)
    TensorAccessorArgs = staticmethod(lambda value: SimpleNamespace(get_compile_time_args=lambda: [1]))

    class RuntimeArgs(dict):
        def __getitem__(self, x):
            return self.setdefault(x, {})


def logits_for(operations, rows, width=62080):
    return operations.tensor((1, 1, rows, width))


class BuilderTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        tp4_sampdraft._LOGGED.clear()
        sarg.REFERENCES.clear()
        sarg.PRODUCED.clear()
        sarg._RESERVED.clear()
        sarg._STATE['rounds'] = 0
        quiet = patch.object(tp4_sampdraft, 'log_line')
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_two_launches_per_call_with_constant_argument_lengths_and_the_reserved_partials_untouched(self):
        lengths = set()
        for rows in (1, 16, 32, 33, 48, 64):
            sarg._RESERVED.clear()
            operations = FakeOperations()
            sarg.reserve(operations, 'mesh')
            logits = logits_for(operations, rows)
            ids, values = sarg.sample(operations, logits, rows)
            self.assertEqual(ids.shape, (1, 1, 1, 64))
            self.assertEqual(values.shape, (1, 1, 1, 64))
            self.assertEqual((ids.dtype, values.dtype), ('u32', 'bf16'))
            self.assertEqual((ids.layout, values.layout), ('row_major', 'row_major'))
            self.assertEqual(len(operations.generic), 2)
            (scan_tensors, scan), (fold_tensors, fold) = operations.generic
            partials = operations.empties[0]
            self.assertEqual(partials.shape, (1, 1, 220, 32))
            self.assertEqual(scan_tensors, [logits, partials])
            self.assertEqual(fold_tensors, [partials, ids, values])
            self.assertEqual(operations.freed, [])
            self.assertEqual(len(operations.empties), 3)     # the reserved partials, ids, values: nothing allocated for the scratch
            self.assertEqual(len(scan), 4)
            self.assertEqual(len(fold), 4)
            for chip, program in enumerate(scan.values()):
                self.assertEqual(len(program['kernels']), 2)
                for role, kernel in enumerate(program['kernels']):
                    self.assertTrue(kernel['kernel_source'].endswith('tp4_shard_argmax_scan.cpp'))
                    self.assertEqual(kernel['config']['processor'], role)
                    self.assertEqual(kernel['config']['noc'], role)
                    cores = [(x, y) for x in kernel['runtime_args'] for y in kernel['runtime_args'][x]]
                    self.assertEqual(len(cores), 110)
                    for x in kernel['runtime_args']:
                        for y, words in kernel['runtime_args'][x].items():
                            lengths.add(('scan', len(words)))
                            self.assertEqual(words[0], logits.shards[chip].buffer_address())
                            self.assertEqual(words[1], partials.shards[chip].buffer_address())
                            self.assertEqual(words[7], role)
                            self.assertEqual(words[8], 1940)
                self.assertEqual([cb['total_size'] for cb in program['cbs']], [2048 * 19, 2048 * 19])
                self.assertEqual([cb['format_descriptors'][0]['buffer_index'] for cb in program['cbs']], [0, 1])
            for chip, program in enumerate(fold.values()):
                words = program['kernels'][0]['runtime_args'][0][0]
                lengths.add(('fold', len(words)))
                self.assertEqual(words[3:], [rows, 110 if rows > 32 else 220, 2 if rows > 32 else 1])
                self.assertEqual(program['cbs'][0]['total_size'], 2048 * 16)
        # the program cache does not hash runtime-arg lengths: every call must carry the same count
        self.assertEqual(lengths, {('scan', 9), ('fold', 6)})

    def test_the_scan_tasks_land_one_per_core_per_risc(self):
        operations = FakeOperations()
        sarg.reserve(operations, 'mesh')
        sarg.sample(operations, logits_for(operations, 64), 64)
        program = next(iter(operations.generic[0][1].values()))
        seen = []
        for role, kernel in enumerate(program['kernels']):
            for x, column in kernel['runtime_args'].items():
                for y, words in column.items():
                    seen.append((words[6], role, 10 * x + y))
        tasks, _, _ = sarg.plan(64, 1940)
        self.assertEqual(sorted(seen), sorted((task[0], task[2], task[1]) for task in tasks))

    def test_a_call_it_cannot_take_returns_none_and_says_why_once(self):
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            operations = FakeOperations()
            sarg.reserve(operations, 'mesh')
            for shape, rows in (((1, 1, 64, 124160), 64), ((1, 1, 32, 62080), 64), ((1, 1, 65, 62080), 65), ((64, 62080), 64)):
                bad = operations.tensor(shape)
                self.assertIsNone(sarg.sample(operations, bad, rows))
            wrong_type = operations.tensor((1, 1, 64, 62080), dtype='bf8')
            self.assertIsNone(sarg.sample(operations, wrong_type, 64))
            self.assertIsNone(sarg.sample(operations, wrong_type, 64))
            placed = operations.tensor((1, 1, 64, 62080))
            placed.memory_config = lambda: 'l1'
            self.assertIsNone(sarg.sample(operations, placed, 64))
        self.assertEqual(operations.generic, [])
        self.assertEqual(len(operations.empties), 1)         # only the reservation
        self.assertTrue(all(line.startswith(tp4_sampdraft.SARG_FALLBACK) for line in lines))
        self.assertEqual(len(lines), len(set(lines)))   # the same reason prints once
        self.assertGreaterEqual(len(lines), 5)

    def test_one_engaged_marker_per_row_count(self):
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            operations = FakeOperations()
            sarg.reserve(operations, 'mesh')
            for rows in (64, 64, 16, 16):
                sarg.sample(operations, logits_for(operations, rows), rows)
        engaged = [line for line in lines if line.startswith(tp4_sampdraft.SARG_ENGAGED)]
        self.assertEqual(len(engaged), 2)
        self.assertIn('rows=64 workers=110 tasks=220 fold=1', engaged[0])

    def test_a_failing_launch_frees_the_outputs_and_keeps_the_reserved_partials(self):
        operations = FakeOperations()
        reserved = sarg.reserve(operations, 'mesh')
        calls = []
        operations.generic_op = lambda tensors, program: (calls.append(1), (_ for _ in ()).throw(RuntimeError('submit')) if len(calls) == 2 else None)
        with self.assertRaises(RuntimeError):
            sarg.sample(operations, logits_for(operations, 64), 64)
        freed = {value.name for value in operations.freed}
        self.assertEqual(freed, {value.name for value in operations.empties} - {reserved.name})

    def test_no_reservation_is_a_logged_fall_back_never_an_allocation_inside_a_capture(self):
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            operations = FakeOperations()
            self.assertIsNone(sarg.sample(operations, logits_for(operations, 64), 64))
        self.assertEqual(operations.empties, [])
        self.assertEqual(operations.generic, [])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(tp4_sampdraft.SARG_FALLBACK))
        self.assertIn('reserved', lines[0])

    def test_the_reservation_is_shared_counted_and_freed_by_its_last_holder(self):
        operations = FakeOperations()
        first = sarg.reserve(operations, 'mesh')
        self.assertIs(sarg.reserve(operations, 'mesh'), first)
        self.assertEqual(len(operations.empties), 1)
        self.assertEqual(first.shape, (1, 1, 220, 32))
        sarg.release_reserved(operations, 'mesh')
        self.assertEqual(operations.freed, [])
        sarg.release_reserved(operations, 'mesh')
        self.assertEqual(operations.freed, [first])
        sarg.release_reserved(operations, 'mesh')          # nothing reserved: a no-op
        self.assertEqual(operations.freed, [first])

    def test_the_audit_holds_todays_outputs_and_a_fallback_holds_nothing(self):
        with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'}):
            operations = FakeOperations()
            sarg.reserve(operations, 'mesh')
            served = Mock(return_value=('old ids', 'old values'))
            ids, values = sarg.sample(operations, logits_for(operations, 16), 16, served=served)
            served.assert_called_once()
            self.assertEqual(sarg.REFERENCES[id(values)], ('old ids', 'old values'))
            sarg.release_audit(operations, values)
            self.assertEqual(sarg.REFERENCES, {})
            self.assertEqual(operations.freed[-2:], ['old ids', 'old values'])

    def test_the_release_also_frees_the_v4a_reference_registered_for_todays_values(self):
        import verify_trace_t1
        with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'}):
            operations = FakeOperations()
            sarg.reserve(operations, 'mesh')
            old_values = object()
            served = Mock(return_value=('old ids', old_values))
            _, values = sarg.sample(operations, logits_for(operations, 16), 16, served=served)
            verify_trace_t1.VALUE_REFERENCES[id(old_values)] = 'ttnn max reference'
            sarg.release_audit(operations, values)
            self.assertEqual(verify_trace_t1.VALUE_REFERENCES, {})
            self.assertIn('ttnn max reference', operations.freed)
            self.assertEqual(sarg.PRODUCED, {})

    def test_a_listed_output_is_kept_alive_so_its_id_cannot_be_reused(self):
        operations = FakeOperations()
        sarg.reserve(operations, 'mesh')
        _, values = sarg.sample(operations, logits_for(operations, 16), 16)
        self.assertIs(sarg.PRODUCED[id(values)], values)


class AuditTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1',
                                          tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        sarg.REFERENCES.clear()
        sarg._STATE['rounds'] = 0

    def operations(self, ids, values):
        shards_ids = [SimpleNamespace(data=torch.tensor(ids)) for _ in range(4)]
        shards_values = [SimpleNamespace(data=torch.tensor(values, dtype=torch.float32)) for _ in range(4)]
        return SimpleNamespace(get_device_tensors=lambda tensor: tensor, to_torch=lambda part: part.data), shards_ids, shards_values

    def test_equal_rows_pass_and_log_one_line(self):
        operations, held_ids, held_values = self.operations([5, 6, 7], [1.0, 2.0, 3.0])
        sarg.REFERENCES[id('v')] = (held_ids, held_values)
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            sarg.audit_round(operations, 'v', 3, [torch.tensor([5, 6, 7])] * 4,
                             [torch.tensor([1.0, 2.0, 3.0])] * 4)
        self.assertEqual(lines, ['%s 1 exact=True rows=3 chips=4' % tp4_sampdraft.SARG_AUDIT])

    def test_a_different_id_raises_and_logs_the_mismatch_marker(self):
        operations, held_ids, held_values = self.operations([5, 6, 7], [1.0, 2.0, 3.0])
        sarg.REFERENCES[id('v')] = (held_ids, held_values)
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append), self.assertRaises(AssertionError):
            sarg.audit_round(operations, 'v', 3, [torch.tensor([5, 6, 8])] * 4, [torch.tensor([1.0, 2.0, 3.0])] * 4)
        self.assertTrue(lines[0].startswith(tp4_sampdraft.SARG_MISMATCH))
        self.assertIn('rows=[2]', lines[0])

    def test_values_compare_as_numbers_and_nan_equals_nan(self):
        operations, held_ids, held_values = self.operations([1, 2], [0.0, float('nan')])
        sarg.REFERENCES[id('v')] = (held_ids, held_values)
        with patch.object(tp4_sampdraft, 'log_line'):
            sarg.audit_round(operations, 'v', 2, [torch.tensor([1, 2])] * 4, [torch.tensor([-0.0, float('nan')])] * 4)
            with self.assertRaises(AssertionError):
                sarg.audit_round(operations, 'v', 2, [torch.tensor([1, 2])] * 4, [torch.tensor([0.5, float('nan')])] * 4)

    def test_an_audited_output_with_no_reference_raises(self):
        with patch.object(tp4_sampdraft, 'log_line'), self.assertRaises(AssertionError):
            sarg.audit_round(SimpleNamespace(), 'unknown', 3, [], [])

    def test_the_audit_is_a_no_op_when_the_audit_flag_is_off(self):
        with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX_AUDIT: '0'}):
            sarg.audit_round(SimpleNamespace(), 'unknown', 3, [], [])


class FlagOffTests(unittest.TestCase):
    """With the flag unset sample_shards is today's calls, byte for byte; with it set and a call the kernels refuse, the same."""

    def fake(self):
        calls = []
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', ROW_MAJOR_LAYOUT='row_major', TILE_LAYOUT='tile', bfloat16='bf16')
        operations.to_layout = Mock(side_effect=lambda value, layout, memory_config=None: (
            calls.append(('to_layout', layout, memory_config)) or SimpleNamespace(name='row-major', shape=value.shape)))
        operations.argmax = Mock(side_effect=lambda value, **options: (
            calls.append(('argmax', value.name, tuple(sorted(options.items())))) or 'ids'))
        operations.max = Mock(side_effect=lambda value, **options: (
            calls.append(('max', value.name, tuple(sorted(options.items())))) or 'values'))
        operations.deallocate = Mock(side_effect=lambda value: calls.append(('free', value.name)))
        return operations, calls

    TODAYS = [('to_layout', 'row_major', 'dram'),
              ('argmax', 'row-major', (('dim', 3), ('keepdim', True), ('memory_config', 'dram'))),
              ('max', 'logits', (('dim', 3), ('keepdim', True), ('memory_config', 'dram'))),
              ('free', 'row-major')]

    def run_case(self, environ, shape=(1, 1, 64, 62080), rows=64, placement='dram'):
        operations, calls = self.fake()
        logits = SimpleNamespace(name='logits', shape=shape, dtype='bf16', layout='tile', memory_config=lambda: placement)
        with patch.dict(os.environ, environ, clear=False):
            result = t1.sample_shards(operations, logits, rows)
        return result, calls

    def test_flag_unset_or_zero_is_todays_call_sequence(self):
        for environ in ({'QWEN_FAST_TP': '4'}, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '0'}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(tp4_sampdraft.SHARD_ARGMAX, None)
                result, calls = self.run_case(environ)
            self.assertEqual(result, ('ids', 'values'))
            self.assertEqual(calls, self.TODAYS)

    def test_flag_on_with_a_refused_call_is_todays_call_sequence(self):
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            tp4_sampdraft._LOGGED.clear()
            result, calls = self.run_case({'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'}, placement='l1')
        self.assertEqual(result, ('ids', 'values'))
        self.assertEqual(calls, self.TODAYS)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith(tp4_sampdraft.SARG_FALLBACK))

    def test_the_old_shape_dtype_and_layout_refusals_come_first(self):
        operations, calls = self.fake()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'}):
            with self.assertRaisesRegex(ValueError, 'pre-gather'):
                t1.sample_shards(operations, SimpleNamespace(name='logits', shape=(1, 1, 64, 124160), dtype='bf16',
                                                             layout='tile'), 64)
            with self.assertRaisesRegex(ValueError, 'bf16 TILE'):
                t1.sample_shards(operations, SimpleNamespace(name='logits', shape=(1, 1, 64, 62080), dtype='bf8_b',
                                                             layout='tile'), 64)
        self.assertEqual(calls, [])

    def test_the_kernel_result_is_returned_when_it_engages(self):
        operations, calls = self.fake()
        logits = SimpleNamespace(name='logits', shape=(1, 1, 64, 62080), dtype='bf16', layout='tile')
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'}), \
                patch('tp4_shard_argmax.sample', return_value=('kernel ids', 'kernel values')) as sample:
            self.assertEqual(t1.sample_shards(operations, logits, 64), ('kernel ids', 'kernel values'))
        sample.assert_called_once()
        self.assertEqual(calls, [])

    def test_the_audit_closure_runs_todays_path(self):
        operations, calls = self.fake()
        logits = SimpleNamespace(name='logits', shape=(1, 1, 64, 62080), dtype='bf16', layout='tile')

        def engaged(ops, tensor, rows, served=None):
            self.assertEqual(served(), ('ids', 'values'))
            return 'kernel ids', 'kernel values'

        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'}), \
                patch('tp4_shard_argmax.sample', side_effect=engaged):
            self.assertEqual(t1.sample_shards(operations, logits, 64), ('kernel ids', 'kernel values'))
        self.assertEqual(calls, self.TODAYS)


class PackedVerifierCouplingTests(unittest.TestCase):
    """The packed block's V4a value audit and its release path know which outputs the kernels made."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1', 'QWEN_FAST_TP4_SHARD_VALUES': '1',
                                          'QWEN_FAST_TP4_VGLUE_AUDIT': '1', 'QWEN_FAST_TP4_GDN_GLUE': '0'})
        patcher.start()
        self.addCleanup(patcher.stop)
        sarg.PRODUCED.clear()
        sarg.REFERENCES.clear()
        t1.VALUE_REFERENCES.clear()

    def test_the_value_audit_leaves_kernel_made_maxima_to_the_kernel_audit(self):
        import packed_verifier
        values = object()
        sarg.PRODUCED[id(values)] = values
        packed_verifier.audit_shard_values(None, (None, None, values), 3, [])      # returns: nothing to compare, nothing raised

    def test_the_value_audit_still_fails_a_served_gather_that_recorded_no_reference(self):
        import packed_verifier
        with patch.object(packed_verifier, 'diagnostic'), self.assertRaisesRegex(AssertionError, 'no reference'):
            packed_verifier.audit_shard_values(None, (None, None, object()), 3, [])

    def test_the_release_forgets_the_output_and_frees_what_the_audit_held(self):
        import packed_verifier
        operations = FakeOperations()
        values = object()
        sarg.PRODUCED[id(values)] = values
        sarg.REFERENCES[id(values)] = ('old ids', 'old values')
        packed_verifier.release_sampdraft_audit(operations, values)
        self.assertEqual((sarg.PRODUCED, sarg.REFERENCES), ({}, {}))
        self.assertEqual(operations.freed, ['old ids', 'old values'])

    def test_the_audit_round_runs_for_the_audited_flag_only(self):
        import packed_verifier
        with patch('tp4_shard_argmax.audit_round') as audit:
            packed_verifier.audit_sampdraft('ops', (None, None, 'values'), 3, ['ids'], ['vals'])
            audit.assert_not_called()
            with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'}):
                packed_verifier.audit_sampdraft('ops', (None, None, 'values'), 3, ['ids'], ['vals'])
            audit.assert_called_once_with('ops', 'values', 3, ['ids'], ['vals'])

    def test_the_block_warm_reserves_the_partials_once_and_the_close_gives_the_holder_back(self):
        import packed_verifier
        operations = FakeOperations()
        sarg._RESERVED.clear()
        self.assertTrue(packed_verifier.reserve_sampdraft(operations, 'mesh'))
        self.assertTrue(packed_verifier.reserve_sampdraft(operations, 'mesh'))       # a second block: the same buffer
        self.assertEqual(len(operations.empties), 1)
        packed_verifier.release_sampdraft_reserve(operations, 'mesh')
        self.assertEqual(operations.freed, [])
        packed_verifier.release_sampdraft_reserve(operations, 'mesh')
        self.assertEqual(operations.freed, operations.empties)

    def test_the_warm_reserves_nothing_with_the_lever_off(self):
        import packed_verifier
        operations = FakeOperations()
        with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX: '0'}):
            self.assertFalse(packed_verifier.reserve_sampdraft(operations, 'mesh'))
        self.assertEqual(operations.empties, [])

    def test_an_audit_without_its_lever_fails_at_the_block_warm_not_at_a_readback(self):
        import packed_verifier
        for lever, audit in ((tp4_sampdraft.SHARD_ARGMAX, tp4_sampdraft.SHARD_ARGMAX_AUDIT),
                             (tp4_sampdraft.DRAFT_CONV, tp4_sampdraft.DRAFT_CONV_AUDIT),
                             (tp4_sampdraft.DRAFT_HEADS, tp4_sampdraft.DRAFT_HEADS_AUDIT)):
            with patch.dict(os.environ, {lever: '0', audit: '1'}), self.assertRaises(ValueError):
                packed_verifier.reserve_sampdraft(FakeOperations(), 'mesh')

    def test_the_audit_round_leaves_the_drafter_audits_to_their_buckets(self):
        import packed_verifier
        with patch.dict(os.environ, {tp4_sampdraft.DRAFT_CONV: '1', tp4_sampdraft.DRAFT_CONV_AUDIT: '1'}), \
                patch('tp4_draft_conv.compare_scope') as compare, patch('tp4_shard_argmax.audit_round') as audit:
            packed_verifier.audit_sampdraft('ops', (None, None, 'values'), 3, ['ids'], ['vals'])
        compare.assert_not_called()
        audit.assert_not_called()
        self.assertFalse(hasattr(__import__('tp4_draft_conv'), 'compare_pending'))
        self.assertFalse(hasattr(__import__('tp4_draft_conv'), 'release_audit'))

    def test_the_request_engine_refuses_the_lever_instead_of_half_engaging_it(self):
        import verifier_engine_tp
        engine = verifier_engine_tp.VerifierEngine.__new__(verifier_engine_tp.VerifierEngine)
        with patch.dict(os.environ, {verifier_engine_tp.SHARD_FLAG: '1'}), self.assertRaisesRegex(ValueError, 'request engine'):
            engine.__init__()


if __name__ == '__main__':
    unittest.main()
