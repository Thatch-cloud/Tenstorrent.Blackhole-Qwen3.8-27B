"""tp4_shard_argmax (QWEN_FAST_TP4_SHARD_ARGMAX, S1): the scan and fold kernels transliterated and held against torch.argmax, the
launch plan, the flag-off path being today's calls, the refusals and the audit.

The C++ kernels (tp4_shard_argmax_scan.cpp, tp4_shard_argmax_fold.cpp and the two-level tp4_shard_argmax_fold2.cpp) cannot run on this
machine. `simulate_scan`, `simulate_fold`, `simulate_fold_level1` and `simulate_fold_level2` below are their line-for-line Python
transliterations over the same tile-face word layout, so the tie-breaking, the fast path's preconditions, the slow path, the fold order, the
tree's equality with the one-core fold (on the plan's real tasks and on random record tables) and the joined words are proved on the CPU;
the card harness (optimisation/ttnn-op/shard_argmax) and the audited gate prove the transliteration is the kernel.

Fusion programme WP1 additions: the tree fold, the words output (the readback layout shared with the W3 mesh-read work), the device grid
(11 x 10 and 13 x 10), the FOLD2 flag and the smoke rule."""

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
    """tp4_shard_argmax_fold.cpp: `pages` the partials pages in task order. Returns (ids, values, words) lists of 64."""
    ids, values, words = [0] * 64, [0] * 64, [0] * 64
    for row in range(rows):
        tile_row, word = row >> 5, row & 31
        best_key = 0
        column = bits = 0
        for task in range(per_tile_row):
            record = pages[tile_row * per_tile_row + task][word]
            key = order_key(record & 0xFFFF)
            if key > best_key:
                best_key, column, bits = key, record >> 16, record & 0xFFFF
        ids[row], values[row], words[row] = column, bits, (column << 16) | bits
    return ids, values, words


def simulate_fold_level1(pages, rows, per_tile_row, tile_rows, groups, group):
    """tp4_shard_argmax_fold2.cpp, FOLD2_LEVEL == 1, one core: this group's tasks of every tile row, folded per live row. Returns the
    64-word group page."""
    first = group * per_tile_row // groups
    last = (group + 1) * per_tile_row // groups
    count = last - first
    scratch = []                                   # slot tile_row * count + i holds page (tile_row * per_tile_row + first + i)
    for tile_row in range(tile_rows):
        for i in range(count):
            scratch.append(pages[tile_row * per_tile_row + first + i])
    out = [0] * 64
    for row in range(64):
        winner = 0
        if row < rows:
            tile_row, word = row >> 5, row & 31
            best_key = 0
            for i in range(count):
                record = scratch[tile_row * count + i][word]
                key = order_key(record & 0xFFFF)
                if key > best_key:
                    best_key, winner = key, record
        out[row] = winner
    return out


def simulate_fold_level2(group_pages, rows, groups):
    """tp4_shard_argmax_fold2.cpp, FOLD2_LEVEL == 2, one core: the group pages folded in ascending group order. Returns (ids, values, words)."""
    ids, values, words = [0] * 64, [0] * 64, [0] * 64
    for row in range(64):
        column = bits = 0
        if row < rows:
            best_key = 0
            for group in range(groups):
                record = group_pages[group][row]
                key = order_key(record & 0xFFFF)
                if key > best_key:
                    best_key, column, bits = key, record >> 16, record & 0xFFFF
        ids[row], values[row], words[row] = column, bits, (column << 16) | bits
    return ids, values, words


def simulate_tree(pages, rows, per_tile_row, tile_rows, groups=None):
    groups = sarg.FOLD2_GROUPS if groups is None else groups
    level1 = [simulate_fold_level1(pages, rows, per_tile_row, tile_rows, groups, group) for group in range(groups)]
    return simulate_fold_level2(level1, rows, groups)


def scan_pages(rows_bits, tile_columns, workers=110):
    """The scan tasks of plan() over `rows_bits` (a rows x 32 * tile_columns matrix of bit patterns): (pages, per_tile_row, tile_rows)."""
    rows = len(rows_bits)
    tasks, per_tile_row, tile_rows = sarg.plan(rows, tile_columns, workers)
    pages = [None] * len(tasks)
    cache = {}
    for task, worker, role, tile_row, first, last, live in tasks:
        tiles = []
        for column in range(first, last):
            if (tile_row, column) not in cache:
                cache[(tile_row, column)] = tile_words(rows_bits, tile_row, column)
            tiles.append(cache[(tile_row, column)])
        pages[task] = simulate_scan(tiles, live, first)
    return pages, per_tile_row, tile_rows


def run_kernels_full(rows_bits, tile_columns, fold='single', workers=110):
    """The scan tasks, then the fold ('single' or 'tree'): (ids, values, words)."""
    pages, per_tile_row, tile_rows = scan_pages(rows_bits, tile_columns, workers)
    rows = len(rows_bits)
    if fold == 'tree':
        return simulate_tree(pages, rows, per_tile_row, tile_rows)
    return simulate_fold(pages, rows, per_tile_row)


def run_kernels(rows_bits, tile_columns, fold='single', workers=110):
    """(ids, values) of run_kernels_full."""
    ids, values, _ = run_kernels_full(rows_bits, tile_columns, fold, workers)
    return ids, values


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
    """The transliterated kernels against torch.argmax: ids AND the winning element's bits. FOLD picks the fold: the one-core kernel or the
    two-level tree (TreeKernelTests below runs every test of this class again on the tree)."""

    FOLD = 'single'
    WORKERS = 110

    def run_kernels(self, rows_bits, tile_columns):
        return run_kernels(rows_bits, tile_columns, self.FOLD, self.WORKERS)

    def check(self, matrix):
        ids, values = self.run_kernels(matrix.rows, matrix.tile_columns)
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
                ids, values = self.run_kernels(matrix.rows, columns)
                self.assertEqual(ids[row], first, (rows, row, first, second))
                self.assertEqual(values[row], top)

    def test_a_tie_across_the_tile_row_seam_belongs_to_each_row_alone(self):
        matrix = Matrix(64, 220, seed=5)
        top = POSITIVE(40.0)
        for row in (0, 31, 32, 63):
            matrix.put(row, 11, top).put(row, 7000, top)
        ids, _ = self.run_kernels(matrix.rows, 220)
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
            ids, values = self.run_kernels(matrix.rows, 220)
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
        ids, _ = self.run_kernels(matrix.rows, 220)
        self.assertEqual(ids[3], 77)
        self.check(matrix)
        ids, values = self.run_kernels([[0xFF80] * (32 * 220)] * 2, 220)
        self.assertEqual(ids[:2], [0, 0])

    def test_a_nan_wins_whether_it_comes_before_or_after_the_maximum_and_the_first_nan_wins(self):
        for nan_column, top_column in ((5, 6000), (6000, 5), (31, 32), (3, 4)):
            matrix = Matrix(16, 220, seed=17)
            top = POSITIVE(55.0)
            for row in range(16):
                matrix.put(row, top_column, top).put(row, nan_column, NAN)
            matrix.put(1, nan_column + 1, 0xFFC1)           # a second, negative NaN later: the first still wins
            matrix.put(2, 4000, NAN)                        # a NaN far after: the earlier one still wins
            ids, values = self.run_kernels(matrix.rows, 220)
            self.assertEqual(ids[:16], [min(nan_column, 4000) if row == 2 else nan_column for row in range(16)][:16])
            self.assertTrue(all((value & 0x7F80) == 0x7F80 and value & 0x7F for value in values[:16]))
            self.check(matrix)

    def test_a_nan_only_in_one_task_still_moves_only_its_own_row(self):
        matrix = Matrix(64, 220, seed=19)
        matrix.put(40, 6500, NAN)
        ids, _ = self.run_kernels(matrix.rows, 220)
        want, _ = reference(matrix.rows)
        self.assertEqual(ids[40], 6500)
        self.assertEqual([ids[row] for row in range(64) if row != 40], [want[row] for row in range(64) if row != 40])

    def test_denormals_and_tiny_values_order_by_magnitude(self):
        matrix = Matrix(4, 220, seed=23)
        for row in range(4):
            matrix.rows[row] = [0x8000 | (1 + column % 3) for column in range(len(matrix.rows[row]))]
            matrix.put(row, 400 + row, 0x0003).put(row, 401 + row, 0x0002)
        self.check(matrix)




class TreeKernelTests(KernelTests):
    """Every KernelTests case again with the two-level tree fold in place of the one-core fold."""

    FOLD = 'tree'


class WideGridKernelTests(unittest.TestCase):
    """The 13 x 10 grid (130 workers, 260 tasks, runs of 14 and 15 tile columns): the same answers from either fold."""

    def test_ties_across_worker_runs_and_group_blocks_at_130_workers(self):
        columns = 260                                   # 2 tile columns a worker: runs of 2 over 130 workers
        runs = sarg.column_runs(columns, 130)
        top = POSITIVE(31.0)
        for rows in (16, 64):
            matrix = Matrix(rows, columns, seed=40 + rows)
            for row in range(rows):
                first_run = (row * 11) % 129
                a = 32 * runs[first_run][1] - 1             # the last column of one worker's run
                b = 32 * runs[first_run + 1][0]             # the first of the next
                matrix.put(row, a, top).put(row, b, top)
            for fold in ('single', 'tree'):
                ids, values = run_kernels(matrix.rows, columns, fold, 130)
                want, bits = reference(matrix.rows)
                self.assertEqual(ids[:rows], want, (rows, fold))
                self.assertEqual(values[:rows], bits, (rows, fold))

    def test_real_shard_width_at_130_workers_tree_equals_single_equals_torch(self):
        matrix = Matrix(64, 1940, seed=29)
        matrix.put(3, 61000, POSITIVE(60.0)).put(3, 61001, POSITIVE(60.0)).put(40, 12, NAN)
        pages, per_tile_row, tile_rows = scan_pages(matrix.rows, 1940, 130)
        self.assertEqual((per_tile_row, tile_rows, len(pages)), (130, 2, 260))
        single = simulate_fold(pages, 64, per_tile_row)
        tree = simulate_tree(pages, 64, per_tile_row, tile_rows)
        self.assertEqual(single, tree)
        want_ids, want_bits = reference(matrix.rows)
        self.assertEqual(single[0][:64], want_ids)
        self.assertEqual(single[1][:64], want_bits)


class TreeFoldTests(unittest.TestCase):
    """The tree against the one-core fold on RANDOM record tables, independent of the scan: every record is a (column, bits) word whose
    columns ascend with the task, drawn from a small alphabet so that equal keys (ties, -0 against +0, many NaNs) are the common case."""

    ALPHABET = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x8001, 0x4000, 0x40A0]

    def table(self, generator, per_tile_row, tile_rows, alphabet):
        pages = []
        for tile_row in range(tile_rows):
            for task in range(per_tile_row):
                page = []
                for word in range(32):
                    bits = alphabet[int(torch.randint(0, len(alphabet), (1,), generator=generator))]
                    column = task * 600 + int(torch.randint(0, 600, (1,), generator=generator))
                    page.append((column << 16) | bits)
                pages.append(page)
        return pages

    def test_the_tree_returns_the_one_core_folds_record_for_every_row_count_task_count_and_alphabet(self):
        generator = torch.Generator().manual_seed(2026)
        cases = 0
        for per_tile_row, tile_rows in ((110, 2), (220, 1), (130, 2), (260, 1), (8, 2), (9, 1)):
            for rows in ((1, 7, 32, 33, 64) if tile_rows == 2 else (1, 5, 32)):
                for alphabet in (self.ALPHABET, self.ALPHABET[:3], self.ALPHABET[:1], self.ALPHABET[6:8]):
                    pages = self.table(generator, per_tile_row, tile_rows, alphabet)
                    single = simulate_fold(pages, rows, per_tile_row)
                    tree = simulate_tree(pages, rows, per_tile_row, tile_rows)
                    self.assertEqual(single, tree, (per_tile_row, tile_rows, rows, alphabet[:3]))
                    cases += 1
        self.assertGreater(cases, 70)

    def test_a_tie_that_straddles_a_group_boundary_keeps_the_earlier_group(self):
        for per_tile_row, tile_rows in ((110, 2), (220, 1)):
            blocks = sarg.group_plan(per_tile_row)
            for boundary in [first for _group, first, _last in blocks[1:]]:
                pages = [[((task * 600) << 16) | 0x3F80 for _word in range(32)] for task in range(per_tile_row * tile_rows)]
                for tile_row in range(tile_rows):
                    pages[tile_row * per_tile_row + boundary - 1] = [(1 << 16) | 0x4000] * 32      # the later of the pair in group g - 1... earlier task
                    pages[tile_row * per_tile_row + boundary] = [(2 << 16) | 0x4000] * 32           # an equal key in the next group
                ids, values, _ = simulate_tree(pages, 32 * tile_rows, per_tile_row, tile_rows)
                self.assertEqual(ids[:32 * tile_rows], [1] * (32 * tile_rows), (per_tile_row, boundary))
                self.assertEqual(simulate_fold(pages, 32 * tile_rows, per_tile_row)[0][:32 * tile_rows], [1] * (32 * tile_rows))

    def test_the_group_plan_partitions_the_tasks_into_ascending_non_empty_blocks(self):
        for per_tile_row in (8, 9, 110, 130, 220, 260):
            blocks = sarg.group_plan(per_tile_row)
            self.assertEqual(len(blocks), sarg.FOLD2_GROUPS)
            flat = [task for _group, first, last in blocks for task in range(first, last)]
            self.assertEqual(flat, list(range(per_tile_row)))
            self.assertTrue(all(last > first for _group, first, last in blocks))
        for bad in (0, 7):
            with self.assertRaises(ValueError):
                sarg.group_plan(bad)

    def test_the_scratch_bound_holds_for_every_row_regime_and_grid(self):
        for workers in (110, 130, 140, 64):
            task_pages = 2 * workers
            bound = sarg.group_pages_bound(task_pages, sarg.FOLD2_GROUPS)
            for per_tile_row, tile_rows in ((workers, 2), (2 * workers, 1)):
                for _group, first, last in sarg.group_plan(per_tile_row):
                    self.assertLessEqual(tile_rows * (last - first), bound, (workers, per_tile_row))
        self.assertEqual(sarg.group_pages_bound(220, 8), 28)
        self.assertEqual(sarg.group_pages_bound(260, 8), 34)


class WordsTests(unittest.TestCase):
    """The joined (column << 16) | bits layout shared with the mesh-read work."""

    def test_the_kernels_words_are_their_ids_and_value_bits_joined(self):
        matrix = Matrix(64, 220, seed=31)
        matrix.put(5, 7000, POSITIVE(70.0)).put(6, 7001, NAN).put(7, 33000 % (32 * 220), 0x8001)
        for fold in ('single', 'tree'):
            ids, values, words = run_kernels_full(matrix.rows, 220, fold)
            self.assertEqual(words, [(ids[row] << 16) | values[row] for row in range(64)])
            self.assertEqual(words[64 - 1], (ids[63] << 16) | values[63])
            rows = len(matrix.rows)
            self.assertTrue(all(word < (1 << 32) for word in words[:rows]))
        _, _, short = run_kernels_full(Matrix(16, 220, seed=2).rows, 220)
        self.assertEqual(short[16:], [0] * 48)                       # rows at and past `rows` are zero

    def test_decode_words_round_trips_columns_past_the_sign_bit_and_every_bit_pattern(self):
        columns = [0, 1, 32767, 32768, 40000, 62079, 65535]
        patterns = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x8001, 0x7F7F, 0xFF7F]
        ids = torch.tensor([columns[i % len(columns)] for i in range(len(patterns))], dtype=torch.int64)
        bits = torch.tensor(patterns, dtype=torch.int64)
        words = (ids << 16) | bits
        as_int32 = words.to(torch.int64).where(words < (1 << 31), words - (1 << 32)).to(torch.int32)    # to_torch of a uint32 tensor
        decoded_ids, decoded_values = sarg.decode_words(as_int32)
        self.assertEqual(decoded_ids.tolist(), ids.tolist())
        got = decoded_values.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        self.assertEqual(got.tolist(), patterns)
        self.assertEqual(sarg.decode_words(as_int32, 5)[0].tolist(), ids[:5].tolist())
        self.assertEqual(sarg.pack_words(ids, decoded_values).tolist(), words.tolist())

    def test_pack_words_matches_the_kernel_word_for_real_bf16_tensors(self):
        values = torch.tensor([1.0, -2.5, 0.0, float('inf')], dtype=torch.bfloat16)
        ids = torch.tensor([3, 62079, 0, 40000], dtype=torch.int32)
        packed = sarg.pack_words(ids, values).tolist()
        bits = bits_of(values.float())
        self.assertEqual(packed, [(int(i) << 16) | b for i, b in zip(ids.tolist(), bits)])


class SourceTests(unittest.TestCase):
    """The C++ sources: one order key in three places, the defines the builders pass, no stray literal grid."""

    def source(self, name):
        from pathlib import Path
        return (Path(sarg.__file__).with_name(name)).read_text()

    def key_function(self, text):
        import re
        match = re.search(r'inline uint32_t order_key\(uint32_t bits\) \{.*?\n\}', text, re.S)
        self.assertIsNotNone(match)
        return re.sub(r'\s+', ' ', match.group(0))

    def test_all_three_kernels_use_the_same_order_key_as_the_transliteration(self):
        keys = {name: self.key_function(self.source(name)) for name in (sarg.SCAN_KERNEL, sarg.FOLD_KERNEL, sarg.FOLD2_KERNEL)}
        self.assertEqual(len(set(keys.values())), 1, keys)
        for text in ('0x7fffu', '0x7f80u', '0x10000u', '0x8000u'):
            self.assertIn(text, keys[sarg.SCAN_KERNEL])

    def test_the_fold_kernels_take_their_sizes_from_defines_the_builders_pass(self):
        fold, fold2 = self.source(sarg.FOLD_KERNEL), self.source(sarg.FOLD2_KERNEL)
        self.assertIn('FOLD_MAX_PAGES', fold)
        self.assertIn('FOLD2_LEVEL', fold2)
        self.assertIn('FOLD2_GROUP_PAGES', fold2)
        self.assertIn('#error', fold2)                    # a build with no level is an error, not a silent level 2
        self.assertNotIn('MAX_PAGES = 220', fold)

    def test_every_output_page_is_written_by_both_folds(self):
        for name in (sarg.FOLD_KERNEL, sarg.FOLD2_KERNEL):
            text = self.source(name)
            for needle in ('out_ids', 'out_values', 'out_words', 'words.get_noc_addr(0)', 'out_words[row] = (column << 16) | bits'):
                self.assertIn(needle, text, (name, needle))


class ServedCompositionTests(unittest.TestCase):
    """S1 against the served composition (Untilize + ArgMax + Gather) at the real (64, 62080) per-shard shape, and after the cross-chip merge.

    The served path is modelled on the host as what it is defined to return: ttnn.argmax's first index holding the maximum along the last
    dimension (torch.argmax's rule: a tie keeps the lowest column, -0 equals +0) and the V4a gather's logits[row, id], the maximum element's
    own bits. The four shards are one (64, 4 x 62080) matrix cut into per-chip (64, 62080) shards, so the cross-chip merge is held against
    torch.argmax over the whole 248,320-wide row too (the pinned sampler's rule), including ties between chips, a tie across the shard seam,
    and a NaN in a later shard."""

    SHARD = 62080

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def served(self, shard_rows):
        """(ids, value bits) of the served composition for one shard given as rows of bit patterns."""
        ids, _ = reference(shard_rows)
        return ids, [shard_rows[row][column] for row, column in enumerate(ids)]

    def merged(self, per_chip):
        """combine_shards over per-chip (ids, value bits) lists, values as bf16 numbers (what the readback hands it)."""
        chip_ids = [torch.tensor(ids, dtype=torch.int32) for ids, _ in per_chip]
        chip_values = [torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16) for _, bits in per_chip]
        return t1.combine_shards(chip_ids, chip_values).tolist()

    def build(self):
        rows = 64
        generator = torch.Generator().manual_seed(77)
        matrix = bits_of(torch.randn(rows, 4 * self.SHARD, generator=generator) * 3.0)
        top = POSITIVE(90.0)
        high = POSITIVE(95.0)
        for row in range(rows):
            kind = row % 8
            if kind == 0:      # the same maximum in every shard: the lowest chip wins
                for chip in range(4):
                    matrix[row][chip * self.SHARD + 100 + row] = top
            elif kind == 1:    # in shards 1 and 3 only
                matrix[row][1 * self.SHARD + 7] = top
                matrix[row][3 * self.SHARD + 7] = top
            elif kind == 2:    # a tie across the seam between shards 0 and 1: the last column of 0 against the first of 1
                matrix[row][1 * self.SHARD - 1] = top
                matrix[row][1 * self.SHARD] = top
            elif kind == 3:    # ties inside one shard at the places boundaries hide them
                matrix[row][2 * self.SHARD + 15] = top
                matrix[row][2 * self.SHARD + 16] = top
                matrix[row][2 * self.SHARD + self.SHARD - 1] = top
            elif kind == 4:    # a higher maximum in a later shard beats an earlier one
                matrix[row][0 * self.SHARD + 5] = top
                matrix[row][3 * self.SHARD + 5] = high
            elif kind == 5:    # a NaN in the last shard beats a finite maximum in the first
                matrix[row][0 * self.SHARD + 9] = high
                matrix[row][3 * self.SHARD + 123] = NAN
            elif kind == 6:    # an all-negative row with a tie of its largest value in two shards
                matrix[row] = [bits | 0x8000 if bits & 0x7FFF else 0x8001 for bits in matrix[row]]
                matrix[row][1 * self.SHARD + 31] = 0x8001
                matrix[row][2 * self.SHARD + 31] = 0x8001
            else:              # +-0 tie at the top of the row's best shard
                matrix[row] = [0x8001] * len(matrix[row])
                matrix[row][3 * self.SHARD + 40] = NEGATIVE_ZERO
                matrix[row][3 * self.SHARD + 41] = POSITIVE_ZERO
        return [[row[chip * self.SHARD:(chip + 1) * self.SHARD] for row in matrix] for chip in range(4)], matrix

    def test_each_shard_equals_the_served_composition_and_so_does_the_merge(self):
        shards, whole = self.build()
        tile_columns = self.SHARD // 32
        per_chip_served, per_chip = [], {'single': [], 'tree': []}
        for chip in range(4):
            pages, per_tile_row, tile_rows = scan_pages(shards[chip], tile_columns)
            single = simulate_fold(pages, 64, per_tile_row)
            tree = simulate_tree(pages, 64, per_tile_row, tile_rows)
            self.assertEqual(single, tree, chip)
            served_ids, served_bits = self.served(shards[chip])
            # the NaN rows follow torch.argmax (the first NaN), which is what combine_shards expects; every other row is the served path's
            self.assertEqual(single[0][:64], served_ids, chip)
            self.assertEqual(single[1][:64], served_bits, chip)
            per_chip_served.append((served_ids, served_bits))
            for fold, result in (('single', single), ('tree', tree)):
                per_chip[fold].append((result[0][:64], result[1][:64]))
        want = torch.argmax(torch.tensor(whole, dtype=torch.int32).to(torch.int16).view(torch.bfloat16), dim=1).tolist()
        merged_served = self.merged(per_chip_served)
        self.assertEqual(merged_served, want)
        for fold in ('single', 'tree'):
            self.assertEqual(self.merged(per_chip[fold]), merged_served, fold)

    def test_the_cross_chip_rules_in_the_built_matrix_are_the_ones_the_rows_claim(self):
        shards, whole = self.build()
        want = torch.argmax(torch.tensor(whole, dtype=torch.int32).to(torch.int16).view(torch.bfloat16), dim=1).tolist()
        # kind 0: chip 0 wins a four-way tie; kind 1: chip 1; kind 2: the seam tie goes to chip 0's last column; kind 4: the later, higher max
        self.assertEqual(want[0], 100 + 0)
        self.assertEqual(want[1], 1 * self.SHARD + 7)
        self.assertEqual(want[2], self.SHARD - 1)
        self.assertEqual(want[4], 3 * self.SHARD + 5)
        self.assertEqual(want[5], 3 * self.SHARD + 123)          # the NaN
        self.assertEqual(want[3], 2 * self.SHARD + 15)


class SmokeRuleTests(unittest.TestCase):
    """tp4_shard_argmax_smoke.problems: the rule WP0 hooks into the smoke check."""

    ON = {tp4_sampdraft.SHARD_ARGMAX: '1'}
    ENGAGED_1 = '%s rows=64 workers=110 tasks=220 fold=1 audit=0 words=1' % tp4_sampdraft.SARG_ENGAGED
    ENGAGED_2 = '%s rows=64 workers=130 tasks=260 fold=2 audit=0 words=1' % tp4_sampdraft.SARG_ENGAGED
    AUDIT_LINE = '%s 7 exact=True rows=64 chips=4' % tp4_sampdraft.SARG_AUDIT

    def test_the_markers_are_the_modules(self):
        import tp4_shard_argmax_smoke as smoke
        self.assertEqual((smoke.ENGAGED, smoke.FELL_BACK, smoke.AUDIT_PASSED, smoke.MISMATCH),
                         (tp4_sampdraft.SARG_ENGAGED, tp4_sampdraft.SARG_FALLBACK, tp4_sampdraft.SARG_AUDIT, tp4_sampdraft.SARG_MISMATCH))
        self.assertEqual((smoke.SHARD_ARGMAX, smoke.AUDIT, smoke.FOLD2), (tp4_sampdraft.SHARD_ARGMAX, tp4_sampdraft.SHARD_ARGMAX_AUDIT, sarg.FOLD2_FLAG))

    def test_a_profile_without_the_lever_logs_nothing_and_sets_no_companion_flag(self):
        import tp4_shard_argmax_smoke as smoke
        self.assertEqual(smoke.problems({}, 'ordinary log\n'), [])
        self.assertEqual(smoke.problems(None, ''), [])
        self.assertEqual(len(smoke.problems({}, self.ENGAGED_1)), 1)
        self.assertEqual(len(smoke.problems({}, self.AUDIT_LINE)), 1)
        self.assertEqual(len(smoke.problems({sarg.FOLD2_FLAG: '1'}, '')), 1)
        self.assertEqual(len(smoke.problems({tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'}, '')), 1)

    def test_the_lever_needs_its_engaged_line_for_the_packed_block(self):
        import tp4_shard_argmax_smoke as smoke
        self.assertEqual(smoke.problems(self.ON, self.ENGAGED_1), [])
        self.assertEqual(len(smoke.problems(self.ON, '')), 1)
        small = self.ENGAGED_1.replace('rows=64', 'rows=16')
        self.assertTrue(any('rows=64' in problem for problem in smoke.problems(self.ON, small)))

    def test_the_fold_in_the_line_is_the_fold_the_profile_asked_for(self):
        import tp4_shard_argmax_smoke as smoke
        both = dict(self.ON, **{sarg.FOLD2_FLAG: '1'})
        self.assertEqual(smoke.problems(both, self.ENGAGED_2), [])
        self.assertTrue(any('wrong fold' in problem for problem in smoke.problems(both, self.ENGAGED_1)))
        self.assertTrue(any('wrong fold' in problem for problem in smoke.problems(self.ON, self.ENGAGED_2)))

    def test_words_and_the_task_count_are_checked(self):
        import tp4_shard_argmax_smoke as smoke
        self.assertTrue(any('words=1' in problem for problem in smoke.problems(self.ON, self.ENGAGED_1.replace(' words=1', ''))))
        self.assertTrue(any('two a worker' in problem for problem in smoke.problems(self.ON, self.ENGAGED_1.replace('tasks=220', 'tasks=110'))))

    def test_a_fall_back_or_a_mismatch_fails_whatever_the_flags(self):
        import tp4_shard_argmax_smoke as smoke
        fell = '%s rows=64 reason=logits are not bfloat16 TILE' % tp4_sampdraft.SARG_FALLBACK
        bad = '%s round=3 chip=1 rows=[2]' % tp4_sampdraft.SARG_MISMATCH
        self.assertEqual(len(smoke.problems({}, fell)), 1)
        self.assertEqual(len(smoke.problems(self.ON, '\n'.join([self.ENGAGED_1, fell]))), 1)
        self.assertEqual(len(smoke.problems(self.ON, '\n'.join([self.ENGAGED_1, bad]))), 1)

    def test_the_audit_flag_needs_a_passing_audit_line_for_four_chips(self):
        import tp4_shard_argmax_smoke as smoke
        audited = dict(self.ON, **{tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'})
        self.assertEqual(smoke.problems(audited, '\n'.join([self.ENGAGED_1, self.AUDIT_LINE])), [])
        self.assertEqual(len(smoke.problems(audited, self.ENGAGED_1)), 1)
        two_chips = self.AUDIT_LINE.replace('chips=4', 'chips=2')
        self.assertEqual(len(smoke.problems(audited, '\n'.join([self.ENGAGED_1, two_chips]))), 1)
        mismatch_only = '%s round=2 chip=0 words rows=[1]' % tp4_sampdraft.SARG_MISMATCH
        self.assertEqual(len(smoke.problems(audited, '\n'.join([self.ENGAGED_1, mismatch_only]))), 2)


class FusionJobTests(unittest.TestCase):
    """The WP1 card-job pack and the manifest the integrator (make_fusion_profiles.py) places from: every template parses (against the profile table with
    the manifest's twins applied by the generator's rule), the order names real files, and the shared files WP0 edits are described, not edited."""

    HERE = __import__('pathlib').Path(__file__).resolve().parent
    ROOT = HERE.parent.parent
    JOBS = ROOT / 'scripts' / 'ci' / 'references' / 'fusion-jobs' / 'WP1'
    MANIFEST = HERE / 'fusion-wp' / 'WP1.json'
    BASE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er'
    PARENT = BASE + '-traffic'

    def manifest(self):
        import json
        return json.loads(self.MANIFEST.read_text())

    def twins(self):
        """[(name, env added to the production profile)] the generator makes from the manifest: per lever its timed and audit twin, then the extra profiles."""
        manifest, out = self.manifest(), []
        for lever in manifest['levers']:
            env = {lever['flag']: lever['value']}
            out.append((self.BASE + '-fx-' + lever['id'], env))
            if lever.get('audit_flag'):
                out.append((self.BASE + '-fx-' + lever['id'] + '-audit', dict(env, **{lever['audit_flag']: '1'})))
        for extra in manifest['profiles']:
            out.append((self.BASE + '-fx-' + extra['name'], dict(extra['env'])))
        return out

    def table_with_twins(self):
        import copy
        import json
        table = json.loads((self.HERE / 'qwen_c2_profiles.json').read_text())
        for name, env in self.twins():
            profile = copy.deepcopy(table['profiles'][self.PARENT])
            profile['env'].update(env)
            profile.pop('owner_traffic_waiver', None)
            profile['gate_only'] = True
            if name in table['profiles']:
                # the integrator has placed it: it must be the production profile plus exactly this env
                self.assertEqual(table['profiles'][name]['env'], profile['env'], name)
                continue
            table['profiles'][name] = profile
        return table

    def twins_in_repo_table(self):
        import json
        table = json.loads((self.HERE / 'qwen_c2_profiles.json').read_text())
        return {name for name, _ in self.twins()} & set(table['profiles'])

    def parse(self, template, table_path):
        import io
        import contextlib
        import c2_serving_job
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(errors):
            status = c2_serving_job.main([str(self.JOBS / template), str(table_path)])
        return status, out.getvalue(), errors.getvalue()

    def test_the_manifest_is_in_the_integrators_schema(self):
        manifest = self.manifest()
        self.assertEqual(manifest['wp'], 'WP1')
        self.assertEqual(manifest['branch'], 'tp4/fx-wp1')
        allowed = {'wp', 'branch', 'head', 'description', 'reason', 'notes', 'card_jobs', 'docs', 'owner', 'ci_run', 'status',
                   'levers', 'profiles', 'smoke', 'image_files', 'tests', 'tp_addresses'}
        self.assertLessEqual(set(manifest), allowed)
        lever = manifest['levers'][0]
        self.assertEqual((lever['id'], lever['flag'], lever['value'], lever['audit_flag'], lever['marker']),
                         ('s1', tp4_sampdraft.SHARD_ARGMAX, '1', tp4_sampdraft.SHARD_ARGMAX_AUDIT, 'tp4 shard argmax'))
        self.assertEqual(len(manifest['levers']), 1)
        # the marker prefixes the generator derives are the module's own
        for suffix, constant in ((' engaged', tp4_sampdraft.SARG_ENGAGED), (' fell back', tp4_sampdraft.SARG_FALLBACK), (' audit', tp4_sampdraft.SARG_AUDIT)):
            self.assertEqual('[PINDIAG] ' + lever['marker'] + suffix, constant)
        names = {extra['name'] for extra in manifest['profiles']}
        self.assertEqual(names, {'s1f2', 's1f2-audit'})
        for extra in manifest['profiles']:
            self.assertTrue(set(extra['env']) <= {tp4_sampdraft.SHARD_ARGMAX, tp4_sampdraft.SHARD_ARGMAX_AUDIT, sarg.FOLD2_FLAG})
            self.assertEqual(extra['env'][tp4_sampdraft.SHARD_ARGMAX], '1')
            self.assertEqual(extra['env'][sarg.FOLD2_FLAG], '1')
            self.assertEqual(bool(extra['audit']), tp4_sampdraft.SHARD_ARGMAX_AUDIT in extra['env'])
        for entry in manifest['image_files']:
            self.assertTrue((self.ROOT / entry['path']).is_file(), entry)
        self.assertEqual([entry['path'] for entry in manifest['image_files']], ['scripts/ci/tp4_shard_argmax_fold2.cpp'])
        self.assertTrue((self.ROOT / 'optimisation' / 'ttnn-op' / 'shard_argmax').is_dir())
        self.assertTrue((self.HERE / 'test_tp4_shard_argmax.py').is_file())
        self.assertEqual(manifest['tp_addresses'], {})

    def test_the_card_m_job_and_the_control_arms_parse_against_the_profile_table_as_it_is(self):
        for template in ('C1-cardm-shard-argmax.env', 'T1-timed-A-control.env', 'T3-timed-A-control.env'):
            status, out, errors = self.parse(template, self.HERE / 'qwen_c2_profiles.json')
            self.assertEqual(status, 0, (template, errors))
        status, out, _ = self.parse('C1-cardm-shard-argmax.env', self.HERE / 'qwen_c2_profiles.json')
        self.assertIn('cardm_harness=optimisation/ttnn-op/shard_argmax/run_card_m.sh', out)
        self.assertIn('cardm_args=--timing always', out)

    def test_every_template_parses_once_the_twins_are_in_the_table(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = __import__('pathlib').Path(directory) / 'profiles.json'
            path.write_text(json.dumps(self.table_with_twins()))
            templates = sorted(item.name for item in self.JOBS.glob('*.env'))
            self.assertEqual(len(templates), 9)
            for template in templates:
                status, out, errors = self.parse(template, path)
                self.assertEqual(status, 0, (template, errors))
        # until the integrator places the twins the four-card lever templates refuse the repo's table (so the manifest is what makes them parse)
        status, _, errors = self.parse('T2-timed-B-s1f2.env', self.HERE / 'qwen_c2_profiles.json')
        if self.twins_in_repo_table():
            self.assertEqual(status, 0, errors)
        else:
            self.assertEqual(status, 1)
            self.assertIn('not a profile', errors)

    def test_the_order_names_every_template_once_with_the_one_image(self):
        lines = [line.split() for line in (self.JOBS / 'ORDER.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
        self.assertEqual(sorted(line[0] + '.env' for line in lines), sorted(item.name for item in self.JOBS.glob('*.env')))
        self.assertEqual({line[2] for line in lines}, {'tp4-fusion-1'})
        self.assertTrue(all(line[1] in ('stop', 'soft') and line[3].isdigit() for line in lines))
        self.assertEqual(lines[0][0], 'C1-cardm-shard-argmax')                      # the card-M job is first
        for template in self.JOBS.glob('*.env'):
            text = template.read_text()
            self.assertIn('C2_IMAGE_TAG=tp4-fusion-1', text)
            for forbidden in ('blackhole-', '192.168', 'thatch@', 'zot', 'ssh '):
                self.assertNotIn(forbidden, text)

    def test_the_pairs_use_the_production_profile_as_control_and_the_lever_is_the_control_plus_the_manifests_flags(self):
        twins = dict(self.twins())
        env = {}
        for template in self.JOBS.glob('*.env'):
            env[template.name] = dict(line.split('=', 1) for line in template.read_text().splitlines() if line.startswith('C2_'))
        self.assertEqual(env['T1-timed-A-control.env']['C2_PROFILE'], self.PARENT)
        self.assertEqual(env['T3-timed-A-control.env']['C2_PROFILE'], self.PARENT)
        self.assertEqual(env['T2-timed-B-s1f2.env']['C2_PROFILE'], env['T4-timed-B-s1f2.env']['C2_PROFILE'])
        self.assertEqual(twins[env['T2alt-timed-B-s1.env']['C2_PROFILE']], {tp4_sampdraft.SHARD_ARGMAX: '1'})
        self.assertEqual(twins[env['T2-timed-B-s1f2.env']['C2_PROFILE']], {tp4_sampdraft.SHARD_ARGMAX: '1', sarg.FOLD2_FLAG: '1'})
        self.assertEqual(twins[env['A1a-s1-audited-attach.env']['C2_PROFILE']],
                         {tp4_sampdraft.SHARD_ARGMAX: '1', tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'})
        self.assertEqual(twins[env['A1b-s1f2-audited-attach.env']['C2_PROFILE']],
                         {tp4_sampdraft.SHARD_ARGMAX: '1', tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1', sarg.FOLD2_FLAG: '1'})
        timed = {name for name in env if name.startswith('T')}
        self.assertEqual({env[name]['C2_SMOKE_TESTS'] for name in timed}, {'warmup,coding,concurrent8_steady,concurrent8_code_32k,concurrent8_code_128k'})

    def test_the_readback_layout_in_the_manifest_is_the_one_the_builder_makes(self):
        notes = self.manifest()['notes']
        for needle in ('(1,1,1,64)', 'ids uint32 (256 B)', 'values bf16 (128 B)', 'words uint32 (256 B)', '(column << 16) | bits', 'decode_words', 'words_for'):
            self.assertIn(needle, notes)
        operations = FakeOperations()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'}), patch.object(tp4_sampdraft, 'log_line'):
            sarg._RESERVED.clear()
            sarg.reserve(operations, 'mesh')
            ids, values = sarg.sample(operations, logits_for(operations, 64), 64)
            words = sarg.words_for(values)
        self.assertEqual([(t.shape, t.dtype, t.layout) for t in (ids, values, words)],
                         [((1, 1, 1, 64), 'u32', 'row_major'), ((1, 1, 1, 64), 'bf16', 'row_major'), ((1, 1, 1, 64), 'u32', 'row_major')])
        sarg._RESERVED.clear()
        sarg.PRODUCED.clear()
        sarg.WORDS.clear()


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
            words = sarg.words_for(values)
            self.assertEqual(ids.shape, (1, 1, 1, 64))
            self.assertEqual(values.shape, (1, 1, 1, 64))
            self.assertEqual(words.shape, (1, 1, 1, 64))
            self.assertEqual((ids.dtype, values.dtype, words.dtype), ('u32', 'bf16', 'u32'))
            self.assertEqual((ids.layout, values.layout, words.layout), ('row_major', 'row_major', 'row_major'))
            self.assertEqual(len(operations.generic), 2)
            (scan_tensors, scan), (fold_tensors, fold) = operations.generic
            partials = operations.empties[0]
            self.assertEqual(partials.shape, (1, 1, 220, 32))
            self.assertEqual(scan_tensors, [logits, partials])
            self.assertEqual(fold_tensors, [partials, ids, values, words])
            self.assertEqual(operations.freed, [])
            self.assertEqual(len(operations.empties), 4)     # the reserved partials, ids, values, words: nothing allocated for the scratch
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
                        for y, words_ in kernel['runtime_args'][x].items():
                            lengths.add(('scan', len(words_)))
                            self.assertEqual(words_[0], logits.shards[chip].buffer_address())
                            self.assertEqual(words_[1], partials.shards[chip].buffer_address())
                            self.assertEqual(words_[7], role)
                            self.assertEqual(words_[8], 1940)
                self.assertEqual([cb['total_size'] for cb in program['cbs']], [2048 * 19, 2048 * 19])
                self.assertEqual([cb['format_descriptors'][0]['buffer_index'] for cb in program['cbs']], [0, 1])
            for chip, program in enumerate(fold.values()):
                kernel = program['kernels'][0]
                self.assertTrue(kernel['kernel_source'].endswith('tp4_shard_argmax_fold.cpp'))
                self.assertEqual(kernel['defines'], [('FOLD_MAX_PAGES', '220')])
                words_ = kernel['runtime_args'][0][0]
                lengths.add(('fold', len(words_)))
                self.assertEqual(words_[:4], [partials.shards[chip].buffer_address(), ids.shards[chip].buffer_address(),
                                              values.shards[chip].buffer_address(), words.shards[chip].buffer_address()])
                self.assertEqual(words_[4:], [rows, 110 if rows > 32 else 220, 2 if rows > 32 else 1])
                self.assertEqual(program['cbs'][0]['total_size'], 2048 * 16)
        # the program cache does not hash runtime-arg lengths: every call must carry the same count
        self.assertEqual(lengths, {('scan', 9), ('fold', 7)})

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


def mesh_with_grid(x, y):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


class FoldFlagTests(unittest.TestCase):
    """QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2: strict, TP4 only, and only with S1 itself."""

    def test_unset_and_zero_are_off(self):
        self.assertFalse(sarg.fold2_enabled({'QWEN_FAST_TP': '4'}))
        self.assertFalse(sarg.fold2_enabled({'QWEN_FAST_TP': '4', sarg.FOLD2_FLAG: '0'}))
        self.assertFalse(sarg.fold2_enabled({sarg.FOLD2_FLAG: '0'}))

    def test_one_with_the_lever_is_on(self):
        self.assertTrue(sarg.fold2_enabled({'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1', sarg.FOLD2_FLAG: '1'}))

    def test_anything_else_raises(self):
        for value in ('2', 'true', '', 'on', ' 1'):
            with self.assertRaises(ValueError, msg=repr(value)):
                sarg.fold2_enabled({'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1', sarg.FOLD2_FLAG: value})

    def test_it_raises_at_the_pair_and_without_the_lever(self):
        with self.assertRaisesRegex(ValueError, 'TP4'):
            sarg.fold2_enabled({sarg.FOLD2_FLAG: '1', tp4_sampdraft.SHARD_ARGMAX: '1'})
        with self.assertRaisesRegex(ValueError, 'needs'):
            sarg.fold2_enabled({'QWEN_FAST_TP': '4', sarg.FOLD2_FLAG: '1'})
        with self.assertRaisesRegex(ValueError, 'needs'):
            sarg.fold2_enabled({'QWEN_FAST_TP': '4', sarg.FOLD2_FLAG: '1', tp4_sampdraft.SHARD_ARGMAX: '0'})

    def test_a_misconfigured_arm_fails_at_the_block_warm_not_at_a_replay(self):
        import packed_verifier
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1', sarg.FOLD2_FLAG: '2'}), \
                self.assertRaises(ValueError):
            sarg._RESERVED.clear()
            packed_verifier.reserve_sampdraft(FakeOperations(), 'mesh')

    def test_the_flag_changes_no_name_other_modules_read(self):
        # tp4_sampdraft (shared with D2a/D2c) does not know the flag: it is this module's alone, and WP0 wires its validation
        self.assertNotIn(sarg.FOLD2_FLAG, tp4_sampdraft.ALL_FLAGS)


class TreeBuilderTests(unittest.TestCase):
    """The launches the tree fold and the 13 x 10 grid describe."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1', sarg.FOLD2_FLAG: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        tp4_sampdraft._LOGGED.clear()
        for table in (sarg.REFERENCES, sarg.PRODUCED, sarg.WORDS, sarg._RESERVED):
            table.clear()
        quiet = patch.object(tp4_sampdraft, 'log_line')
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_reserve_adds_the_group_buffer_and_the_last_holder_frees_both(self):
        operations = FakeOperations()
        partials = sarg.reserve(operations, 'mesh')
        self.assertEqual([tensor.shape for tensor in operations.empties], [(1, 1, 220, 32), (1, 1, 8, 64)])
        self.assertTrue(all(tensor.dtype == 'u32' and tensor.layout == 'row_major' for tensor in operations.empties))
        self.assertIs(sarg.reserve(operations, 'mesh'), partials)
        self.assertEqual(len(operations.empties), 2)
        sarg.release_reserved(operations, 'mesh')
        self.assertEqual(operations.freed, [])
        sarg.release_reserved(operations, 'mesh')
        self.assertEqual(operations.freed, operations.empties)

    def test_three_launches_a_call_with_constant_argument_lengths(self):
        lengths = set()
        for rows in (1, 16, 32, 33, 48, 64):
            sarg._RESERVED.clear()
            operations = FakeOperations()
            partials = sarg.reserve(operations, 'mesh')
            group_buffer = operations.empties[1]
            logits = logits_for(operations, rows)
            ids, values = sarg.sample(operations, logits, rows)
            words = sarg.words_for(values)
            self.assertEqual([len(call[1]) for call in operations.generic], [4, 4, 4])
            (scan_tensors, _scan), (level1_tensors, level1), (level2_tensors, level2) = operations.generic
            self.assertEqual(scan_tensors, [logits, partials])
            self.assertEqual(level1_tensors, [partials, group_buffer])
            self.assertEqual(level2_tensors, [group_buffer, ids, values, words])
            self.assertEqual(operations.freed, [])
            self.assertEqual(len(operations.empties), 5)       # partials, groups, ids, values, words
            per_tile_row, tile_rows = (110, 2) if rows > 32 else (220, 1)
            for chip, program in enumerate(level1.values()):
                kernel = program['kernels'][0]
                self.assertTrue(kernel['kernel_source'].endswith('tp4_shard_argmax_fold2.cpp'))
                self.assertEqual(kernel['defines'], [('FOLD2_LEVEL', '1'), ('FOLD2_GROUP_PAGES', '28')])
                self.assertEqual(kernel['config']['processor'], 0)
                cores = [(x, y) for x in kernel['runtime_args'] for y in kernel['runtime_args'][x]]
                self.assertEqual(sorted(cores), [(group, 0) for group in range(8)])
                for group in range(8):
                    words_ = kernel['runtime_args'][group][0]
                    lengths.add(('level1', len(words_)))
                    self.assertEqual(words_, [partials.shards[chip].buffer_address(), group_buffer.shards[chip].buffer_address(),
                                              rows, per_tile_row, tile_rows, 8, group])
                self.assertEqual(program['cbs'][0]['total_size'], 2048 * 2)           # 28 x 128 + 256 = 3,840 bytes
            for chip, program in enumerate(level2.values()):
                kernel = program['kernels'][0]
                self.assertEqual(kernel['defines'], [('FOLD2_LEVEL', '2'), ('FOLD2_GROUP_PAGES', '28')])
                words_ = kernel['runtime_args'][0][0]
                lengths.add(('level2', len(words_)))
                self.assertEqual(words_, [group_buffer.shards[chip].buffer_address(), ids.shards[chip].buffer_address(),
                                          values.shards[chip].buffer_address(), words.shards[chip].buffer_address(), rows, 8])
                self.assertEqual(program['cbs'][0]['total_size'], 2048 * 2)           # 8 x 256 + 640 = 2,688 bytes
        self.assertEqual(lengths, {('level1', 7), ('level2', 6)})

    def test_the_marker_names_the_fold(self):
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            operations = FakeOperations()
            sarg.reserve(operations, 'mesh')
            sarg.sample(operations, logits_for(operations, 64), 64)
        engaged = [line for line in lines if line.startswith(tp4_sampdraft.SARG_ENGAGED)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('rows=64 workers=110 tasks=220 fold=2 audit=0 words=1', engaged[0])

    def test_a_failing_second_level_frees_every_output_and_keeps_the_reservations(self):
        operations = FakeOperations()
        sarg.reserve(operations, 'mesh')
        kept = list(operations.empties)
        calls = []
        operations.generic_op = lambda tensors, program: (calls.append(1), (_ for _ in ()).throw(RuntimeError('submit')) if len(calls) == 3 else None)
        with self.assertRaises(RuntimeError):
            sarg.sample(operations, logits_for(operations, 64), 64)
        self.assertEqual({value.name for value in operations.freed}, {value.name for value in operations.empties} - {value.name for value in kept})

    def test_the_tree_launch_without_its_group_buffer_is_refused(self):
        operations = FakeOperations()
        with patch.dict(os.environ, {sarg.FOLD2_FLAG: '0'}):
            partials = sarg.reserve(operations, 'mesh')              # reserved by a process that had the flag off
        ids = operations.empty((1, 1, 1, 64), 'u32', 'row_major')
        values = operations.empty((1, 1, 1, 64), 'bf16', 'row_major')
        words = operations.empty((1, 1, 1, 64), 'u32', 'row_major')
        with self.assertRaisesRegex(ValueError, 'group buffer'):
            sarg.launch(operations, logits_for(operations, 64), 64, partials, None, ids, values, words, sarg.GRID, True)


class WideGridBuilderTests(unittest.TestCase):
    """13 x 10 (130 workers): nothing in the builder is a literal 110."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        tp4_sampdraft._LOGGED.clear()
        for table in (sarg.REFERENCES, sarg.PRODUCED, sarg.WORDS, sarg._RESERVED):
            table.clear()
        quiet = patch.object(tp4_sampdraft, 'log_line')
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_the_grid_is_read_from_the_mesh(self):
        self.assertEqual(sarg.grid_of(mesh_with_grid(13, 10)), (13, 10))
        self.assertEqual(sarg.grid_of(mesh_with_grid(11, 10)), (11, 10))
        self.assertEqual(sarg.grid_of('a fake mesh'), sarg.GRID)
        self.assertEqual(sarg.grid_of(mesh_with_grid(0, 10)), sarg.GRID)

    def test_plan_at_130_workers_covers_every_tile_column_once_per_tile_row(self):
        for rows in (1, 16, 32, 33, 64):
            tasks, per_tile_row, tile_rows = sarg.plan(rows, 1940, 130)
            self.assertEqual(len(tasks), 260)
            self.assertEqual(per_tile_row * tile_rows, 260)
            self.assertEqual(sorted(task[0] for task in tasks), list(range(260)))
            for tile_row in range(tile_rows):
                mine = sorted((task for task in tasks if task[3] == tile_row), key=lambda task: task[0])
                self.assertEqual([column for task in mine for column in range(task[4], task[5])], list(range(1940)))
            self.assertTrue(all(task[5] - task[4] <= sarg.MAX_TILES for task in tasks))
        runs = sarg.column_runs(1940, 130)
        self.assertEqual([last - first for first, last in runs].count(15), 120)
        self.assertEqual([last - first for first, last in runs].count(14), 10)

    def test_reserve_sizes_the_partials_for_the_grid_and_the_scan_uses_every_worker(self):
        operations = FakeOperations()
        mesh = mesh_with_grid(13, 10)
        partials = sarg.reserve(operations, mesh)
        self.assertEqual(partials.shape, (1, 1, 260, 32))
        for rows in (16, 64):
            operations.generic.clear()
            logits = logits_for(operations, rows)
            sarg.sample(operations, logits, rows)
            scan, fold = operations.generic[0][1], operations.generic[1][1]
            for program in scan.values():
                for role, kernel in enumerate(program['kernels']):
                    cores = {(x, y) for x in kernel['runtime_args'] for y in kernel['runtime_args'][x]}
                    self.assertEqual(cores, {(x, y) for x in range(13) for y in range(10)})
                    self.assertEqual(kernel['core_ranges'], [((0, 0), (12, 9))])
                self.assertEqual(sorted(task for kernel in program['kernels'] for column in kernel['runtime_args'].values()
                                        for words in column.values() for task in [words[6]]), list(range(260)))
            for program in fold.values():
                kernel = program['kernels'][0]
                self.assertEqual(kernel['defines'], [('FOLD_MAX_PAGES', '260')])
                self.assertEqual(kernel['runtime_args'][0][0][4:], [rows, 130 if rows > 32 else 260, 2 if rows > 32 else 1])
                self.assertEqual(program['cbs'][0]['total_size'], 2048 * 17)            # 260 x 128 + 640 bytes
        self.assertEqual(sarg.fold_scratch_pages(220), 16)
        self.assertEqual(sarg.fold_scratch_pages(260), 17)

    def test_the_tree_at_130_workers(self):
        with patch.dict(os.environ, {sarg.FOLD2_FLAG: '1'}):
            operations = FakeOperations()
            partials = sarg.reserve(operations, mesh_with_grid(13, 10))
            self.assertEqual(operations.empties[1].shape, (1, 1, 8, 64))
            sarg.sample(operations, logits_for(operations, 64), 64)
            level1 = operations.generic[1][1]
            for program in level1.values():
                kernel = program['kernels'][0]
                self.assertEqual(kernel['defines'], [('FOLD2_LEVEL', '1'), ('FOLD2_GROUP_PAGES', '34')])
                self.assertEqual(kernel['runtime_args'][0][0][3:], [130, 2, 8, 0])
                self.assertEqual(program['cbs'][0]['total_size'], 2048 * 3)             # 34 x 128 + 256 = 4,608 bytes

    def test_the_engaged_marker_reports_the_grid_it_used(self):
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            operations = FakeOperations()
            sarg.reserve(operations, mesh_with_grid(13, 10))
            sarg.sample(operations, logits_for(operations, 64), 64)
        self.assertIn('rows=64 workers=130 tasks=260 fold=1 audit=0 words=1', [line for line in lines if 'engaged' in line][0])

    def test_a_grid_too_small_for_the_scan_scratch_is_a_logged_fall_back(self):
        lines = []
        with patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            operations = FakeOperations()
            sarg.reserve(operations, mesh_with_grid(8, 10))             # 80 workers: runs of 24-25 tile columns
            self.assertIsNone(sarg.sample(operations, logits_for(operations, 64), 64))
        self.assertEqual(operations.generic, [])
        self.assertTrue(lines[0].startswith(tp4_sampdraft.SARG_FALLBACK))
        self.assertIn('scan scratch', lines[0])


class WordsLifecycleTests(unittest.TestCase):
    """The words tensor is freed with the trace output it belongs to, and the audit holds it to the kernel's own ids and values."""

    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', tp4_sampdraft.SHARD_ARGMAX: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        for table in (sarg.REFERENCES, sarg.PRODUCED, sarg.WORDS, sarg._RESERVED):
            table.clear()
        sarg._STATE['rounds'] = 0
        quiet = patch.object(tp4_sampdraft, 'log_line')
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_release_audit_frees_the_words_tensor_and_forgets_it(self):
        operations = FakeOperations()
        sarg.reserve(operations, 'mesh')
        _, values = sarg.sample(operations, logits_for(operations, 16), 16)
        words = sarg.words_for(values)
        self.assertIsNotNone(words)
        sarg.release_audit(operations, values)
        self.assertEqual(operations.freed, [words])
        self.assertIsNone(sarg.words_for(values))
        self.assertEqual(sarg.WORDS, {})
        sarg.release_audit(operations, values)                      # twice: nothing more to free
        self.assertEqual(operations.freed, [words])
        self.assertIsNone(sarg.words_for(None))

    def test_the_packed_blocks_release_hook_frees_it(self):
        import packed_verifier
        operations = FakeOperations()
        sarg.reserve(operations, 'mesh')
        _, values = sarg.sample(operations, logits_for(operations, 64), 64)
        words = sarg.words_for(values)
        packed_verifier.release_sampdraft_audit(operations, values)
        self.assertIn(words, operations.freed)

    def audit_operations(self, ids, values, words):
        """A recording stand-in: three held tensors per chip (today's ids and values, the kernel's words)."""
        class Held:
            def __init__(self, data):
                self.data = data

        def parts(tensor):
            return tensor
        return SimpleNamespace(get_device_tensors=parts, to_torch=lambda part: part.data), \
            [Held(torch.tensor(ids)) for _ in range(4)], [Held(torch.tensor(values, dtype=torch.float32)) for _ in range(4)], \
            [Held(word) for word in words]

    def test_the_audit_checks_the_words_against_the_kernels_ids_and_value_bits(self):
        kernel_ids = torch.tensor([5, 40000, 7], dtype=torch.int32)
        kernel_values = torch.tensor([1.0, -2.0, 0.0], dtype=torch.bfloat16)
        packed = sarg.pack_words(kernel_ids, kernel_values)
        wire = packed.where(packed < (1 << 31), packed - (1 << 32)).to(torch.int32)         # what to_torch of uint32 returns
        operations, held_ids, held_values, held_words = self.audit_operations([5, 40000, 7], [1.0, -2.0, 0.0], [wire] * 4)
        sarg.REFERENCES[id('v')] = (held_ids, held_values)
        sarg.WORDS[id('v')] = held_words
        lines = []
        with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'}), patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append):
            sarg.audit_round(operations, 'v', 3, [kernel_ids] * 4, [kernel_values] * 4)
        self.assertEqual(lines, ['%s 1 exact=True rows=3 chips=4' % tp4_sampdraft.SARG_AUDIT])
        # one wrong word on chip 2
        bad = wire.clone()
        bad[1] ^= 1
        sarg.WORDS[id('v')] = [held_words[0], held_words[1], SimpleNamespace(data=bad), held_words[3]]
        lines.clear()
        with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'}), patch.object(tp4_sampdraft, 'log_line', side_effect=lines.append), \
                self.assertRaises(AssertionError):
            sarg.audit_round(operations, 'v', 3, [kernel_ids] * 4, [kernel_values] * 4)
        self.assertIn('chip=2 words rows=[1]', lines[0])
        self.assertTrue(lines[0].startswith(tp4_sampdraft.SARG_MISMATCH))

    def test_an_audited_output_without_words_still_audits_ids_and_values(self):
        operations, held_ids, held_values, _ = self.audit_operations([5, 6, 7], [1.0, 2.0, 3.0], [None] * 4)
        sarg.REFERENCES[id('v')] = (held_ids, held_values)
        with patch.dict(os.environ, {tp4_sampdraft.SHARD_ARGMAX_AUDIT: '1'}):
            sarg.audit_round(operations, 'v', 3, [torch.tensor([5, 6, 7])] * 4, [torch.tensor([1.0, 2.0, 3.0])] * 4)


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
