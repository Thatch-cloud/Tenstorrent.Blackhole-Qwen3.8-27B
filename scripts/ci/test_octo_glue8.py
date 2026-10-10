"""QWEN_FAST_OCTO_GLUE8: the quarter-tile row mover (gdn_rows_dma8_tp) against the served split, merge and window stacking at the octo block's EIGHT-row users, the flag, and the
smoke rule.

The kernel (gdn_rows_dma8_tp.cpp) is emulated from its own loops over the very runtime arguments the host planner writes, on RAW 2048-byte tiles (four 16 x 16 faces, face f at
int16 offset 256 f, row r of a face at 16 r), so the quarter chunk offsets the kernel computes are checked against the physical layout and not only against a logical matrix.
Values are bf16 bit patterns (-0, denormals, NaN payloads, infinities are real values). What the served ops define is stated on LOGICAL matrices: a raw tile Slice for the users that
start on a tile boundary (0 and 4), an untilize / slice / tilize round trip that maps a zero-exponent bf16 to +0 for the users that start inside a tile (1, 2, 3, 5, 6, 7), an untilize
/ concat / tilize that does the same to every user. The value rule is the one measured on card M for sixteen-row users (gdn_prefill_conv_exact.py); that the ops apply it at EIGHT-row
slice starts too is the in-trace audit's (QWEN_FAST_TP4_VGLUE_AUDIT) and the edge probe's question, not a CPU fact.

Run: `python -m unittest test_octo_glue8` from scripts/ci.
"""

import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import torch

import gdn_rows_dma8_tp as rows8
import gdn_rows_dma_tp as rows16
import octo_glue8
import tp4_vglue
import tp_shapes
from test_tp4_vglue_gdn import canon, specials, to_tiles

HERE = Path(__file__).resolve().parent


def env(**values):
    return patch.dict(os.environ, values)


def four():
    return patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


def geometry():
    with four():
        return tp_shapes.active()


# --- raw tiles ---------------------------------------------------------------------------------------------------------------

def to_raw(tile):
    """A logical 32 x 32 int16 tile -> its 1024 int16 in the device's face order (faces 0, 1, 2, 3; each 16 x 16 row-major)."""
    raw = torch.zeros(1024, dtype=torch.int16)
    for face in range(4):
        row, column = (face // 2) * 16, (face % 2) * 16
        raw[face * 256:(face + 1) * 256] = tile[row:row + 16, column:column + 16].reshape(256)
    return raw


def from_raw(raw):
    tile = torch.zeros(32, 32, dtype=torch.int16)
    for face in range(4):
        row, column = (face // 2) * 16, (face % 2) * 16
        tile[row:row + 16, column:column + 16] = raw[face * 256:(face + 1) * 256].reshape(16, 16)
    return tile


def raw_tiles(matrix):
    return {page: to_raw(tile) for page, tile in to_tiles(matrix).items()}


def logical(tiles, rows, columns):
    """{page: raw tile} -> the (rows, columns) logical int16 matrix (columns a multiple of 32), page = tile row * tile columns + tile column."""
    tile_columns = columns // 32
    tile_rows = -(-rows // 32)
    matrix = torch.zeros(tile_rows * 32, columns, dtype=torch.int16)
    for page, raw in tiles.items():
        row, column = divmod(page, tile_columns)
        matrix[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32] = from_raw(raw)
    return matrix[:rows]


def canon_words(values, canon_denorm):
    if canon_denorm:
        return canon(values)
    return torch.where(values == -32768, torch.zeros_like(values), values)


def quarter_slice(quarter):
    """The two int16 ranges (left face, right face) of quarter `quarter`, from the kernel's quarter_offset arithmetic (bytes / 2)."""
    left = ((quarter >> 1) * 1024 + (quarter & 1) * 256) // 2
    return (left, left + 128), (left + 256, left + 256 + 128)


def run_kernel(argument_lists, source_addresses, destination_addresses, sources, destinations, canon_denorm=True, generator=None):
    """gdn_rows_dma8_tp.cpp's loops over each core's [tasks, NSRC addresses, NDST addresses, five words per task] list. `sources` maps a buffer address to {page: raw tile},
    `destinations` the same (filled). The scratch starts as junk: every byte of a destination tile must be written or zeroed by the task itself."""
    generator = generator or torch.Generator().manual_seed(3)
    nsrc, ndst = len(source_addresses), len(destination_addresses)
    for words in argument_lists:
        assert words[1:1 + nsrc] == list(source_addresses) and words[1 + nsrc:1 + nsrc + ndst] == list(destination_addresses)
        first = 1 + nsrc + ndst
        for task in range(words[0]):
            base = first + 5 * task
            head = words[base]
            destination, page = (head >> 16) & 0xFF, head & 0xFFFF
            scratch = torch.randint(-32768, 32768, (1024,), generator=generator, dtype=torch.int16)
            for quarter in range(4):
                word = words[base + 1 + quarter]
                mode = (word >> 16) & 0xFF
                (to_left, to_right) = quarter_slice(quarter)
                if mode == 0:
                    scratch[to_left[0]:to_left[1]] = 0
                    scratch[to_right[0]:to_right[1]] = 0
                    continue
                source = sources[words[1 + (word >> 24)]][word & 0xFFFF]
                (from_left, from_right) = quarter_slice((mode - 1) & 3)
                for target, origin in ((to_left, from_left), (to_right, from_right)):
                    chunk = source[origin[0]:origin[1]].clone()
                    if mode >= 5:
                        chunk = canon_words(chunk, canon_denorm)
                    scratch[target[0]:target[1]] = chunk
            destinations[words[1 + nsrc + destination]][page] = scratch


def execute(tasks, sources, destinations, cores=110, canon_denorm=True):
    """Plan `tasks` over `cores`, resolve tensor indices to fake addresses (sources 100+, destinations 900+) and run. `sources` is a list of {page: raw tile}."""
    per_core = rows8.distribute(tasks, cores, len(sources), destinations)
    source_addresses = [100 + index for index in range(len(sources))]
    destination_addresses = [900 + index for index in range(destinations)]
    arguments = rows8.runtime_arguments(per_core, source_addresses, destination_addresses)
    for words in arguments:
        assert len(words) <= rows8.MAX_ARGUMENT_WORDS
    built = {address: {} for address in destination_addresses}
    run_kernel(arguments, source_addresses, destination_addresses, dict(zip(source_addresses, sources)), built, canon_denorm)
    return [built[address] for address in destination_addresses]


# --- the planners ----------------------------------------------------------------------------------------------------------------

class PositionTests(unittest.TestCase):
    def test_users_are_quarters_and_only_the_two_tile_aligned_ones_are_raw(self):
        self.assertEqual([rows8.position(user) for user in range(8)],
                         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)])
        self.assertEqual([rows8.served_canon(user) for user in range(8)], [False, True, True, True, False, True, True, True])
        self.assertEqual([user for user in range(8) if rows8.served_canon(user)], list(rows8.BLOCK_CANON_USERS))

    def test_the_quarter_offsets_are_the_physical_layout_of_rows_8q_to_8q_plus_7(self):
        for quarter in range(4):
            (left, left_end), (right, right_end) = quarter_slice(quarter)
            logical_tile = torch.arange(1024, dtype=torch.int16).reshape(32, 32)
            raw = to_raw(logical_tile)
            expected_left = logical_tile[8 * quarter:8 * quarter + 8, :16].reshape(-1)
            expected_right = logical_tile[8 * quarter:8 * quarter + 8, 16:].reshape(-1)
            self.assertTrue(torch.equal(raw[left:left_end], expected_left), quarter)
            self.assertTrue(torch.equal(raw[right:right_end], expected_right), quarter)
            self.assertEqual(rows8.quarter_offsets(quarter), (left * 2, right * 2))

    def test_the_sixteen_row_mover_is_untouched_and_unimported_by_this_one(self):
        self.assertEqual((rows16.USER_ROWS, rows16.TASK_WORDS), (16, 8))
        self.assertIs(rows8.Unsupported, rows16.Unsupported)
        source = (HERE / 'gdn_rows_dma8_tp.py').read_text()
        self.assertNotIn('import gdn_rows_dma_tp', source.replace('from gdn_rows_dma_tp import', ''))


class MoverTests(unittest.TestCase):
    def setUp(self):
        self.width = geometry().gdn_qkvzab            # 4120 -> 129 tile columns
        self.generator = torch.Generator().manual_seed(7)
        self.columns = rows8.tile_columns(self.width)
        self.block = specials(self.generator, (64, self.columns * 32))
        self.block[:, self.width:] = 0                 # the padding columns of a TILE tensor read back zero

    def test_the_split_is_the_served_slices_for_all_eight_positions(self):
        pieces = execute(rows8.split_pieces(8, self.width), [raw_tiles(self.block)], 8)
        for user in range(8):
            rows = self.block[8 * user:8 * user + 8]
            served = canon(rows) if user in (1, 2, 3, 5, 6, 7) else rows          # untilize / slice / tilize vs a tile Slice
            expected = torch.zeros(32, self.columns * 32, dtype=torch.int16)
            expected[:8] = served
            self.assertEqual(sorted(pieces[user]), list(range(self.columns)), 'every page written once')
            self.assertTrue(torch.equal(logical(pieces[user], 32, self.columns * 32), expected), user)

    def test_negative_zero_and_denormals_are_canonical_exactly_where_the_slice_starts_inside_a_tile(self):
        block = torch.full((64, self.columns * 32), -32768, dtype=torch.int16)        # -0 everywhere
        denormal = torch.full((64, self.columns * 32), 0x0001, dtype=torch.int16)      # the smallest positive denormal
        for values, name in ((block, '-0'), (denormal, 'denormal')):
            pieces = execute(rows8.split_pieces(8, self.width), [raw_tiles(values)], 8)
            for user in range(8):
                first = logical(pieces[user], 32, self.columns * 32)[:8, :16]
                want = (0 if user in rows8.BLOCK_CANON_USERS else int(values[0, 0]))
                self.assertTrue(bool((first == want).all()), (name, user))
        # a NaN payload, an infinity and a normal are never touched, canonical or not
        keep = torch.tensor([0x7FC1, 0x7F80, -128, 0x3F80, -16512, 0x7F81], dtype=torch.int16)
        block = keep.repeat(64, self.columns * 32 // 6 + 1)[:, :self.columns * 32].contiguous()
        pieces = execute(rows8.split_pieces(8, self.width), [raw_tiles(block)], 8)
        for user in range(8):
            self.assertTrue(torch.equal(logical(pieces[user], 32, self.columns * 32)[:8], block[8 * user:8 * user + 8]), user)

    def test_the_canonical_block_takes_all_four_quarters_from_one_tile_and_canonicalises_the_unaligned_users(self):
        (built,) = execute(rows8.canon_block(8, self.width), [raw_tiles(self.block)], 1)
        expected = self.block.clone()
        for user in rows8.BLOCK_CANON_USERS:
            expected[8 * user:8 * user + 8] = canon(self.block[8 * user:8 * user + 8])
        self.assertEqual(sorted(built), list(range(2 * self.columns)))
        self.assertTrue(torch.equal(logical(built, 64, self.columns * 32), expected))

    def test_the_merge_is_the_served_untilize_concat_tilize(self):
        width = geometry().gdn_value                   # 1536: the gated output
        columns = rows8.tile_columns(width)
        outputs = [specials(self.generator, (8, columns * 32)) for _ in range(8)]
        tiles = []
        for output in outputs:
            tile_set = to_tiles(output)
            for page in tile_set:
                # the recurrence leaves the tile's padding rows anything: only rows 0-7 are the output
                tile_set[page][8:] = torch.randint(-32768, 32768, (24, 32), generator=self.generator, dtype=torch.int16)
            tiles.append({page: to_raw(tile) for page, tile in tile_set.items()})
        (built,) = execute(rows8.merge_outputs(8, width), tiles, 1)
        expected = canon(torch.cat(outputs, dim=0))
        self.assertEqual(sorted(built), list(range(2 * columns)))
        self.assertTrue(torch.equal(logical(built, 64, columns * 32), expected))

    def test_a_partial_block_pads_the_last_quarters_with_zeros(self):
        width = 96
        outputs = [specials(self.generator, (8, 96)) for _ in range(5)]
        (built,) = execute(rows8.merge_outputs(5, width), [raw_tiles(output) for output in outputs], 1)
        expected = torch.zeros(64, 96, dtype=torch.int16)
        expected[:40] = canon(torch.cat(outputs, dim=0))
        self.assertTrue(torch.equal(logical(built, 64, 96), expected))

    def test_windows_stack_raw_and_unstack_raw(self):
        width = geometry().gdn_qkv
        columns = rows8.tile_columns(width)
        windows = [specials(self.generator, (8, columns * 32)) for _ in range(8)]
        (block,) = execute(rows8.stack_users(8, width), [raw_tiles(window) for window in windows], 1)
        self.assertTrue(torch.equal(logical(block, 64, columns * 32), torch.cat(windows, dim=0)))
        back = execute(rows8.unstack_users(8, columns, 0, columns), [block], 8)
        for user in range(8):
            matrix = logical(back[user], 32, columns * 32)
            self.assertTrue(torch.equal(matrix[:8], windows[user]), user)
            self.assertFalse(bool(matrix[8:].any()), 'rows 8-31 of a piece are zero')

    def test_unstack_reads_a_column_range_of_a_wider_block(self):
        columns = self.columns
        pieces = execute(rows8.unstack_users(8, columns, 80, 48), [raw_tiles(self.block)], 8)         # z: columns 80..127 of the projection
        for user in range(8):
            self.assertEqual(sorted(pieces[user]), list(range(48)))
            matrix = logical(pieces[user], 32, 48 * 32)
            self.assertTrue(torch.equal(matrix[:8], self.block[8 * user:8 * user + 8, 80 * 32:128 * 32]), user)

    def test_canonical_flag_off_keeps_denormals(self):
        block = torch.zeros(64, 32, dtype=torch.int16)
        block[:, :4] = torch.tensor([-32768, 1, -32767, 0x3F80], dtype=torch.int16)
        for canon_denorm, want in ((False, [0, 1, -32767, 0x3F80]), (True, [0, 0, 0, 0x3F80])):
            (piece,) = execute([(0, 0, ((0, 0, rows8.mode(1, True)), rows8.Z, rows8.Z, rows8.Z))], [raw_tiles(block)], 1, canon_denorm=canon_denorm)
            self.assertEqual(logical(piece, 32, 32)[0, :4].tolist(), want)

    def test_every_planned_destination_page_is_written_once_and_counts_are_as_derived(self):
        columns = self.columns
        for tasks, count in ((rows8.split_pieces(8, self.width), 8 * columns),
                             (rows8.canon_block(8, self.width), 2 * columns),
                             (rows8.merge_outputs(8, 1536), 2 * 48),
                             (rows8.stack_users(8, 2560), 2 * 80)):
            self.assertEqual(len(tasks), count)
            self.assertEqual(len({(task[0], task[1]) for task in tasks}), count)

    def test_every_launch_the_octo_layer_makes_fits_the_cores_argument_words(self):
        from gdn_block_conv8_tp import plan_unstack, plan_windows

        found = geometry()
        patcher = four()
        patcher.start()
        self.addCleanup(patcher.stop)
        launches = [(rows8.split_pieces(8, found.gdn_qkvzab), 1, 8), (rows8.merge_outputs(8, found.gdn_value), 8, 1),
                    (rows8.canon_block(8, found.gdn_qkvzab), 1, 1), (plan_windows(8), 32, 4), (plan_unstack(8, found.gdn_qkvzab), 8, 64)]
        for tasks, sources, destinations in launches:
            per_core = rows8.distribute(tasks, 110, sources, destinations)
            arguments = rows8.runtime_arguments(per_core, list(range(sources)), list(range(destinations)))
            self.assertTrue(all(len(words) == len(arguments[0]) <= rows8.MAX_ARGUMENT_WORDS for words in arguments))
        # the biggest, the advanced-window unstack, uses 3,600 of 3,960 task slots; fewer cores than the p150a grid would not carry it
        self.assertEqual(len(plan_unstack(8, found.gdn_qkvzab)), 8 * 80 + 8 + 8 + 8 * 48 + 32 * 80)
        self.assertEqual(rows8.capacity_for(8, 64), 36)
        with self.assertRaises(rows8.Unsupported):
            rows8.distribute(plan_unstack(8, found.gdn_qkvzab), 64, 8, 64)

    def test_a_task_list_the_kernel_cannot_carry_is_unsupported(self):
        tasks = [(0, page, ((0, page, 1), rows8.Z, rows8.Z, rows8.Z)) for page in range(60 * 110)]
        with self.assertRaises(rows8.Unsupported):
            rows8.distribute(tasks, 110)
        with self.assertRaises(rows8.Unsupported):
            rows8.distribute([], 110)

    def test_the_packed_words_round_trip_and_refuse_out_of_range(self):
        word = rows8.pack_source((7, 4000, rows8.mode(3, True)))
        self.assertEqual((word >> 24, (word >> 16) & 0xFF, word & 0xFFFF), (7, 8, 4000))
        self.assertEqual(rows8.pack_source(rows8.Z), 0)
        self.assertEqual(rows8.pack_head(63, 259) >> 16, 63)
        for bad in ((256, 0, 1), (0, 65536, 1), (0, 0, 9)):
            with self.assertRaises(ValueError):
                rows8.pack_source(bad)

    def test_the_kernel_text_matches_the_task_encoding(self):
        text = (HERE / 'gdn_rows_dma8_tp.cpp').read_text()
        self.assertIn('constexpr uint32_t TASK_WORDS = %d;' % rows8.TASK_WORDS, text)
        self.assertIn('constexpr uint32_t LANES = %d;' % rows8.LANES, text)
        self.assertEqual(rows8.SCRATCH_BYTES, rows8.LANES * 2048)
        self.assertIn('#define CANON_DENORM 1', text)
        self.assertIn('(value & 0x7F80u) == 0 ? 0u : value', text)
        self.assertIn('value == 0x8000u ? 0u : value', text)
        # the packed words and the quarter arithmetic, as the emulator and the planner encode them
        self.assertIn('return (quarter >> 1) * (2 * FACE_BYTES) + (quarter & 1) * CHUNK_BYTES;', text)
        self.assertIn('constexpr uint32_t CHUNK_BYTES = 256;', text)
        self.assertIn('(word >> 16) & 0xFFu', text)
        self.assertIn('(word >> 24)', text)
        self.assertIn('word & 0xFFFFu', text)
        self.assertIn('(head >> 16) & 0xFFu', text)
        self.assertIn('head & 0xFFFFu', text)
        self.assertIn('(mode - 1) & 3', text)
        self.assertIn('mode >= 5', text)
        self.assertIn('constexpr uint32_t DESTINATION_TABLE = SOURCE_TABLE + NSRC;', text)
        self.assertIn('constexpr uint32_t TASKS = DESTINATION_TABLE + NDST;', text)
        self.assertEqual(rows8.QUARTER_STRIDE, 256)
        self.assertEqual(rows8.CHUNK_BYTES, 256)
        # every destination tile is fully defined by its task: each quarter is read or zeroed
        self.assertIn('if (mode == 0) { continue; }', text)
        self.assertIn('words[word] = 0;', text)

    def test_the_kernel_never_names_the_half_tile_constants(self):
        text = (HERE / 'gdn_rows_dma8_tp.cpp').read_text()
        self.assertNotIn('HALF_BYTES', text)


# --- the flag and the smoke rule ----------------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_strict_zero_one_and_default_off(self):
        with four():
            with env():
                os.environ.pop(octo_glue8.FLAG, None)
                self.assertFalse(octo_glue8.enabled())
            for value, want in (('0', False), ('1', True)):
                with env(**{octo_glue8.FLAG: value}):
                    self.assertEqual(octo_glue8.enabled(), want)
            for value in ('', '2', 'true', 'on'):
                with env(**{octo_glue8.FLAG: value}), self.assertRaises(ValueError):
                    octo_glue8.enabled()

    def test_the_pair_refuses_it(self):
        with env(**{octo_glue8.FLAG: '1', 'QWEN_FAST_TP': '2'}), self.assertRaises(ValueError):
            octo_glue8.enabled()

    def test_the_sites_follow_the_levers_that_are_on(self):
        levers = {'QWEN_FAST_TP': '4', 'QWEN_FAST_OCTO': 'alternate'}
        t2_off = {octo_glue8.T2_FLAG: '0'}
        cases = [
            ({}, []),
            ({octo_glue8.FLAG: '1'}, ['windows']),                                  # the image's ENV has T2 on; a profile need not name it
            ({octo_glue8.FLAG: '1', **t2_off}, []),
            ({octo_glue8.FLAG: '1', octo_glue8.V2_FLAG: '1', **t2_off}, ['split', 'merge']),
            ({octo_glue8.FLAG: '1', octo_glue8.V2_FLAG: '1', octo_glue8.V1_FLAG: '1', **t2_off}, ['split', 'merge']),
            ({octo_glue8.FLAG: '1', octo_glue8.T2_FLAG: '1'}, ['windows']),
            ({octo_glue8.FLAG: '1', octo_glue8.V2_FLAG: '1', octo_glue8.V1_FLAG: '1'}, ['split', 'merge', 'block_conv', 'windows']),
            ({octo_glue8.FLAG: '1', octo_glue8.V2_FLAG: '1', octo_glue8.V1_FLAG: '1', octo_glue8.T2_FLAG: '1'}, ['split', 'merge', 'block_conv', 'windows']),
            ({octo_glue8.FLAG: '1', octo_glue8.V2_FLAG: '1', octo_glue8.V1_FLAG: '1', octo_glue8.T2_FLAG: '1', octo_glue8.T2_SKIP_FLAG: 'windows'}, ['split', 'merge']),
            ({octo_glue8.FLAG: '1', octo_glue8.V1_FLAG: '1'}, ['windows']),
        ]
        for extra, want in cases:
            with self.subTest(extra=extra):
                environ = dict(levers, **extra)
                self.assertEqual(octo_glue8.sites_wanted(environ), want)

    def test_refusal_names_a_missing_octo_block_a_missing_gate_marker_or_nothing_to_make_native(self):
        base = {'QWEN_FAST_TP': '4', octo_glue8.FLAG: '1', octo_glue8.V2_FLAG: '1'}
        self.assertIn('QWEN_FAST_OCTO', octo_glue8.refusal(base))
        octo = dict(base, QWEN_FAST_OCTO='live')
        self.assertIn('gate only', octo_glue8.refusal(octo))
        gated = dict(octo, QWEN_C2_GATE_PROFILE='1')
        self.assertIsNone(octo_glue8.refusal(gated))
        self.assertIn('nothing to make native', octo_glue8.refusal({'QWEN_FAST_TP': '4', octo_glue8.FLAG: '1', 'QWEN_FAST_OCTO': 'live', octo_glue8.T2_FLAG: '0',
                                                                   'QWEN_C2_GATE_PROFILE': '1'}))
        self.assertIsNone(octo_glue8.refusal({'QWEN_FAST_TP': '4', 'QWEN_FAST_OCTO': 'live'}), 'flag off: nothing to refuse')

    def test_octo_spans(self):
        self.assertTrue(octo_glue8.is_octo_spans([(8 * user, 8 * user + 8) for user in range(8)]))
        self.assertFalse(octo_glue8.is_octo_spans([(16 * user, 16 * user + 16) for user in range(4)]))
        self.assertFalse(octo_glue8.is_octo_spans([(8 * user, 8 * user + 8) for user in range(4)]))
        self.assertFalse(octo_glue8.is_octo_spans(None))

    def test_the_markers_are_not_the_vglue_ones(self):
        self.assertFalse(octo_glue8.FALLBACK.startswith(tp4_vglue.FALLBACK))
        self.assertFalse(octo_glue8.ENGAGED.startswith(tp4_vglue.ENGAGED))
        self.assertEqual(octo_glue8.SERVED, tp4_vglue.EIGHT_ROW_SERVED)
        self.assertEqual(octo_glue8.VGLUE_ENGAGED, tp4_vglue.ENGAGED)


ALL_ON = {'QWEN_FAST_TP': '4', 'QWEN_FAST_OCTO': 'alternate', 'QWEN_C2_GATE_PROFILE': '1', octo_glue8.FLAG: '1', octo_glue8.V2_FLAG: '1', octo_glue8.V1_FLAG: '1', octo_glue8.T2_FLAG: '1'}


def clean_log(sites=octo_glue8.SITES):
    lines = ['[PINDIAG] tp4 octo glue8 engaged site=%s users=8 rows=8' % site for site in sites]
    counts = ' '.join('glue8_%s=48' % site for site in sites)
    lines.append('[PINDIAG] tp4 vglue engaged site=packed_verify levers=gdn_glue,gdn_block_conv gdn_glue=48 gdn_merge=48 %s' % counts)
    lines.append('[PINDIAG] tp4 vglue engaged site=packed_verify levers=gdn_glue,gdn_block_conv gdn_glue=48 gdn_merge=48')       # an M3 block's capture
    return '\n'.join(lines) + '\n'


class SmokeRuleTests(unittest.TestCase):
    def test_a_clean_arm_has_no_problem(self):
        self.assertEqual(octo_glue8.glue8_problems(clean_log(), ALL_ON), [])

    def test_the_real_producers_lines_parse(self):
        lines = []
        with patch.object(octo_glue8, 'log_line', side_effect=lines.append), patch.object(octo_glue8, '_LOGGED', set()):
            tp4_vglue.take()
            for site in octo_glue8.SITES:
                for layer in range(48):
                    octo_glue8.note_engaged(site)
            counts = tp4_vglue.take()
        self.assertEqual(counts, dict((name, 48) for name in ('glue8_split', 'glue8_merge', 'glue8_block_conv', 'glue8_windows')))
        text = '\n'.join(lines + [tp4_vglue.marker('packed_verify', levers='gdn_glue', **counts)])
        self.assertEqual(len(lines), 4, 'one engaged line per site per process')
        self.assertEqual(octo_glue8.glue8_problems(text, ALL_ON), [])

    def test_a_missing_site_line_or_count_is_a_problem(self):
        problems = octo_glue8.glue8_problems(clean_log(('split', 'merge', 'windows')), ALL_ON)
        self.assertTrue(any('site=block_conv' in problem for problem in problems), problems)
        partial = clean_log().replace('glue8_split=48', 'glue8_split=47')
        problems = octo_glue8.glue8_problems(partial, ALL_ON)
        self.assertTrue(any('site=split engaged on 47 of 48' in problem for problem in problems), problems)

    def test_a_fallback_line_is_a_problem(self):
        text = clean_log() + octo_glue8.fallback_line('merge', 'no fit') + '\n'
        problems = octo_glue8.glue8_problems(text, ALL_ON)
        self.assertTrue(any('fell back' in problem and 'site=merge' in problem for problem in problems), problems)

    def test_the_served_path_marker_at_a_native_site_is_a_problem_and_at_a_lever_that_is_off_it_is_not(self):
        served = '[PINDIAG] tp4 vglue octo 8-row served path site=split reason=eight-row users\n'
        problems = octo_glue8.glue8_problems(clean_log() + served, ALL_ON)
        self.assertTrue(any('served 8-row path ran at site=split' in problem for problem in problems), problems)
        v2_off = dict(ALL_ON)
        del v2_off[octo_glue8.V2_FLAG], v2_off[octo_glue8.V1_FLAG]
        self.assertEqual(octo_glue8.sites_wanted(v2_off), ['windows'])
        text = clean_log(('windows',)) + served
        self.assertEqual(octo_glue8.glue8_problems(text, v2_off), [])

    def test_flag_off_means_no_glue8_line_and_no_glue8_count(self):
        off = {'QWEN_FAST_TP': '4', 'QWEN_FAST_OCTO': 'alternate'}
        self.assertEqual(octo_glue8.glue8_problems('[PINDIAG] tp4 vglue octo 8-row served path site=split reason=x\n', off), [])
        self.assertEqual(len(octo_glue8.glue8_problems(clean_log(), off)), 2)
        self.assertEqual(len(octo_glue8.glue8_problems(octo_glue8.fallback_line('merge', 'x'), off)), 1)

    def test_an_unservable_configuration_is_one_problem_not_a_pile(self):
        problems = octo_glue8.glue8_problems('', {'QWEN_FAST_TP': '4', octo_glue8.FLAG: '1', 'QWEN_C2_GATE_PROFILE': '1'})
        self.assertEqual(len(problems), 1)
        self.assertIn('QWEN_FAST_OCTO', problems[0])


class FilesTests(unittest.TestCase):
    def test_runtime_files_exist_and_are_the_new_modules(self):
        for name in octo_glue8.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file(), name)
        self.assertEqual(set(octo_glue8.RUNTIME_FILES), {'octo_glue8.py', 'gdn_rows_dma8_tp.py', 'gdn_rows_dma8_tp.cpp', 'gdn_block_conv8_tp.py',
                                                         'gdn_conv_windows_packed8.py', 'gdn_conv_windows_packed8.cpp'})

    def test_the_new_modules_are_py37_stdlib_at_import(self):
        for name in ('octo_glue8.py', 'gdn_rows_dma8_tp.py', 'gdn_block_conv8_tp.py', 'gdn_conv_windows_packed8.py'):
            text = (HERE / name).read_text()
            for line in text.splitlines():
                if re.match(r'^(import|from) ', line):
                    self.assertNotRegex(line, r'^(import|from) (ttnn|torch)\b', (name, line))

    def test_no_lf_violations_or_private_strings_in_the_new_files(self):
        for name in octo_glue8.RUNTIME_FILES + ('test_octo_glue8.py',):
            data = (HERE / name).read_bytes()
            self.assertNotIn(b'\r', data, name)
            if name == 'test_octo_glue8.py':
                continue                                # this file names the needles
            text = data.decode('utf-8')
            for needle in ('/home/', '192.168', '10.10.', 'thatch'):
                self.assertNotIn(needle, text, (name, needle))


if __name__ == '__main__':
    unittest.main()
