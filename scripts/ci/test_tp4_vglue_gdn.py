"""V2 (QWEN_FAST_TP4_GDN_GLUE) and V1 (QWEN_FAST_TP4_GDN_BLOCK_CONV): the half-tile row mover against the served split, merge
and window stacking, and the twin class against the pinned one.

The kernel (gdn_rows_dma_tp.cpp) is emulated from its own loops over the very runtime arguments the host planner writes, on
32x32 int16 tiles (bf16 bit patterns, so -0, denormals and NaN payloads are real values), and compared with what the served
ops define: a raw tile Slice for users 0 and 2, an untilize / slice / tilize round trip that maps a zero-exponent bf16 to +0
for users 1 and 3, an untilize / concat / tilize that does the same to every user. The value rule is the one measured on card
M (gdn_prefill_conv_exact.py); what a card must still confirm is that the ops really apply it, which is what the in-trace
audit (QWEN_FAST_TP4_VGLUE_AUDIT) and the edge probe are for.

Run at py 3.11: `py -3.11 -m unittest test_tp4_vglue_gdn` from scripts/ci.
"""

import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import gdn_rows_dma_tp as rows_dma
import tp4_vglue
import tp_shapes
from test_gdn_tp_twins import four, pair

HERE = Path(__file__).resolve().parent


def env(**values):
    return patch.dict(os.environ, values)


# --- int16 tiles ------------------------------------------------------------------------------------------------------------

def geometry():
    with four():
        return tp_shapes.active()


def specials(generator, shape):
    """Random bf16 bit patterns with every edge value present: +-0, denormals of both signs, infinities, NaN payloads."""
    bits = torch.randint(-32768, 32768, shape, generator=generator, dtype=torch.int16)
    flat = bits.reshape(-1)
    palette = torch.tensor([0, -32768, 1, -32767, 0x007F, -32641, 0x7F80, -128, 0x7FC0, -64, 0x3F80, -16512],
                           dtype=torch.int16)
    picks = torch.randint(0, flat.numel(), (flat.numel() // 3,), generator=generator)
    flat[picks] = palette[torch.randint(0, len(palette), (len(picks),), generator=generator)]
    return bits


def canon(bits):
    """The served round trip's value rule: a zero exponent (-0 and every denormal, either sign) comes out +0."""
    raw = bits.to(torch.int32) & 0xFFFF
    return torch.where((raw & 0x7F80) == 0, torch.zeros_like(bits), bits)


def to_tiles(matrix):
    """(rows, columns) int16 -> {page: 32x32 tile}, rows and columns padded to whole tiles, page = tile row * columns + column."""
    rows, columns = matrix.shape
    tile_rows, tile_columns = -(-rows // 32), -(-columns // 32)
    padded = torch.zeros(tile_rows * 32, tile_columns * 32, dtype=torch.int16)
    padded[:rows, :columns] = matrix
    return {row * tile_columns + column: padded[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32].clone()
            for row in range(tile_rows) for column in range(tile_columns)}


def run_rows_kernel(argument_lists, sources, destinations, canon_denorm=True):
    """gdn_rows_dma_tp.cpp's loops over each core's [tasks, eight words per task] list. `sources` and `destinations` map a
    buffer address to {page: tile}."""
    for words in argument_lists:
        for task in range(words[0]):
            base = 1 + task * 8
            destination, page = words[base], words[base + 1]
            tile = torch.zeros(32, 32, dtype=torch.int16)
            for half in (0, 1):
                address, source_page, mode = words[base + 2 + 3 * half:base + 5 + 3 * half]
                if mode == 0:
                    continue
                rows = sources[address][source_page][16 * ((mode - 1) & 1):16 * ((mode - 1) & 1) + 16].clone()
                if mode >= 3:
                    if canon_denorm:
                        rows = canon(rows)
                    else:
                        rows = torch.where(rows == -32768, torch.zeros_like(rows), rows)
                tile[16 * half:16 * half + 16] = rows
            destinations[destination][page] = tile


def execute(tasks, sources, destinations, cores=110):
    """Plan `tasks` over `cores`, resolve tensor indices to fake addresses (sources 100+, destinations 900+) and run."""
    per_core = rows_dma.distribute(tasks, cores)
    source_addresses = [100 + index for index in range(len(sources))]
    destination_addresses = [900 + index for index in range(len(destinations))]
    arguments = rows_dma.runtime_arguments(per_core, source_addresses, destination_addresses)
    for words in arguments:
        assert len(words) <= rows_dma.MAX_ARGUMENT_WORDS
    built = {address: {} for address in destination_addresses}
    run_rows_kernel(arguments, dict(zip(source_addresses, sources)), built)
    return [built[address] for address in destination_addresses]


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.width = geometry().gdn_qkvzab            # 4120 -> 129 tile columns
        self.generator = torch.Generator().manual_seed(7)
        self.columns = rows_dma.tile_columns(self.width)
        self.block = specials(self.generator, (64, self.columns * 32))
        self.block[:, self.width:] = 0                 # the padding columns of a TILE tensor read back zero

    def test_the_split_is_the_served_slices(self):
        pieces = execute(rows_dma.split_pieces(4, self.width), [to_tiles(self.block)], [{} for _ in range(4)])
        for user in range(4):
            rows = self.block[16 * user:16 * user + 16]
            served = canon(rows) if user in (1, 3) else rows          # untilize / slice / tilize vs a tile Slice
            expected = torch.zeros(32, self.columns * 32, dtype=torch.int16)
            expected[:16] = served
            self.assertEqual(sorted(pieces[user]), list(range(self.columns)), 'every page written once')
            for page, tile in pieces[user].items():
                self.assertTrue(torch.equal(tile, expected[:, page * 32:(page + 1) * 32]), (user, page))

    def test_users_one_and_three_are_the_canonical_ones_and_only_they(self):
        self.assertEqual([rows_dma.served_canon(user) for user in range(4)], [False, True, False, True])
        self.assertEqual([rows_dma.position(user) for user in range(4)], [(0, 0), (0, 1), (1, 0), (1, 1)])
        with_negative_zero = torch.zeros(64, self.columns * 32, dtype=torch.int16)
        with_negative_zero[:, :] = -32768
        pieces = execute(rows_dma.split_pieces(4, self.width), [to_tiles(with_negative_zero)], [{} for _ in range(4)])
        for user in range(4):
            first = pieces[user][0][:16]
            self.assertTrue(bool((first == (0 if user % 2 else -32768)).all()), user)

    def test_the_canonical_block_takes_both_halves_from_one_tile(self):
        (built,) = execute(rows_dma.canon_block(4, self.width), [to_tiles(self.block)], [{}])
        expected = self.block.clone()
        expected[16:32] = canon(self.block[16:32])
        expected[48:64] = canon(self.block[48:64])
        self.assertEqual(sorted(built), list(range(2 * self.columns)))
        for page, tile in built.items():
            row, column = divmod(page, self.columns)
            self.assertTrue(torch.equal(tile, expected[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32]), page)

    def test_the_merge_is_the_served_untilize_concat_tilize(self):
        width = geometry().gdn_value                   # 1536: the K5-A output
        columns = rows_dma.tile_columns(width)
        outputs = [specials(self.generator, (16, columns * 32)) for _ in range(4)]
        # K5-A leaves the tile's padding rows anything: only rows 0-15 are the output
        tiles = []
        for output in outputs:
            tile_set = to_tiles(output)
            for page in tile_set:
                tile_set[page][16:] = torch.randint(-32768, 32768, (16, 32), generator=self.generator, dtype=torch.int16)
            tiles.append(tile_set)
        (built,) = execute(rows_dma.merge_outputs(4, width), tiles, [{}])
        expected = canon(torch.cat(outputs, dim=0))
        self.assertEqual(sorted(built), list(range(2 * columns)))
        for page, tile in built.items():
            row, column = divmod(page, columns)
            self.assertTrue(torch.equal(tile, expected[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32]), page)

    def test_an_odd_user_count_pads_the_last_tile_with_zeros(self):
        width = 96
        outputs = [specials(self.generator, (16, 96)) for _ in range(3)]
        (built,) = execute(rows_dma.merge_outputs(3, width), [to_tiles(output) for output in outputs], [{}])
        expected = torch.zeros(64, 96, dtype=torch.int16)
        expected[:48] = canon(torch.cat(outputs, dim=0))
        for page, tile in built.items():
            row, column = divmod(page, 3)
            self.assertTrue(torch.equal(tile, expected[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32]), page)

    def test_windows_stack_raw_and_unstack_raw(self):
        width = geometry().gdn_qkv
        columns = rows_dma.tile_columns(width)
        windows = [specials(self.generator, (16, columns * 32)) for _ in range(4)]
        (block,) = execute(rows_dma.stack_users(4, width), [to_tiles(window) for window in windows], [{}])
        expected = torch.cat(windows, dim=0)
        for page, tile in block.items():
            row, column = divmod(page, columns)
            self.assertTrue(torch.equal(tile, expected[row * 32:(row + 1) * 32, column * 32:(column + 1) * 32]), page)
        # and back: the advanced windows the commit reads, rows 0-15 the user's, rows 16-31 zero
        back = execute(rows_dma.unstack_users(4, columns, 0, columns), [block], [{} for _ in range(4)])
        for user in range(4):
            for page, tile in back[user].items():
                self.assertTrue(torch.equal(tile[:16], windows[user][:, page * 32:(page + 1) * 32]), (user, page))
                self.assertFalse(bool(tile[16:].any()))

    def test_unstack_reads_a_column_range_of_a_wider_block(self):
        columns = self.columns
        tasks = rows_dma.unstack_users(4, columns, 80, 48)            # z: columns 80..127 of the projection
        pieces = execute(tasks, [to_tiles(self.block)], [{} for _ in range(4)])
        for user in range(4):
            self.assertEqual(sorted(pieces[user]), list(range(48)))
            for column, tile in pieces[user].items():
                self.assertTrue(torch.equal(tile[:16], self.block[16 * user:16 * user + 16, (80 + column) * 32:(81 + column) * 32]))

    def test_canonical_flag_off_keeps_denormals(self):
        block = torch.tensor([[-32768, 1, -32767, 0x3F80]] * 16 + [[-32768, 1, -32767, 0x3F80]] * 16, dtype=torch.int16)
        tile = torch.zeros(32, 32, dtype=torch.int16)
        tile[:, :4] = block
        tasks = [(0, 0, (0, 0, rows_dma.mode(1, True)), (0, 0, 0))]
        arguments = rows_dma.runtime_arguments(rows_dma.distribute(tasks, 1), [1], [2])
        built = {2: {}}
        run_rows_kernel(arguments, {1: {0: tile}}, built, canon_denorm=False)
        self.assertEqual(built[2][0][0, :4].tolist(), [0, 1, -32767, 0x3F80])
        built = {2: {}}
        run_rows_kernel(arguments, {1: {0: tile}}, built, canon_denorm=True)
        self.assertEqual(built[2][0][0, :4].tolist(), [0, 0, 0, 0x3F80])

    def test_a_task_list_the_kernel_cannot_carry_is_unsupported(self):
        tasks = [(0, page, (0, page, 1), (0, 0, 0)) for page in range(40 * 110)]
        with self.assertRaises(rows_dma.Unsupported):
            rows_dma.distribute(tasks, 110)
        with self.assertRaises(rows_dma.Unsupported):
            rows_dma.distribute([], 110)

    def test_every_planned_destination_page_is_written_once(self):
        columns = self.columns
        for tasks, count in ((rows_dma.split_pieces(4, self.width), 4 * columns),
                             (rows_dma.canon_block(4, self.width), 2 * columns),
                             (rows_dma.merge_outputs(4, 1536), 2 * 48),
                             (rows_dma.stack_users(4, 2560), 2 * 80)):
            self.assertEqual(len(tasks), count)
            self.assertEqual(len({(task[0], task[1]) for task in tasks}), count)

    def test_the_kernel_text_matches_the_task_encoding(self):
        text = (HERE / 'gdn_rows_dma_tp.cpp').read_text()
        self.assertIn('constexpr uint32_t TASK_WORDS = %d;' % rows_dma.TASK_WORDS, text)
        self.assertIn('constexpr uint32_t LANES = %d;' % rows_dma.LANES, text)
        self.assertEqual(rows_dma.SCRATCH_BYTES, rows_dma.LANES * 2048)
        self.assertIn('#define CANON_DENORM 1', text)
        # the canonical rule the emulator runs is the kernel's: zero exponent -> +0 (CANON_DENORM), else 0x8000 -> +0
        self.assertIn('(value & 0x7F80u) == 0 ? 0u : value', text)
        self.assertIn('value == 0x8000u ? 0u : value', text)


if __name__ == '__main__':
    unittest.main()
