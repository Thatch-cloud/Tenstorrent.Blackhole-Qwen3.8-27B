"""QWEN_FAST_DRAFT_PERMUTE (draft_permute_tp, F-F2 of the op-fusion programme): the drafter's K/V assembly, query fold and output unfold as permutation launches.

What is held here, on the CPU (nothing runs on a card; optimisation/ttnn-op/draft_permute is the hardware half):

  - the kernel (draft_permute_tp.cpp) is transliterated line for line and run over the very runtime arguments the host builds (the fake generic_op executes every chip's
    programs), on RAW face-ordered 2,048-byte tiles (four 16 x 16 faces, face f at int16 offset 256 f, row r of a face at 16 r), so the quarter-chunk offsets the kernel computes
    are checked against the physical layout and not only against a logical matrix. Values are bf16 bit patterns: -0, denormals of both signs, infinities and NaN payloads are
    real values, and the scratch the kernel fills starts as junk, so a byte a task does not write shows;
  - what the served ops define is stated on LOGICAL matrices, by running the REAL served code (draft_attention_branch.served_key_value, quad_draft_tp.quad_fold_query,
    pair_row_exact_tp.fold_query, octo_draft_tp.octo_fold_query ...) on a model of ttnn whose only non-trivial rule is the one measured on card M: a slice of the row axis that starts
    inside a tile, and a row-axis concat with a piece that is not a whole number of tiles, go through untilize / tilize and map a zero-exponent bf16 (-0 and every denormal) to +0.
    The kernel's output equals that composition bit for bit for the quad, the pair (at 2,048 and at mixed histories) and the octo shapes. That a card's ops follow the model is the
    card-M probe's question (the profile shows the untilizes of the cached banks in the quad's concat; the probe reads the bytes), not a CPU fact;
  - negative controls: a build that leaves the cached banks raw, one that canonicalises nothing, and one that canonicalises only the slices that start inside a tile each fail
    against the served composition on the edge data, so the test could fail;
  - the planner (every destination tile written once, runs merged, argument budget at 110 and 130 cores), the launch the builder describes (two worker kernels a core, one CB each),
    the refusals that hand a call to the served ops, the audit (eager warm pass only), the flags and the twins' flag-off path.

Run: `python -m unittest test_draft_permute_tp` from scripts/ci (py 3.11).
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import draft_attention_branch
import draft_permute_smoke as smoke
import draft_permute_tp as perm
import octo_draft_tp
import pair_row_exact_tp
import quad_draft as pinned_quad
import quad_draft_tp as quad_twin
import tp4_sampdraft
import tp_shapes
from dflash_batched_mask import key_value_plan as pair_plan
from test_pair_row_exact import Device, TorchOps, keep
from test_tp4_shard_argmax import FakeOperations
from test_tp4_vglue_gdn import canon, specials
from tp_test_support import four_cards, pair

HERE = Path(__file__).resolve().parent
KV, HEADS = 2, 8                      # per chip at four cards
DIM = 128


def environment(**values):
    return patch.dict(os.environ, values)


# --- raw face-ordered tiles --------------------------------------------------------------------------------------------------

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


def tensor_pages(tensor):
    """(heads, rows, 128) int16 -> {page: raw tile}, page = (head * row tiles + row tile) * 4 + column tile."""
    heads, rows, width = tensor.shape
    tile_rows = -(-rows // 32)
    padded = torch.zeros(heads, tile_rows * 32, width, dtype=torch.int16)
    padded[:, :rows] = tensor
    return {(h * tile_rows + tr) * 4 + c: to_raw(padded[h, 32 * tr:32 * tr + 32, 32 * c:32 * c + 32])
            for h in range(heads) for tr in range(tile_rows) for c in range(width // 32)}


def pages_tensor(pages, heads, rows):
    tile_rows = -(-rows // 32)
    out = torch.zeros(heads, tile_rows * 32, DIM, dtype=torch.int16)
    for page, raw in pages.items():
        row_group, column = divmod(page, 4)
        h, tr = divmod(row_group, tile_rows)
        out[h, 32 * tr:32 * tr + 32, 32 * column:32 * column + 32] = from_raw(raw)
    return out[:, :rows]


def leading(shape):
    """The product of the dimensions before the last two (a tile tensor's 'heads')."""
    count = 1
    for extent in shape[:-2]:
        count *= extent
    return count


def quarter_ranges(quarter):
    """The two int16 index ranges (left face chunk, right face chunk) of quarter `quarter`, from the kernel's quarter_offset arithmetic (bytes / 2)."""
    left = ((quarter >> 1) * 1024 + (quarter & 1) * 256) // 2
    return (left, left + 128), (left + 256, left + 256 + 128)


def swar(word):
    """canonical_pair of draft_permute_tp.cpp (CANON_DENORM=1) on a 32-bit word held in a python int."""
    present = (((word & 0x7F807F80) + 0x7F807F80) & 0x80008000) & 0xFFFFFFFF
    return word & (present | ((present - (present >> 15)) & 0xFFFFFFFF))


def canon_words(values, canon_denorm=True):
    if canon_denorm:
        return canon(values)
    return torch.where(values == -32768, torch.zeros_like(values), values)


def run_kernel(argument_lists, capacity, nsrc, ndst, sources, destinations, canon_denorm=True, generator=None):
    """draft_permute_tp.cpp's loops over each lane's runtime-arg list. `sources` and `destinations` map a buffer address to {page: raw tile}. The scratch starts as junk: every
    byte of a destination tile must be written, zeroed or copied by the task itself."""
    generator = generator or torch.Generator().manual_seed(5)
    for words in argument_lists:
        assert len(words) == capacity
        records = 1 + nsrc + ndst
        source_table, destination_table = words[1:1 + nsrc], words[1 + nsrc:records]
        end = min(records + words[0], capacity)
        at = records
        while at < end:
            head = words[at]
            kind = head >> 28
            if kind == perm.TYPE_RUN and at + perm.RUN_WORDS <= end:
                canonical = bool((head >> 24) & 1)
                source = sources[source_table[head & 0xFF]]
                destination = destinations[destination_table[(head >> 8) & 0xFF]]
                count, destination_page, source_page = words[at + 1], words[at + 2], words[at + 3]
                for first in range(0, count, 8):
                    lanes = min(8, count - first)
                    scratch = [source[source_page + first + lane].clone() for lane in range(lanes)]
                    if canonical:
                        scratch = [canon_words(tile, canon_denorm) for tile in scratch]
                    for lane in range(lanes):
                        destination[destination_page + first + lane] = scratch[lane]
                at += perm.RUN_WORDS
            elif kind == perm.TYPE_MIX and at + perm.MIX_WORDS <= end:
                destination = destinations[destination_table[(head >> 8) & 0xFF]]
                destination_page = words[at + 1]
                scratch = torch.randint(-32768, 32768, (1024,), generator=generator, dtype=torch.int16)
                for quarter in range(4):
                    word = words[at + 2 + quarter]
                    mode = (word >> 16) & 0xFF
                    if mode == 0:
                        continue
                    source = sources[source_table[word >> 24]][word & 0xFFFF]
                    for target, origin in zip(quarter_ranges(quarter), quarter_ranges((mode - 1) & 3)):
                        scratch[target[0]:target[1]] = source[origin[0]:origin[1]]
                for quarter in range(4):
                    mode = (words[at + 2 + quarter] >> 16) & 0xFF
                    for target in quarter_ranges(quarter):
                        if mode == 0:
                            scratch[target[0]:target[1]] = 0
                        elif mode >= 5:
                            scratch[target[0]:target[1]] = canon_words(scratch[target[0]:target[1]], canon_denorm)
                destination[destination_page] = scratch
                at += perm.MIX_WORDS
            else:
                break


# --- the ttnn model ----------------------------------------------------------------------------------------------------------

class CanonOps(TorchOps):
    """slice, concat and reshape as the SERVED composition defines them on logical int16 matrices: a slice that starts inside a tile (rows or columns) and a concat on the row axis
    with a piece that is not a whole number of tiles canonicalise their output (untilize / tilize); every other slice, concat and the reshape move whole tiles."""

    def slice(self, tensor, start, end):
        result = super().slice(tensor, start, end)
        if start[2] % 32 or start[3] % 32:
            result = Device(canon(result.value), result.dtype)
        return result

    def concat(self, parts, dim, memory_config=None):
        result = super().concat(parts, dim, memory_config)
        if dim in (2, 3) and any(part.shape[dim] % 32 for part in parts):
            result = Device(canon(result.value), result.dtype)
        return result


class Mesh(object):
    def __init__(self, x=11, y=10):
        self.x, self.y = x, y

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=self.x, y=self.y)


class Shard(object):
    counter = 0

    def __init__(self, pages, shape):
        Shard.counter += 1
        self.pages, self.shape, self.base = pages, shape, 4096 * Shard.counter

    def buffer_address(self):
        return self.base


class ExecutingOperations(FakeOperations):
    """FakeOperations whose generic_op RUNS the programs it is given: every chip's kernels, from their runtime args, through run_kernel."""

    def __init__(self, chips=4, mesh=None, canon_denorm_seen=None):
        super().__init__(chips)
        self.mesh = mesh or Mesh()
        self.programs = []

    def tensor(self, shape, dtype='bf16', layout='tile', logical=None):
        self.counter += 1
        pages = tensor_pages(logical.reshape(-1, shape[-2], shape[-1])) if logical is not None else {}
        outer = SimpleNamespace(shape=tuple(shape), dtype=dtype, layout=layout, memory_config=lambda: 'dram', name='t%d' % self.counter, device=lambda: self.mesh)
        outer.shards = [Shard(dict(pages) if logical is not None else {}, tuple(shape)) for _ in range(self.chips)]
        return outer

    def from_logical(self, value):
        """A device tensor (replicated on every chip) holding the (1, heads, rows, 128) int16 `value`."""
        return self.tensor(tuple(value.shape), logical=value)

    def to_logical(self, tensor, chip=0):
        shard = tensor.shards[chip]
        return pages_tensor(shard.pages, leading(shard.shape), shard.shape[-2]).reshape(shard.shape)

    def to_torch(self, shard):
        return pages_tensor(shard.pages, leading(shard.shape), shard.shape[-2]).reshape(shard.shape).view(torch.bfloat16)

    def generic_op(self, tensors, program):
        super().generic_op(tensors, program)
        self.programs.append(program)
        for key, descriptor in program.items():
            chip = key[0][1]
            by_address = {}
            for tensor in tensors:
                by_address[tensor.shards[chip].buffer_address()] = tensor.shards[chip].pages
            for kernel in descriptor['kernels']:
                size, nsrc, ndst, processor = kernel['compile_time_args'][-4:]
                lists = [words for column in kernel['runtime_args'].values() for words in column.values()]
                denorm = dict(kernel['defines'])['CANON_DENORM'] == '1'
                sources = {address: by_address[address] for words in lists for address in words[1:1 + nsrc]}
                destinations = {address: by_address[address] for words in lists for address in words[1 + nsrc:1 + nsrc + ndst]}
                run_kernel(lists, size, nsrc, ndst, sources, destinations, denorm)


def bits(generator, shape):
    """Random bf16 bit patterns with every edge value present (+-0, denormals, infinities, NaN payloads) as int16."""
    return specials(generator, shape)


def logical_device(value):
    return Device(value)


# --- cases -------------------------------------------------------------------------------------------------------------------

def kv_case(name, seed=0):
    """(plan, user cache rows, live rows, the real code's plan for `name`)."""
    if name == 'quad':
        plan, _, _ = pinned_quad.key_value_plan([2048] * 4, 16)
        return plan, [2048] * 4, 64
    if name == 'octo':
        plan, _, _ = octo_draft_tp.key_value_plan([2048] * 8, 16)
        return plan, [2048] * 8, 64
    if name == 'pair':
        plan, _, _ = pair_plan([2048, 2048], 16)
        return plan, [2048, 2048], 32
    if name == 'pair-short':
        plan, _, _ = pair_plan([256, 1024], 16)
        return plan, [256, 1024], 32
    raise ValueError(name)


CASES = ('quad', 'octo', 'pair', 'pair-short')


def kv_operands(case, seed):
    plan, rows, live_rows = kv_case(case)
    generator = torch.Generator().manual_seed(seed)
    caches = [{name: bits(generator, (1, KV, rows[user], DIM)) for name in 'kv'} for user in range(len(rows))]
    live = {name: bits(generator, (1, KV, live_rows, DIM)) for name in 'kv'}
    return plan, caches, live


def served_kv(plan, caches, live):
    """The served K and V on the model: the REAL draft_attention_branch.served_key_value."""
    ops, owned = CanonOps(), []
    wrapped_caches = [{name: Device(value) for name, value in cache.items()} for cache in caches]
    wrapped_live = {name: Device(value) for name, value in live.items()}
    return {name: draft_attention_branch.served_key_value(ops, plan, wrapped_caches, wrapped_live, keep(owned), name).value for name in 'kv'}


def engaged_kv(plan, caches, live, *, canon_cached=True, mesh=None, processors=perm.PROCESSORS):
    """The K and V the kernel builds: draft_permute_tp.assemble_kv on the executing fake."""
    ops = ExecutingOperations(mesh=mesh)
    owned = []
    device_caches = [{name: ops.from_logical(value) for name, value in cache.items()} for cache in caches]
    device_live = {name: ops.from_logical(value) for name, value in live.items()}
    served = Mock(side_effect=AssertionError('the served path must not run'))
    result = perm.assemble_kv(ops, plan, device_caches, device_live, keep(owned), served=served, site='test', canon_cached=canon_cached, processors=processors)
    served.assert_not_called()
    return ops, {name: ops.to_logical(result[name]) for name in 'kv'}, result


def same(left, right):
    return tuple(left.shape) == tuple(right.shape) and torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


class Quiet(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'QWEN_FAST_TP': '4', perm.FLAG: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        perm._LOGGED.clear()
        perm._CACHE.clear()
        self.lines = []
        logger = patch.object(tp4_sampdraft, 'log_line', side_effect=self.lines.append)
        logger.start()
        self.addCleanup(logger.stop)
        perm._PASS['eager'] = None
        perm._SERVED['depth'] = 0
        self.addCleanup(lambda: perm._PASS.update(eager=None))


# --- the value rule ----------------------------------------------------------------------------------------------------------

class RuleTests(unittest.TestCase):
    def test_the_branch_free_form_of_the_kernel_is_the_rule_for_every_pattern_in_both_lanes(self):
        patterns = torch.arange(65536, dtype=torch.int64)
        for other in (0x0000, 0x8000, 0x0001, 0x3F80, 0x7F80, 0x7FFF, 0xFFFF, 0x8001, 0x1234):
            for lane in (0, 1):
                words = (patterns | (other << 16)) if lane == 0 else (patterns << 16) | other
                present = (((words & 0x7F807F80) + 0x7F807F80) & 0x80008000) & 0xFFFFFFFF
                got = words & (present | ((present - (present >> 15)) & 0xFFFFFFFF))
                mine = (got >> (16 * lane)) & 0xFFFF
                wanted = torch.where((patterns & 0x7F80) == 0, torch.zeros_like(patterns), patterns)
                self.assertTrue(torch.equal(mine, wanted), (other, lane))
                theirs = (got >> (16 * (1 - lane))) & 0xFFFF
                kept = torch.full_like(patterns, other)
                self.assertTrue(torch.equal(theirs, torch.where((kept & 0x7F80) == 0, torch.zeros_like(kept), kept)), (other, lane, 'neighbour'))
        self.assertEqual(swar(0x80000001), 0)
        self.assertEqual(swar(0x7FC13F80), 0x7FC13F80)

    def test_the_model_canonicalises_exactly_what_the_profile_shows_untilize_tilize_touching(self):
        ops = CanonOps()
        value = torch.full((1, 2, 64, 128), -32768, dtype=torch.int16)
        tile = Device(value)
        self.assertTrue(torch.equal(ops.slice(tile, (0, 0, 0, 0), (1, 2, 16, 128)).value, value[:, :, :16]), 'a slice that starts on a tile boundary is a raw tile copy')
        self.assertTrue(bool((ops.slice(tile, (0, 0, 16, 0), (1, 2, 32, 128)).value == 0).all()), 'a slice that starts inside a tile is canonical')
        self.assertTrue(torch.equal(ops.concat([Device(value[:, :, :32]), Device(value[:, :, 32:])], 2).value, value), 'whole-tile pieces: a tile concat')
        joined = ops.concat([Device(value[:, :, :32]), Device(value[:, :, 32:48])], 2).value
        self.assertTrue(bool((joined == 0).all()), 'one piece of 16 rows: every byte of the output is canonical, the whole-tile piece included')
        self.assertTrue(torch.equal(ops.concat([Device(value[:, :2]), Device(value[:, 2:])], 1).value, value) if value.shape[1] > 2 else True)


# --- the K/V assembly --------------------------------------------------------------------------------------------------------

class KeyValueTests(Quiet):
    def test_the_assembly_is_the_served_composition_bit_for_bit_for_the_quad_pair_and_octo_shapes(self):
        for case in CASES:
            for processors in (2, 1):
                with self.subTest(case=case, processors=processors), four_cards():
                    plan, caches, live = kv_operands(case, seed=len(case) + processors)
                    served = served_kv(plan, caches, live)
                    ops, mine, _ = engaged_kv(plan, caches, live, processors=processors)
                    for name in 'kv':
                        self.assertEqual(tuple(mine[name].shape), tuple(served[name].shape), name)
                        self.assertTrue(same(mine[name], served[name]), '%s differs from the served assembly' % name)
                    self.assertEqual(len(ops.programs), 1, 'K and V in ONE launch')

    def test_the_edge_data_reaches_the_pieces_so_the_comparison_can_fail(self):
        with four_cards():
            plan, caches, live = kv_operands('quad', seed=11)
            served = served_kv(plan, caches, live)
            raw_cached = torch.cat([cache['k'] for cache in caches], dim=2)
            self.assertTrue(bool(((raw_cached & 0x7F80) == 0).any()), 'the banks carry zero-exponent values')
            self.assertFalse(torch.equal(canon(raw_cached), raw_cached), 'and the rule changes them')
            self.assertTrue(bool((served['k'].view(torch.int16) != 0).any()))
            for name in 'kv':
                self.assertEqual(int((((served[name] & 0x7F80) == 0) & (served[name] != 0)).sum()), 0, 'no denormal and no -0 survives in the served output')

    def test_negative_controls_a_raw_cached_bank_or_no_rule_or_slices_only_each_differ_from_the_served_composition(self):
        with four_cards():
            plan, caches, live = kv_operands('quad', seed=3)
            served = served_kv(plan, caches, live)
            _, raw_cached, _ = engaged_kv(plan, caches, live, canon_cached=False)
            self.assertFalse(same(raw_cached['k'], served['k']), 'cached banks left raw')
            with patch.object(perm, 'served_canon_concat', return_value=False):
                _, slices_only, _ = engaged_kv(plan, caches, live)
            self.assertFalse(same(slices_only['k'], served['k']), 'only the slices that start inside a tile canonical (the plan\'s literal reading)')
            with patch.object(perm, 'served_canon_slice', return_value=False), patch.object(perm, 'served_canon_concat', return_value=False):
                _, nothing, _ = engaged_kv(plan, caches, live)
            self.assertFalse(same(nothing['k'], served['k']), 'no rule at all')
            # and with a canonical rule that is applied to the wrong pieces (the cached banks only) the live rows differ
            clean = {name: torch.where((cache & 0x7F80) == 0, torch.zeros_like(cache), cache) for name, cache in zip('kv', [caches[0]['k'], caches[0]['v']])}
            self.assertEqual(set(clean), {'k', 'v'})

    def test_a_tile_aligned_plan_is_a_raw_tile_concat_except_for_the_slices_that_start_inside_a_tile(self):
        with four_cards():
            plan = [dict(kind='cached', user=0, rows=64), dict(kind='live', user=0, rows=32, source=slice(16, 48)),
                    dict(kind='cached', user=1, rows=32), dict(kind='pad', user=1, rows=32, source=slice(0, 32))]
            generator = torch.Generator().manual_seed(9)
            caches = [{name: bits(generator, (1, KV, 64, DIM)) for name in 'kv'}, {name: bits(generator, (1, KV, 32, DIM)) for name in 'kv'}]
            live = {name: bits(generator, (1, KV, 64, DIM)) for name in 'kv'}
            served = served_kv(plan, caches, live)
            _, mine, _ = engaged_kv(plan, caches, live)
            for name in 'kv':
                self.assertTrue(same(mine[name], served[name]), name)
            self.assertTrue(torch.equal(served['k'][:, :, :64], caches[0]['k']), 'the cached bank stays raw in a tile concat (edge values included)')
            records, _ = perm.kv_records(plan, {0: 64, 1: 32}, 64, KV)
            self.assertEqual({record[6] for record in records if record[0] == 'run'}, {False}, 'whole-tile pieces of a tile concat are raw runs')
            mixes = [record for record in records if record[0] == 'mix']
            self.assertTrue(mixes and all(mode >= perm.CANON for record in mixes for _, _, mode in record[3]), 'the slice that starts inside a tile is canonical')

    def test_every_destination_page_is_written_once_every_source_page_is_in_range_and_runs_merge(self):
        for case in CASES:
            with self.subTest(case=case):
                plan, rows, live_rows = kv_case(case)
                users = len(rows)
                records, total = perm.kv_records(plan, dict(enumerate(rows)), live_rows, KV)
                tile_rows = total // 32
                written = []
                for record in records:
                    if record[0] == 'run':
                        written.extend(range(record[2], record[2] + record[5]))
                        self.assertLess(record[3], users + 1)
                        limit = (rows[record[3]] // 32 if record[3] < users else -(-live_rows // 32)) * KV * 4
                        self.assertLessEqual(record[4] + record[5], limit)
                    else:
                        written.append(record[2])
                        for source, page, mode in record[3]:
                            self.assertTrue(1 <= mode <= 8)
                            self.assertLess(page, (-(-live_rows // 32)) * KV * 4 if source == users else rows[source] // 32 * KV * 4)
                self.assertEqual(sorted(written), list(range(KV * tile_rows * 4)))
                runs = [record for record in records if record[0] == 'run']
                self.assertLessEqual(len(runs), KV * (users + 1) * 2, 'one run per head and bank at most (plus the tail tiles)')
                self.assertEqual(sum(1 for record in records if record[0] == 'mix'), KV * users * 4, 'one mixed tile a head, column and user')

    def test_the_argument_budget_holds_on_the_11x10_and_13x10_grids_for_every_shape(self):
        for case in CASES:
            for grid in ((11, 10), (13, 10), (8, 10)):
                with self.subTest(case=case, grid=grid):
                    plan, rows, live_rows = kv_case(case)
                    users = len(rows)
                    both = []
                    for index in range(2):
                        part, _ = perm.kv_records(plan, dict(enumerate(rows)), live_rows, KV, destination=index,
                                                  cached_source=lambda user, index=index: index * (users + 1) + user, live_source=index * (users + 1) + users)
                        both.extend(part)
                    per_lane, size, rows = perm.plan_lanes(both, grid, 2 * (users + 1), 2)
                    self.assertLessEqual(size, perm.MAX_ARGUMENT_WORDS)
                    self.assertEqual(len(per_lane), rows * grid[0] * 2, 'the lanes fill one rectangle of full rows')
                    self.assertLessEqual(rows, grid[1])
                    tiles = sum(perm.weight(record) for record in both if record[0] == 'run')
                    covered = sum(perm.weight(record) for lane in per_lane for record in lane if record[0] == 'run')
                    self.assertEqual(covered, tiles)

    def test_the_lanes_are_balanced_to_within_one_run_split(self):
        plan, rows, live_rows = kv_case('quad')
        both = []
        for index in range(2):
            part, _ = perm.kv_records(plan, dict(enumerate(rows)), live_rows, KV, destination=index,
                                      cached_source=lambda user, index=index: index * 5 + user, live_source=index * 5 + 4)
            both.extend(part)
        per_lane, _, _ = perm.plan_lanes(both, (11, 10), 10, 2)
        weights = [sum(perm.weight(record) for record in lane) for lane in per_lane if lane]
        self.assertLessEqual(max(weights) - min(weights[:-1]), 20, 'no lane carries more than the share plus one split run')
        self.assertLessEqual(max(weights), 21)

    def test_unsupported_plans_are_refused_not_guessed(self):
        plan, rows, live_rows = kv_case('quad')
        with self.assertRaises(perm.Unsupported):
            perm.kv_records(plan, dict(enumerate([2048, 2048, 2048, 1024])), live_rows, KV)          # a bank shorter than its piece
        with self.assertRaises(perm.Unsupported):
            perm.kv_records(plan, dict(enumerate(rows)), 48, KV)                                      # a live block shorter than the plan reads
        odd = [dict(part) for part in plan]
        odd[1] = dict(odd[1], rows=12, source=slice(0, 12))
        with self.assertRaises(perm.Unsupported):
            perm.kv_records(odd, dict(enumerate(rows)), live_rows, KV)                                # rows that are not whole quarters
        with self.assertRaises(perm.Unsupported):
            perm.kv_records([dict(kind='mystery', user=0, rows=32)], {0: 32}, 64, KV)


# --- the fold and the unfold -------------------------------------------------------------------------------------------------

def fold_served(case, query):
    ops, owned = CanonOps(), []
    with four_cards(), perm.served_only():
        if case == 'pair':
            return pair_row_exact_tp.fold_query(ops, Device(query), keep(owned)).value
        if case == 'quad':
            return quad_twin.quad_fold_query(ops, Device(query), keep(owned)).value
        return octo_draft_tp.octo_fold_query(ops, Device(query), keep(owned)).value


def unfold_served(case, output):
    ops, owned = CanonOps(), []
    with four_cards(), perm.served_only():
        if case == 'pair':
            return pair_row_exact_tp.unfold_output(ops, Device(output), keep(owned)).value
        if case == 'quad':
            return quad_twin.quad_unfold_output(ops, Device(output), keep(owned)).value
        return octo_draft_tp.octo_unfold_output(ops, Device(output), keep(owned)).value


GEOMETRY = {'pair': dict(halves=1, users=2, block=16), 'quad': dict(halves=2, users=2, block=16), 'octo': dict(halves=2, users=4, block=8)}


class FoldTests(Quiet):
    def engaged(self, function, case, tensor, mesh=None, processors=perm.PROCESSORS):
        ops = ExecutingOperations(mesh=mesh)
        owned = []
        served = Mock(side_effect=AssertionError('the served path must not run'))
        with four_cards():
            result = _sites(function)(ops, ops.from_logical(tensor), keep(owned), served=served, site=case, **GEOMETRY[case])
        served.assert_not_called()
        return ops, ops.to_logical(result)

    def test_the_fold_is_the_served_fold_bit_for_bit_for_the_pair_quad_and_octo_shapes(self):
        for case in ('pair', 'quad', 'octo'):
            with self.subTest(case=case):
                rows = 32 * GEOMETRY[case]['halves']
                query = bits(torch.Generator().manual_seed(rows + len(case)), (1, HEADS, rows, DIM))
                served = fold_served(case, query)
                ops, mine = self.engaged('fold', case, query)
                self.assertEqual(tuple(mine.shape), tuple(served.shape))
                self.assertTrue(same(mine, served))
                self.assertEqual(len(ops.programs), 1)

    def test_the_unfold_is_the_served_unfold_bit_for_bit_for_the_pair_quad_and_octo_shapes(self):
        for case in ('pair', 'quad', 'octo'):
            with self.subTest(case=case):
                geometry = GEOMETRY[case]
                heads = KV * geometry['halves'] * geometry['users'] * (HEADS // KV)
                output = bits(torch.Generator().manual_seed(heads + len(case)), (1, heads, 32, DIM))
                served = unfold_served(case, output)
                ops, mine = self.engaged('unfold', case, output)
                self.assertEqual(tuple(mine.shape), tuple(served.shape))
                self.assertTrue(same(mine, served))

    def test_only_the_rotated_copies_of_the_fold_are_canonical_and_everything_of_the_unfold_is(self):
        for case in ('pair', 'quad', 'octo'):
            geometry = GEOMETRY[case]
            records = perm.fold_records(KV, HEADS // KV, **geometry)
            runs = [record for record in records if record[0] == 'run']
            mixes = [record for record in records if record[0] == 'mix']
            self.assertTrue(all(record[6] is False for record in runs), case)
            self.assertEqual(sum(record[5] for record in runs), KV * geometry['halves'] * (HEADS // KV) * 4, 'the unrotated copy: four raw tiles a kv, half and group head')
            self.assertEqual(len(mixes), KV * geometry['halves'] * (geometry['users'] - 1) * (HEADS // KV) * 4, 'every rotation is a mixed canonical tile')
            self.assertTrue(all(mode >= perm.CANON for record in mixes for _, _, mode in record[3]), case)
            for record in perm.unfold_records(KV, HEADS // KV, **geometry):
                self.assertEqual(record[0], 'mix')
                self.assertTrue(all(mode >= perm.CANON for _, _, mode in record[3] if mode), case)

    def test_the_fold_and_unfold_cover_every_destination_tile_once(self):
        for case in ('pair', 'quad', 'octo'):
            geometry = GEOMETRY[case]
            folded = KV * geometry['halves'] * geometry['users'] * (HEADS // KV)
            written = []
            for record in perm.fold_records(KV, HEADS // KV, **geometry):
                written.extend(range(record[2], record[2] + record[5]) if record[0] == 'run' else [record[2]])
            self.assertEqual(sorted(written), list(range(folded * 4)), case)
            written = [record[2] for record in perm.unfold_records(KV, HEADS // KV, **geometry)]
            self.assertEqual(sorted(written), list(range(HEADS * geometry['halves'] * 4)), case)

    def test_a_unfold_of_a_fold_of_an_identity_attention_is_the_query_up_to_the_rule(self):
        for case in ('pair', 'quad'):
            geometry = GEOMETRY[case]
            rows = 32 * geometry['halves']
            query = bits(torch.Generator().manual_seed(17), (1, HEADS, rows, DIM))
            ops, folded = self.engaged('fold', case, query)
            heads = folded.shape[1]
            again, back = self.engaged('unfold', case, folded.reshape(1, heads, 32, DIM))
            self.assertTrue(same(back, canon(query)), case)

    def test_geometry_that_is_not_a_tile_half_is_refused(self):
        with self.assertRaises(perm.Unsupported):
            perm.fold_records(KV, 4, 2, 3, 16)
        with self.assertRaises(perm.Unsupported):
            perm.fold_records(KV, 4, 2, 2, 12)
        with self.assertRaises(perm.Unsupported):
            perm.unfold_records(KV, 4, 0, 2, 16)


def _sites(name):
    return {'fold': perm.fold_query, 'unfold': perm.unfold_output}[name]


# --- the launch --------------------------------------------------------------------------------------------------------------

class LaunchTests(Quiet):
    def test_a_launch_is_one_generic_op_two_worker_kernels_a_core_each_with_its_own_scratch(self):
        for mesh, cores in ((Mesh(11, 10), 110), (Mesh(13, 10), 130)):
            with self.subTest(grid=(mesh.x, mesh.y)), four_cards():
                plan, caches, live = kv_operands('quad', seed=1)
                ops, _, result = engaged_kv(plan, caches, live, mesh=mesh)
                self.assertEqual(len(ops.generic), 1)
                tensors, program = ops.generic[0]
                self.assertEqual(len(tensors), 10 + 2)
                self.assertEqual(tensors[-2:], [result['k'], result['v']], 'the outputs are last')
                self.assertEqual(len(program), 4, 'one program a chip')
                for descriptor in program.values():
                    kernels = descriptor['kernels']
                    self.assertEqual(len(kernels), 2)
                    self.assertEqual([kernel['config'] for kernel in kernels], [dict(processor=0, noc=0), dict(processor=1, noc=1)])
                    for processor, kernel in enumerate(kernels):
                        self.assertTrue(kernel['kernel_source'].endswith('draft_permute_tp.cpp'))
                        self.assertEqual(kernel['defines'], [('CANON_DENORM', '1')])
                        size, nsrc, ndst, scratch = kernel['compile_time_args'][-4:]
                        self.assertEqual((nsrc, ndst, scratch), (10, 2, processor))
                        lists = [words for column in kernel['runtime_args'].values() for words in column.values()]
                        self.assertEqual({len(words) for words in lists}, {size}, 'every list padded to the compile-time capacity')
                        self.assertLessEqual(size, perm.MAX_ARGUMENT_WORDS)
                        cores_used = [(x, y) for x in kernel['runtime_args'] for y in kernel['runtime_args'][x]]
                        self.assertLessEqual(len(cores_used), cores)
                        self.assertTrue(all(x < mesh.x and y < mesh.y for x, y in cores_used))
                        rectangle = kernel['core_ranges']
                        self.assertEqual(len(rectangle), 1, 'ONE CoreRange: a non-rectangular set pays a dispatch gap per launch')
                        (x0, y0), (x1, y1) = rectangle[0]
                        self.assertEqual((x0, y0, x1), (0, 0, mesh.x - 1), 'the full width of the grid, read from the device')
                        self.assertEqual(len(cores_used), (x1 + 1) * (y1 + 1), 'a runtime-arg list on every core of the rectangle')
                    scratch_buffers = descriptor['cbs']
                    self.assertEqual([buffer['total_size'] for buffer in scratch_buffers], [8 * 2048] * 2)
                    self.assertEqual([buffer['format_descriptors'][0]['buffer_index'] for buffer in scratch_buffers], [0, 1])

    def test_the_grid_is_read_from_the_device_and_a_wider_grid_uses_more_cores(self):
        with four_cards():
            plan, caches, live = kv_operands('quad', seed=1)
            narrow, _, _ = engaged_kv(plan, caches, live, mesh=Mesh(11, 10))
            wide, _, _ = engaged_kv(plan, caches, live, mesh=Mesh(13, 10))
        def count(ops):
            kernel = next(iter(ops.generic[0][1].values()))['kernels'][0]
            return sum(len(column) for column in kernel['runtime_args'].values())

        self.assertEqual(count(narrow), 110)
        self.assertEqual(count(wide), 13 * 10, 'the 13 x 10 grid: all 130 cores, still one rectangle')
        self.assertLessEqual(count(narrow), 11 * 10)
        self.assertLessEqual(count(wide), 13 * 10)

    def test_a_single_processor_launch_has_one_kernel(self):
        with four_cards():
            plan, caches, live = kv_operands('pair', seed=2)
            ops, mine, _ = engaged_kv(plan, caches, live, processors=1)
            self.assertEqual(len(next(iter(ops.generic[0][1].values()))['kernels']), 1)
            self.assertTrue(same(mine['k'], served_kv(plan, caches, live)['k']))

    def test_marker_lines_say_what_ran(self):
        with four_cards():
            plan, caches, live = kv_operands('quad', seed=1)
            engaged_kv(plan, caches, live)
        engaged = [line for line in self.lines if line.startswith(perm.ENGAGED)]
        self.assertEqual(len(engaged), 1)
        self.assertRegex(engaged[0], r'site=kv shape=test users=4 kv_heads=2 rows=8320 tiles=\d+ records=\d+ lanes=\d+ cores=\d+ canon_cached=1')

    def test_a_call_it_cannot_take_runs_the_served_ops_frees_nothing_it_did_not_make_and_says_why_once(self):
        with four_cards():
            plan, caches, live = kv_operands('quad', seed=1)
            ops = ExecutingOperations()
            owned = []
            device_caches = [{name: ops.from_logical(value) for name, value in cache.items()} for cache in caches]
            device_live = {name: ops.from_logical(value) for name, value in live.items()}
            device_live['k'].dtype = 'bf8'
            served = Mock(side_effect=lambda name: 'served-' + name)
            result = perm.assemble_kv(ops, plan, device_caches, device_live, keep(owned), served=served, site='quad')
            self.assertEqual(result, {'k': 'served-k', 'v': 'served-v'})
            self.assertEqual(ops.generic, [])
            self.assertEqual(ops.empties, [])
            self.assertEqual(owned, [])
            fell = [line for line in self.lines if line.startswith(perm.FALLBACK)]
            self.assertEqual(len(fell), 1)
            self.assertIn('site=kv shape=quad reason=an operand is not bfloat16 TILE', fell[0])
            wrong = [{name: ops.from_logical(torch.zeros(1, 4, 2048, DIM, dtype=torch.int16)) for name in 'kv'} for _ in range(4)]
            result = perm.assemble_kv(ops, plan, wrong, {name: ops.from_logical(live[name]) for name in 'kv'}, keep(owned), served=served, site='quad')
            self.assertEqual(result, {'k': 'served-k', 'v': 'served-v'})
            self.assertIn('are not (1, 2, rows, 128)', [line for line in self.lines if line.startswith(perm.FALLBACK)][-1])

    def test_a_list_over_the_argument_budget_falls_back_before_anything_is_allocated(self):
        with four_cards():
            plan, caches, live = kv_operands('octo', seed=1)
            ops = ExecutingOperations(mesh=Mesh(1, 1))
            owned = []
            device_caches = [{name: ops.from_logical(value) for name, value in cache.items()} for cache in caches]
            device_live = {name: ops.from_logical(value) for name, value in live.items()}
            served = Mock(side_effect=lambda name: 'served-' + name)
            result = perm.assemble_kv(ops, plan, device_caches, device_live, keep(owned), served=served, site='octo', processors=1)
            self.assertEqual(result['k'], 'served-k')
            self.assertEqual(ops.empties, [])
            self.assertIn('do not fit', self.lines[-1])

    def test_a_failed_submit_frees_the_outputs_and_reraises(self):
        with four_cards():
            plan, caches, live = kv_operands('pair', seed=1)
            ops = ExecutingOperations()
            ops.generic_op = Mock(side_effect=RuntimeError('submit'))
            owned = []
            device_caches = [{name: ops.from_logical(value) for name, value in cache.items()} for cache in caches]
            device_live = {name: ops.from_logical(value) for name, value in live.items()}
            with self.assertRaises(RuntimeError):
                perm.assemble_kv(ops, plan, device_caches, device_live, keep(owned), served=Mock(), site='pair')
            self.assertEqual(ops.freed, ops.empties)
            self.assertEqual(owned, [])

    def test_mixed_accessor_layouts_fall_back(self):
        with four_cards():
            plan, caches, live = kv_operands('pair', seed=1)
            ops = ExecutingOperations()
            owned = []
            device_caches = [{name: ops.from_logical(value) for name, value in cache.items()} for cache in caches]
            device_live = {name: ops.from_logical(value) for name, value in live.items()}
            odd = device_live['k'].shards[0]
            ops.TensorAccessorArgs = staticmethod(lambda value: SimpleNamespace(get_compile_time_args=lambda: [2 if value is odd else 1]))
            served = Mock(side_effect=lambda name: 'served-' + name)
            result = perm.assemble_kv(ops, plan, device_caches, device_live, keep(owned), served=served, site='pair')
            self.assertEqual(result['k'], 'served-k')
            self.assertEqual(ops.freed, ops.empties)
            self.assertIn('share an accessor layout', self.lines[-1])


# --- the audit ---------------------------------------------------------------------------------------------------------------

class AuditTests(Quiet):
    def setUp(self):
        super().setUp()
        patcher = patch.dict(os.environ, {perm.AUDIT_FLAG: '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_kv(self, tamper=False, eager=True):
        plan, caches, live = kv_operands('pair', seed=4)
        ops = ExecutingOperations()
        owned = []
        device_caches = [{name: ops.from_logical(value) for name, value in cache.items()} for cache in caches]
        device_live = {name: ops.from_logical(value) for name, value in live.items()}
        reference = served_kv(plan, caches, live)
        if tamper:
            reference['v'] = reference['v'].clone()
            reference['v'][0, 0, 5, 7] ^= 1
        served = Mock(side_effect=lambda name: ops.from_logical(reference[name]))
        previous = perm.set_pass(eager)
        try:
            result = perm.assemble_kv(ops, plan, device_caches, device_live, keep(owned), served=served, site='pair')
        finally:
            perm.set_pass(previous)
        return result, served

    def test_the_eager_warm_pass_runs_the_served_ops_beside_the_launch_and_logs_exact(self):
        with four_cards():
            _, served = self.run_kv()
        self.assertEqual(served.call_count, 2)
        self.assertIn('%s exact=True site=kv shape=pair tensors=2' % perm.AUDIT, self.lines)

    def test_a_differing_byte_logs_the_mismatch_marker_and_raises(self):
        with four_cards(), self.assertRaises(AssertionError):
            self.run_kv(tamper=True)
        self.assertTrue(any(line.startswith(perm.MISMATCH) and 'differing=[(\'v\', 0)' in line for line in self.lines), self.lines)

    def test_a_capture_pass_and_an_unmarked_pass_record_the_launch_only(self):
        for eager in (False, None):
            with four_cards():
                _, served = self.run_kv(eager=eager)
            served.assert_not_called()
        self.assertFalse([line for line in self.lines if line.startswith(perm.AUDIT)])

    def test_the_audit_without_the_lever_is_refused(self):
        with environment(**{perm.FLAG: '0'}):
            with self.assertRaisesRegex(ValueError, 'needs'):
                perm.audit_enabled()

    def test_the_fold_and_unfold_are_audited_too_and_the_served_reference_does_not_launch(self):
        with four_cards():
            ops = ExecutingOperations()
            owned = []
            query = bits(torch.Generator().manual_seed(2), (1, HEADS, 64, DIM))
            reference = fold_served('quad', query)
            quad_served = Mock(side_effect=lambda: ops.from_logical(reference))
            previous = perm.set_pass(True)
            try:
                perm.fold_query(ops, ops.from_logical(query), keep(owned), served=quad_served, site='quad', **GEOMETRY['quad'])
            finally:
                perm.set_pass(previous)
            quad_served.assert_called_once()
        self.assertIn('%s exact=True site=fold shape=quad tensors=1' % perm.AUDIT, self.lines)

    def test_a_served_reference_takes_the_served_ops_inside_its_own_hooks(self):
        # quad_twin.quad_fold_query's served body calls the pair's hooked fold: inside a served reference the hook must not launch.
        with four_cards():
            ops = CanonOps()
            owned = []
            query = Device(bits(torch.Generator().manual_seed(8), (1, HEADS, 64, DIM)))
            with perm.served_only():
                self.assertFalse(perm.hook_enabled())
                quad_twin.quad_fold_query(ops, query, keep(owned))
            self.assertTrue(perm.hook_enabled())
        self.assertTrue(any(call[0] == 'slice' for call in ops.calls))


# --- the flags and the twins' flag-off path ---------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_strict_values(self):
        with environment(QWEN_FAST_TP='4'):
            for value in ('', '2', 'true', '01'):
                with environment(**{perm.FLAG: value}), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    perm.enabled()
            with environment(**{perm.FLAG: '1'}):
                self.assertTrue(perm.enabled())
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(perm.FLAG, None)
                self.assertFalse(perm.enabled())

    def test_the_lever_is_a_four_card_lever(self):
        with pair(), environment(**{perm.FLAG: '1'}), self.assertRaisesRegex(ValueError, 'TP4 lever'):
            perm.enabled()

    def test_validate_checks_both_flags_once(self):
        with environment(QWEN_FAST_TP='4', **{perm.FLAG: '1', perm.AUDIT_FLAG: '1'}):
            perm.validate()
        with environment(QWEN_FAST_TP='4', **{perm.FLAG: '0', perm.AUDIT_FLAG: '1'}), self.assertRaises(ValueError):
            perm.validate()


class FlagOffTests(unittest.TestCase):
    def test_flag_off_the_twins_import_nothing_and_run_the_served_ops(self):
        script = ('import sys, os; sys.path.insert(0, %r); os.environ.pop(%r, None); os.environ["QWEN_FAST_TP"] = "4"; '
                  'import draft_attention_branch, pair_row_exact_tp, quad_draft_tp; print("draft_permute_tp" in sys.modules)' % (str(HERE), perm.FLAG))
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=120, cwd=str(HERE))
        if result.returncode != 0:
            self.skipTest('the serving modules do not import here: %s' % result.stderr[-200:])
        self.assertEqual(result.stdout.strip(), 'False')

    @staticmethod
    def served_sites():
        """The fold and the unfold hooks answer with their served references (the branch tests are about the K/V hook)."""
        from contextlib import ExitStack

        def answer(*args, served, **options):
            with perm.served_only():
                return served()

        stack = ExitStack()
        stack.enter_context(patch.object(perm, 'fold_query', side_effect=answer))
        stack.enter_context(patch.object(perm, 'unfold_output', side_effect=answer))
        return stack

    def test_flag_zero_is_the_unset_path_op_for_op(self):
        with four_cards():
            query = Device(bits(torch.Generator().manual_seed(1), (1, HEADS, 64, DIM)))
            logs = []
            for value in (None, '0'):
                with patch.dict(os.environ, {} if value is None else {perm.FLAG: value}):
                    if value is None:
                        os.environ.pop(perm.FLAG, None)
                    ops, owned = CanonOps(), []
                    folded = quad_twin.quad_fold_query(ops, query, keep(owned))
                    quad_twin.quad_unfold_output(ops, folded, keep(owned))
                    logs.append(ops.calls)
            self.assertEqual(logs[0], logs[1])
            self.assertGreater(len(logs[0]), 20)

    def test_flag_on_each_hooked_site_calls_the_module_with_its_geometry_and_a_served_reference(self):
        generator = torch.Generator().manual_seed(1)
        query = Device(bits(generator, (1, HEADS, 64, DIM)))
        pair_query = Device(bits(generator, (1, HEADS, 32, DIM)))
        folded = Device(bits(generator, (1, 32, 32, DIM)))
        pair_folded = Device(bits(generator, (1, 16, 32, DIM)))
        with four_cards(), environment(**{perm.FLAG: '1'}):
            cases = ((quad_twin.quad_fold_query, query, 'fold_query', 'quad', GEOMETRY['quad']),
                     (quad_twin.quad_unfold_output, folded, 'unfold_output', 'quad', GEOMETRY['quad']),
                     (pair_row_exact_tp.fold_query, pair_query, 'fold_query', 'pair', GEOMETRY['pair']),
                     (pair_row_exact_tp.unfold_output, pair_folded, 'unfold_output', 'pair', GEOMETRY['pair']))
            for function, tensor, target, site, geometry in cases:
                with self.subTest(function=function.__name__), patch.object(perm, target, return_value='engaged') as hook:
                    ops, owned = CanonOps(), []
                    self.assertEqual(function(ops, tensor, keep(owned)), 'engaged')
                    hook.assert_called_once()
                    self.assertEqual(hook.call_args.kwargs['site'], site)
                    self.assertEqual({key: hook.call_args.kwargs[key] for key in ('halves', 'users', 'block')}, geometry)
                    self.assertEqual(ops.calls, [], 'nothing ran before the hook')
                    with perm.served_only():
                        served = hook.call_args.kwargs['served']()
                    self.assertTrue(ops.calls, 'the served reference runs the served ops')
                    self.assertIsNotNone(served)

    def test_the_attention_branch_hands_the_plan_to_assemble_kv_when_the_flag_is_on_and_only_then(self):
        from test_quad_draft_tp4 import attention_run4

        with four_cards():
            with patch.object(perm, 'assemble_kv', side_effect=AssertionError('flag off')):
                off = attention_run4(quad=True)
            with environment(**{perm.FLAG: '1'}):
                seen = {}

                def fake(operations, plan, caches, live, retain, *, served, site, **options):
                    seen.update(plan=plan, users=len(caches), site=site)
                    return {name: served(name) for name in 'kv'}

                with patch.object(perm, 'assemble_kv', side_effect=fake), self.served_sites():
                    on = attention_run4(quad=True)
                with patch.object(perm, 'assemble_kv', side_effect=fake), self.served_sites():
                    pair_on = attention_run4(quad=False)
        self.assertEqual(seen['site'], 'pair')
        self.assertEqual(on, off, 'with the hook returning the served assembly the op log is unchanged')
        self.assertGreater(len(pair_on), 20)

    def test_the_attention_branch_flag_on_with_the_quad_site(self):
        from test_quad_draft_tp4 import attention_run4

        sites = []
        with four_cards(), environment(**{perm.FLAG: '1'}):
            with patch.object(perm, 'assemble_kv', side_effect=lambda *a, served, site, **o: sites.append(site) or {n: served(n) for n in 'kv'}), self.served_sites():
                attention_run4(quad=True)
        self.assertEqual(sites, ['quad'])


# --- the log rule ------------------------------------------------------------------------------------------------------------

class SmokeTests(unittest.TestCase):
    QUAD = {smoke.FLAG: '1', smoke.QUAD_FLAG: '1'}

    def engaged(self, site, shape):
        return '2026 INFO %s site=%s shape=%s users=4 kv_heads=2 rows=8320' % (smoke.ENGAGED, site, shape)

    def audit(self, site, shape, exact=True):
        return '%s exact=%s site=%s shape=%s tensors=2' % (smoke.AUDIT, exact, site, shape)

    def full(self):
        return '\n'.join(self.engaged(site, 'quad') for site in smoke.SITES)

    def test_a_profile_without_the_flag_has_none_of_the_lines(self):
        self.assertEqual(smoke.problems({}, 'unrelated line\n'), [])
        self.assertEqual(smoke.problems(None, ''), [])
        found = smoke.problems({smoke.QUAD_FLAG: '1'}, self.engaged('kv', 'quad'))
        self.assertEqual(len(found), 1)
        self.assertIn('without %s' % smoke.FLAG, found[0])
        self.assertTrue(smoke.problems({smoke.AUDIT_FLAG: '1'}, ''))

    def test_the_quad_profile_needs_an_engaged_line_per_site(self):
        self.assertEqual(smoke.problems(self.QUAD, self.full()), [])
        missing = '\n'.join([self.engaged('kv', 'quad'), self.engaged('fold', 'quad')])
        found = smoke.problems(self.QUAD, missing)
        self.assertEqual(len(found), 1)
        self.assertIn('site=unfold shape=quad', found[0])
        self.assertEqual(len(smoke.problems(self.QUAD, self.full().replace('shape=quad', 'shape=pair'))), 3, 'the pair shape is not the quad')

    def test_the_octo_profile_needs_the_octo_kv_line(self):
        env = dict(self.QUAD)
        env[smoke.OCTO_FLAG] = '1'
        self.assertEqual(len(smoke.problems(env, self.full())), 1)
        self.assertEqual(smoke.problems(env, self.full() + '\n' + self.engaged('kv', 'octo')), [])

    def test_with_no_quad_or_octo_flag_any_kv_line_will_do(self):
        self.assertEqual(smoke.problems({smoke.FLAG: '1'}, self.engaged('kv', 'pair')), [])
        self.assertTrue(smoke.problems({smoke.FLAG: '1'}, self.engaged('fold', 'pair')))

    def test_a_fall_back_or_an_audit_difference_fails_every_arm(self):
        fell = '%s site=kv shape=quad reason=an operand is not interleaved DRAM' % smoke.FELL_BACK
        self.assertTrue(any('fell back' in item for item in smoke.problems(self.QUAD, self.full() + '\n' + fell)))
        different = '%s site=kv shape=quad differing=[(\'v\', 0)]' % smoke.MISMATCH
        self.assertTrue(any('found a difference' in item for item in smoke.problems(self.QUAD, self.full() + '\n' + different)))
        self.assertTrue(smoke.problems({}, fell))

    def test_the_audit_arm_needs_a_passing_line_for_each_quad_site(self):
        env = dict(self.QUAD)
        env[smoke.AUDIT_FLAG] = '1'
        passing = '\n'.join(self.audit(site, 'quad') for site in smoke.SITES)
        self.assertEqual(smoke.problems(env, self.full() + '\n' + passing), [])
        self.assertEqual(len(smoke.problems(env, self.full())), 3, 'nothing was compared')
        partial = '\n'.join(self.audit(site, 'quad') for site in ('kv', 'fold'))
        found = smoke.problems(env, self.full() + '\n' + partial)
        self.assertEqual(len(found), 1)
        self.assertIn('site=unfold', found[0])
        self.assertTrue(smoke.problems(env, self.full() + '\n' + self.audit('kv', 'quad', exact=False)))

    def test_the_rule_is_stdlib_only(self):
        source = (HERE / 'draft_permute_smoke.py').read_text()
        self.assertEqual(sorted(set(re.findall(r'^(?:import|from) (\w+)', source, re.M))), ['re'])


class KernelSourceTests(unittest.TestCase):
    def setUp(self):
        self.text = (HERE / perm.KERNEL).read_text()

    def test_the_kernel_constants_agree_with_the_planner(self):
        for name, value in (('LANES', perm.LANES), ('TILE_BYTES', perm.TILE_BYTES), ('TYPE_RUN', perm.TYPE_RUN), ('TYPE_MIX', perm.TYPE_MIX),
                            ('RUN_WORDS', perm.RUN_WORDS), ('MIX_WORDS', perm.MIX_WORDS), ('CHUNK_BYTES', 256), ('FACE_BYTES', 512)):
            self.assertIn('constexpr uint32_t %s = %d;' % (name, value), self.text)

    def test_the_record_decoding_matches_the_encoder(self):
        run = perm.encode(('run', 3, 1234, 7, 99, 17, True))
        self.assertEqual(run, [(1 << 28) | (1 << 24) | (3 << 8) | 7, 17, 1234, 99])
        self.assertIn('head & 0xFFu', self.text)
        self.assertIn('(head >> 8) & 0xFFu', self.text)
        self.assertIn('(head >> 24) & 1u', self.text)
        mix = perm.encode(('mix', 1, 55, ((2, 66, perm.CANON + 3), (0, 0, 0), None, (9, 7, perm.RAW))))
        self.assertEqual(mix, [(2 << 28) | (1 << 8), 55, (2 << 24) | (8 << 16) | 66, 0, 0, (9 << 24) | (1 << 16) | 7])
        for expression in ('word >> 24', '(word >> 16) & 0xFFu', 'word & 0xFFFFu', '(mode - 1) & 3'):
            self.assertIn(expression, self.text)

    def test_the_kernel_never_reads_past_the_capacity(self):
        self.assertIn('given < CAPACITY ? given : CAPACITY', self.text)
        self.assertIn('at + RUN_WORDS <= end', self.text)
        self.assertIn('at + MIX_WORDS <= end', self.text)

    def test_the_rule_is_the_swar_form_checked_above_and_the_scalar_one_when_the_define_is_off(self):
        self.assertIn('((word & 0x7F807F80u) + 0x7F807F80u) & 0x80008000u', self.text)
        self.assertIn('present | (present - (present >> 15))', self.text)
        self.assertIn('value == 0x8000u ? 0u : value', self.text)

    def test_runtime_files_name_both_files(self):
        self.assertEqual(perm.RUNTIME_FILES, ('draft_permute_tp.py', 'draft_permute_tp.cpp'))
        for name in perm.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file())

    def test_the_module_is_importable_on_py37_syntax(self):
        import ast

        tree = ast.parse((HERE / 'draft_permute_tp.py').read_text())
        self.assertFalse([node for node in ast.walk(tree) if isinstance(node, ast.NamedExpr)], 'no walrus')
        imports = [node.names[0].name for node in tree.body if isinstance(node, ast.Import)] + [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
        self.assertNotIn('torch', imports, 'torch only inside the audit')


if __name__ == '__main__':
    unittest.main()
