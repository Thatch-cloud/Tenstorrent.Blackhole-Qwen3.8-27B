"""tp4_rs_tile_spike on the CPU, against a numeric fake of the four-chip ring all-reduce.

The fake adds the four partials in real bfloat16 arithmetic, forward (chip 0 first) for even chunks and backward for odd
ones, the chunk taken from the tile's flat index in the per-chip slice of 5,120 columns (40 tiles a row, 8-tile chunks)
- reduce_scatter_common::chunk_ring_parity - so the harness's reproduction, its fix check, its controls and its verdict
are exercised on numbers that really differ in the last bit."""

import json
import os
import sys
import tempfile
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tile_collective_tp
import tp4_rs_tile_spike as spike

CHIPS = 4
SLICE_TILES = 40


class Device:
    def __init__(self, parts):
        self.parts = parts
        self.freed = False

    @property
    def shape(self):
        return tuple(self.parts[0].shape)

    def memory_config(self):
        return 'DRAM'


class Mesh:
    def get_device_ids(self):
        return [0, 1, 3, 2]

    def get_num_devices(self):
        return CHIPS


def fake_ttnn():
    def from_torch(source, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        return Device([piece.clone() for piece in source.chunk(CHIPS, dim=0)])

    def slice_(tensor, start, stop, memory_config=None):
        assert not tensor.freed
        cut = tuple(builtin_slice(a, b) for a, b in zip(start, stop))
        return Device([part[cut].clone() for part in tensor.parts])

    def concat(tensors, dim, memory_config=None):
        return Device([torch.cat([tensor.parts[chip] for tensor in tensors], dim=dim) for chip in range(CHIPS)])

    def deallocate(tensor):
        tensor.freed = True

    def reshape(tensor, shape):
        return Device([part.reshape(shape) for part in tensor.parts])

    builtin_slice = slice
    return types.SimpleNamespace(
        bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='DRAM',
        Topology=types.SimpleNamespace(Ring='Ring', Linear='Linear'),
        FabricConfig=types.SimpleNamespace(FABRIC_1D='FABRIC_1D', FABRIC_1D_RING='FABRIC_1D_RING'),
        MeshShape=lambda *shape: shape, set_fabric_config=lambda config: None,
        open_mesh_device=lambda shape, **kwargs: Mesh(), close_mesh_device=lambda mesh: None,
        ShardTensorToMesh=lambda mesh, dim: (mesh, dim), from_torch=from_torch,
        to_torch=lambda part: part, get_device_tensors=lambda tensor: list(tensor.parts),
        slice=slice_, concat=concat, deallocate=deallocate, reshape=reshape)


def ring_all_reduce(chunk_tiles=8, nondeterministic=False, whole_call_order=False):
    """tt_all_reduce as far as the order of its four-way sums goes: reduce-scatter then gather, so every chip ends with
    the full reduced tensor. `chunk_tiles` 10**6 makes every chunk even (no dependence on the tile); `whole_call_order`
    makes the order depend on the call's own height instead (a fake that breaks the one-tile invariance)."""
    calls = []

    def call(tensor, mesh, ccl, cluster_axis=0, dim=3, topology='Ring', memory_config='DRAM'):
        assert not tensor.freed and dim == 3
        calls.append((tensor.shape, topology))
        partials = tensor.parts                       # four (1, units, rows, width) partials
        units, rows, width = partials[0].shape[1:]
        forward = ((partials[0] + partials[1]) + partials[2]) + partials[3]
        backward = ((partials[3] + partials[2]) + partials[1]) + partials[0]
        parity = torch.zeros(units, rows, width, dtype=torch.bool)
        if topology == 'Ring':
            for row_tile in range(-(-rows // 32)):     # a short tensor is tile padded: four rows are one tile
                for column_tile in range(width // 32):
                    chunk = (row_tile * SLICE_TILES + column_tile % SLICE_TILES) // chunk_tiles
                    if whole_call_order:
                        chunk = 1 if rows == 32 else 0
                    parity[:, row_tile * 32:(row_tile + 1) * 32, column_tile * 32:(column_tile + 1) * 32] = bool(chunk % 2)
        reduced = torch.where(parity.unsqueeze(0), backward, forward)
        if nondeterministic:
            reduced = reduced + torch.rand(1).to(torch.bfloat16) * 1e-3
        tensor.freed = True
        return Device([reduced.clone() for _ in range(CHIPS)])

    call.calls = calls
    return call


def run_spike(all_reduce, heights='64', seeds='1', ttnn=None):
    ttnn = ttnn or fake_ttnn()
    modules = dict(torch=torch, ttnn=ttnn, TT_CCL=lambda mesh: 'ccl', tt_all_reduce=all_reduce,
                   get_num_links=lambda mesh: 2)
    lines = []
    with tempfile.TemporaryDirectory() as directory:
        output = os.path.join(directory, 'spike.json')
        status = spike.main(['--output', output, '--heights', heights, '--seeds', seeds],
                            runner=lambda options, log: spike.run(options, log, modules=modules), log=lines.append)
        with open(output) as handle:
            report = json.load(handle)
    return status, report, lines


class Reproduction(unittest.TestCase):
    def test_the_ring_order_is_reproduced_and_the_tile_split_is_exact(self):
        status, report, lines = run_spike(ring_all_reduce())
        self.assertEqual(status, 0, [line for line in lines if 'reason' in line or 'ERROR' in line])
        self.assertEqual(report['verdict'], 'PASS')
        rows = {row['name']: row for row in report['records']}
        self.assertEqual(rows['r64/s0/block-tile0-vs-one-tile']['differing'], 0)
        self.assertGreater(rows['r64/s0/block-tile1-vs-one-tile']['differing'], 0)
        self.assertEqual(rows['r64/s0/split-tile0-vs-one-tile']['differing'], 0)
        self.assertEqual(rows['r64/s0/split-tile1-vs-one-tile']['differing'], 0)
        self.assertEqual(rows['r64/s0/wrapper-splits']['splits'], 1)

    def test_the_verdict_line_carries_the_comparison_and_differing_counts(self):
        status, report, lines = run_spike(ring_all_reduce())
        line = [text for text in lines if text.startswith('TP4_RS_TILE verdict=')][0]
        self.assertIn('verdict=PASS', line)
        self.assertIn('unfixed_unequal=1/2', line)
        self.assertIn('fixed_unequal=0/2', line)
        self.assertIn('fixed_differing=0', line)
        self.assertIn('heights=64 seeds=1 topology=Ring links=2', line)
        self.assertEqual(report['verdict_line'], line)
        self.assertGreater(report['summary']['unfixed']['differing'], 0)

    def test_controls_hold_on_the_fake(self):
        status, report, lines = run_spike(ring_all_reduce())
        controls = [row for row in report['records'] if row['group'] == 'control']
        self.assertTrue(controls)
        self.assertEqual([row['name'] for row in controls if row['differing']], [])
        names = [row['name'] for row in controls]
        for expected in ('r64/s0/one-tile-twice', 'r64/s0/four-rows-vs-tile0', 'r64/s0/block/chip3-vs-chip0',
                         'r64/s0/linear-block-tile1-vs-one-tile'):
            self.assertIn(expected, names)

    def test_unit_major_is_informational_and_equal_on_the_fake(self):
        status, report, lines = run_spike(ring_all_reduce())
        info = [row for row in report['records'] if row['group'] == 'informational']
        self.assertEqual([row['differing'] for row in info], [0, 0])

    def test_two_heights_and_seeds_run_every_combination(self):
        status, report, lines = run_spike(ring_all_reduce(), heights='64,128', seeds='2')
        self.assertEqual(status, 0)
        unfixed = [row['name'] for row in report['records'] if row['group'] == 'unfixed']
        self.assertEqual(len(unfixed), 2 * (2 + 4))
        rows = {row['name']: row for row in report['records']}
        # 128 rows: tile 2 starts at flat tile 80, chunk 10, an even chunk: the same order as tile 0, so equal
        self.assertEqual(rows['r128/s0/block-tile2-vs-one-tile']['differing'], 0)
        self.assertGreater(rows['r128/s0/block-tile3-vs-one-tile']['differing'], 0)
        fixed = [row for row in report['records'] if row['group'] == 'fixed']
        self.assertEqual(len(fixed), 2 * (2 + 4))
        self.assertEqual(sum(row['differing'] for row in fixed), 0)

    def test_the_block_is_reduced_whole_once_and_the_wrapper_only_ever_issues_one_tile_calls(self):
        function = ring_all_reduce()
        run_spike(function)
        ring = [shape for shape, topology in function.calls if topology == 'Ring']
        # the unfixed whole block once, the unit-major reshape once; the fix's calls are one tile each
        self.assertEqual(ring.count((1, 1, 64, 5120)), 1)
        self.assertEqual(ring.count((1, 2, 32, 5120)), 1)
        self.assertGreaterEqual(ring.count((1, 1, 32, 5120)), 4)


class Verdicts(unittest.TestCase):
    def test_no_order_dependence_is_not_reproduced(self):
        status, report, lines = run_spike(ring_all_reduce(chunk_tiles=10 ** 6))
        self.assertEqual(status, 2)
        self.assertEqual(report['verdict'], 'NOT_REPRODUCED')
        self.assertEqual(report['summary']['unfixed']['differing'], 0)
        self.assertEqual(report['summary']['fixed']['differing'], 0)

    def test_a_baseline_that_is_not_repeatable_fails(self):
        status, report, lines = run_spike(ring_all_reduce(nondeterministic=True))
        self.assertEqual(status, 1)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertTrue(any('control' in reason for reason in report['reasons']))

    def test_a_first_tile_that_differs_fails(self):
        status, report, lines = run_spike(ring_all_reduce(whole_call_order=True))
        self.assertEqual(status, 1)
        self.assertTrue(any('first tile' in reason for reason in report['reasons']))

    def test_a_mesh_that_did_not_open_fails(self):
        ttnn = fake_ttnn()

        def refuse(shape, **kwargs):
            raise RuntimeError('No core coordinate found at (1, 2)')
        ttnn.open_mesh_device = refuse
        status, report, lines = run_spike(ring_all_reduce(), ttnn=ttnn)
        self.assertEqual(status, 1)
        self.assertFalse(report['opened'])
        self.assertIn('49701', report['known_failure'])

    def test_a_split_that_engaged_the_wrong_number_of_calls_fails(self):
        report = dict(opened=True, records=[
            spike.record('unfixed', 't0', 10, 0, None, tile=0), spike.record('unfixed', 't1', 10, 4, 3, tile=1),
            spike.record('fixed', 'f0', 10, 0, None, tile=0),
            dict(group='control', name='splits', elements=1, differing=0, first=None, splits=0, splits_expected=1)])
        outcome, reasons = spike.verdict(report)
        self.assertEqual(outcome, 'FAIL')
        self.assertTrue(any('wrong number' in reason for reason in reasons))

    def test_an_inexact_fix_fails_even_when_the_order_is_reproduced(self):
        report = dict(opened=True, records=[
            spike.record('unfixed', 't0', 10, 0, None, tile=0), spike.record('unfixed', 't1', 10, 4, 3, tile=1),
            spike.record('fixed', 'f1', 10, 2, 1, tile=1)])
        outcome, reasons = spike.verdict(report)
        self.assertEqual(outcome, 'FAIL')
        self.assertTrue(any('not exact' in reason for reason in reasons))

    def test_an_empty_report_fails(self):
        self.assertEqual(spike.verdict(dict(opened=True, records=[]))[0], 'FAIL')
        self.assertEqual(spike.verdict(dict(opened=False, error='x'))[0], 'FAIL')


class Pieces(unittest.TestCase):
    def test_heights_are_whole_tiles_beyond_one(self):
        self.assertEqual(spike.parse_heights('64,128'), (64, 128))
        for text in ('32', '48', '64,64', '', 'x', '0'):
            with self.assertRaises(ValueError, msg=text):
                spike.parse_heights(text)

    def test_bits_are_compared_not_values(self):
        left = torch.tensor([0.0, 1.0, float('nan')], dtype=torch.bfloat16)
        right = torch.tensor([-0.0, 1.0, float('nan')], dtype=torch.bfloat16)
        self.assertEqual(spike.compare_bits(torch, left, right), (3, 1, 0))
        self.assertEqual(spike.compare_bits(torch, left, left.clone()), (3, 0, None))
        with self.assertRaises(ValueError):
            spike.compare_bits(torch, left, left[:2])

    def test_the_partials_are_full_mantissa_and_seeded(self):
        first = spike.partials(torch, 7, 32, 4)
        self.assertEqual(tuple(first.shape), (4, 1, 32, 5120))
        self.assertTrue(torch.equal(first, spike.partials(torch, 7, 32, 4)))
        self.assertFalse(torch.equal(first, spike.partials(torch, 8, 32, 4)))
        # not integers: the fabric probe's x % 17 values are exact under any order and could never show this
        self.assertGreater(float((first.float() != first.float().round()).float().mean()), 0.9)

    def test_the_spike_leaves_no_scope_open(self):
        run_spike(ring_all_reduce())
        self.assertIsNone(tile_collective_tp._STATE['rows'])


if __name__ == '__main__':
    unittest.main()
