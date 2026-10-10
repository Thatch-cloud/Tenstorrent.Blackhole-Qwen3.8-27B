"""CPU tests of the F-F2 card-M probe: its host data, its compare, its verdict rule, the reading of the cached banks, and its whole run on a fake ttnn.

The fake ttnn holds tensors as RAW face-ordered tiles (test_draft_permute_tp's ExecutingOperations: generic_op runs the kernel's transliteration over the very runtime args the launch
builder writes), and its slice / concat / reshape are the served composition's model (CanonOps rules), so the whole probe - the served arm on real served code, the launch arm, the compare,
the information arm, the timing - runs on the CPU. A second fake whose concat does NOT round-trip the cached banks shows the probe telling the two readings apart.

Run: python -B -m unittest discover -s optimisation/ttnn-op/draft_permute -p 'test_*.py'
"""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
CI = HERE.parent.parent.parent / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))
import draft_permute_card_m as probe  # noqa: E402
import draft_permute_tp as perm  # noqa: E402
import test_draft_permute_tp as support  # noqa: E402
from test_tp4_vglue_gdn import canon  # noqa: E402


class FakeTTNN(support.ExecutingOperations):
    """ExecutingOperations as the module `ttnn` the probe imports: uploads, readbacks, the served ops' model, trace calls that run eagerly."""

    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'

    def __init__(self, concat_rule='all', width=13):
        super().__init__(chips=1, mesh=support.Mesh(width, 10))
        self.concat_rule = concat_rule
        self.opened, self.closed, self.traces = [], False, 0

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape, **keywords):
        self.opened.append((shape, keywords))
        return self.mesh

    def close_mesh_device(self, mesh):
        self.closed = True

    def ReplicateTensorToMesh(self, mesh):
        return mesh

    def from_torch(self, value, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        return self.from_logical(value.contiguous().view(torch.int16))

    def to_torch(self, shard):
        return super().to_torch(shard)

    def synchronize_device(self, mesh):
        pass

    def begin_trace_capture(self, mesh, cq_id=0):
        self.traces += 1
        return self.traces

    def end_trace_capture(self, mesh, handle, cq_id=0):
        pass

    def execute_trace(self, mesh, handle, cq_id=0, blocking=False):
        pass

    def release_trace(self, mesh, handle):
        pass

    # the served ops, on logical int16 matrices
    def logical(self, tensor):
        return self.to_logical(tensor)

    def slice(self, tensor, start, end):
        value = self.logical(tensor)[tuple(slice(low, high) for low, high in zip(start, end))].clone()
        if start[2] % 32 or start[3] % 32:
            value = canon(value)
        return self.from_logical(value)

    def concat(self, parts, dim, memory_config=None):
        values = [self.logical(part) for part in parts]
        if self.concat_rule == 'all':
            result = torch.cat(values, dim=dim)
            if dim in (2, 3) and any(value.shape[dim] % 32 for value in values):
                result = canon(result)
        else:       # 'unaligned-pieces': only the pieces that are not whole tiles are round-tripped
            result = torch.cat([canon(value) if dim in (2, 3) and value.shape[dim] % 32 else value for value in values], dim=dim)
        return self.from_logical(result)

    def reshape(self, tensor, shape):
        return self.from_logical(self.logical(tensor).reshape(tuple(shape)))


def run(options, fake=None, perm_module=perm, environment=None):
    fake = fake or FakeTTNN()
    with tempfile.TemporaryDirectory() as folder:
        out = os.path.join(folder, 'report.json')
        values = {'QWEN_FAST_TP': '4'}
        values.update(environment or {})
        with patch.dict(os.environ, values), contextlib.redirect_stdout(io.StringIO()):
            status = probe.main(['--out', out] + options, torch=torch, ttnn=fake, perm=perm_module)
        with open(out) as handle:
            report = json.load(handle)
    return status, report, fake


class HostDataTests(unittest.TestCase):
    def test_the_sweep_holds_every_pattern_the_edge_regime_is_finite_and_full_of_zero_exponents_and_random_is_normal(self):
        generator = torch.Generator().manual_seed(1)
        sweep = probe.host_bits(torch, (1, 2, 2048, 128), 'sweep', generator, offset=7)
        values = (sweep.to(torch.int64) & 0xFFFF).reshape(-1)
        self.assertEqual(len(set(values.tolist())), 65536)
        edge = probe.host_bits(torch, (1, 2, 64, 128), 'edge', generator)
        unsigned = edge.to(torch.int32) & 0xFFFF
        self.assertFalse(bool(((unsigned & 0x7F80) == 0x7F80).any()), 'no infinity and no NaN')
        self.assertGreater(float(((unsigned & 0x7F80) == 0).float().mean()), 0.2)
        self.assertTrue(bool((unsigned == 0x8000).any()) and bool((unsigned == 0x0001).any()))
        normal = probe.host_bits(torch, (1, 2, 64, 128), 'random', generator)
        self.assertEqual(normal.dtype, torch.int16)
        self.assertLess(float(((normal.to(torch.int32) & 0x7F80) == 0).float().mean()), 0.01)

    def test_the_seed_makes_the_data_and_the_sweep_offset_moves_it(self):
        one = probe.host_bits(torch, (1, 2, 64, 128), 'edge', torch.Generator().manual_seed(3))
        two = probe.host_bits(torch, (1, 2, 64, 128), 'edge', torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(one, two))
        a = probe.host_bits(torch, (1, 2, 64, 128), 'sweep', None, offset=0)
        b = probe.host_bits(torch, (1, 2, 64, 128), 'sweep', None, offset=5)
        self.assertFalse(torch.equal(a, b))

    def test_locate_names_the_piece_and_the_value(self):
        plan, rows, live_rows = probe.plan_of('quad')
        left = torch.zeros(1, 2, 8320, 128, dtype=torch.int16)
        right = left.clone()
        right[0, 1, 2048 + 3, 9] = -1                       # user 0's live rows
        right[0, 0, 2080 + 100, 0] = 5                      # user 1's cached bank
        found = probe.locate(torch, left, right, plan)
        self.assertEqual([(item['head'], item['row'], item['kind'], item['user']) for item in found],
                         [(0, 2180, 'cached', 1), (1, 2051, 'live', 0)])
        self.assertEqual((found[1]['mine'], found[1]['served']), ('0x0000', '0xFFFF'))

    def test_the_verdict_rule(self):
        exact = dict(differing=0, fell_back=False)
        self.assertEqual(probe.verdict([exact, exact]), ('PASS', 0))
        self.assertEqual(probe.verdict([exact, dict(differing=3, fell_back=False)]), ('FAIL', 1))
        self.assertEqual(probe.verdict([exact, dict(differing=-1, fell_back=True)]), ('FAIL', 1))
        self.assertEqual(probe.verdict([exact, dict(error='boom')]), ('NOT-RUN', 4))
        self.assertEqual(probe.verdict([]), ('NOT-RUN', 4))


class RunTests(unittest.TestCase):
    def setUp(self):
        quiet = patch.object(perm.tp4_sampdraft, 'log_line')
        quiet.start()
        self.addCleanup(quiet.stop)
        perm._LOGGED.clear()
        perm._CACHE.clear()

    def test_a_whole_run_passes_on_the_model_names_the_canonical_reading_and_times_every_case(self):
        status, report, fake = run(['--regimes', 'random,edge,sweep', '--shapes', 'quad,pair,pair-short,octo', '--timing', 'on'])
        self.assertEqual(status, 0, report.get('error') or [section for section in report['compare'] if section.get('differing')][:3])
        self.assertEqual(report['verdict'], 'PASS')
        self.assertEqual(report['grid'], [13, 10])
        sites = {(section['site'], section['shape']) for section in report['compare']}
        self.assertEqual(sites, {('kv', 'quad'), ('kv', 'pair'), ('kv', 'pair-short'), ('kv', 'octo'),
                                 ('fold', 'quad'), ('fold', 'pair'), ('fold', 'octo'), ('unfold', 'quad'), ('unfold', 'pair'), ('unfold', 'octo')})
        self.assertTrue(all(section['differing'] == 0 and section['fell_back'] is False for section in report['compare']))
        self.assertEqual(report['cached_banks'], 'canonical (the model holds)')
        self.assertTrue(all(item['differing'] > 0 for item in report['cached_raw_reading'] if 'differing' in item), 'the other reading differs on edge data: the probe can tell them apart')
        self.assertEqual(len(report['timing']), 10)
        self.assertTrue(all('error' not in item for item in report['timing']), report['timing'])
        self.assertTrue(fake.closed)
        self.assertEqual(fake.opened[0][0], (1, 1))
        self.assertEqual(fake.opened[0][1], {'trace_region_size': probe.TRACE_REGION})

    def test_a_concat_that_does_not_round_trip_the_banks_fails_the_verdict_and_names_the_raw_reading(self):
        status, report, _ = run(['--regimes', 'edge', '--shapes', 'quad,pair', '--sites', 'kv', '--timing', 'off'], fake=FakeTTNN(concat_rule='unaligned-pieces'))
        self.assertEqual(status, 1)
        self.assertEqual(report['verdict'], 'FAIL')
        failing = [section for section in report['compare'] if section['differing']]
        self.assertTrue(failing)
        self.assertEqual({item['kind'] for section in failing for item in section['first']['k']}, {'cached'}, 'the differing rows are the cached banks')
        self.assertEqual(report['cached_banks'], 'raw (flip canon_cached)')

    def test_a_launch_that_falls_back_fails_the_run(self):
        with patch.object(perm, 'plan_lanes', side_effect=perm.Unsupported('no room')):
            status, report, _ = run(['--regimes', 'edge', '--shapes', 'pair', '--sites', 'kv,fold', '--timing', 'off'])
        self.assertEqual(status, 1)
        self.assertTrue(all(section['fell_back'] for section in report['compare']))

    def test_a_section_that_raises_is_not_run(self):
        fake = FakeTTNN()
        fake.generic_op = lambda *arguments: (_ for _ in ()).throw(RuntimeError('compile failed'))
        status, report, _ = run(['--regimes', 'edge', '--shapes', 'pair', '--sites', 'kv', '--timing', 'off'], fake=fake)
        self.assertEqual(status, 4)
        self.assertEqual(report['verdict'], 'NOT-RUN')
        self.assertIn('compile failed', report['compare'][0]['error'])

    def test_without_the_four_card_geometry_the_run_is_not_run(self):
        status, report, _ = run(['--regimes', 'edge', '--shapes', 'pair', '--sites', 'kv', '--timing', 'off'], environment={'QWEN_FAST_TP': '2'})
        self.assertEqual(status, 4)
        self.assertIn('QWEN_FAST_TP=4', report['error'])

    def test_a_11_by_10_grid_is_served_too(self):
        status, report, _ = run(['--regimes', 'edge', '--shapes', 'quad', '--sites', 'kv,fold,unfold', '--timing', 'off'], fake=FakeTTNN(width=11))
        self.assertEqual((status, report['grid']), (0, [11, 10]))

    def test_bad_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            self.assertEqual(probe.main(['--out', os.path.join(folder, 'x.json'), '--sites', 'nope'], torch=torch, ttnn=FakeTTNN(), perm=perm), 2)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.script = (HERE / 'run_card_m.sh').read_text()

    def test_the_harness_runs_the_probe_on_one_named_board_and_mounts_this_checkouts_scripts(self):
        self.assertIn('qual_card_select', self.script)
        self.assertIn('draft_permute_card_m.py', self.script)
        self.assertIn('-e QWEN_FAST_TP=4', self.script)
        self.assertIn('QWEN_C2_SERVING=0', self.script)
        self.assertIn('env -u TT_MESH_GRAPH_DESC_PATH', self.script.replace('--entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH', 'env -u TT_MESH_GRAPH_DESC_PATH'))
        self.assertIn('/bench/ci', self.script)
        self.assertIn('PYTHONPATH', self.script)
        self.assertNotRegex(self.script, r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')


if __name__ == '__main__':
    unittest.main()
