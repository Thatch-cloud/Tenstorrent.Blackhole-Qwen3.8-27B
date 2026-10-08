"""tp4_kv_read_probe and kv_region_read_card on the CPU: the QUALIFY probe of the region read, on an in-memory fake ttnn at four chips.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_kv_read_probe` from scripts/ci."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'optimisation' / 'ttnn-op' / 'kv_region_read'))

import c2_serving_job as job  # noqa: E402
import kv_region_read_card as card  # noqa: E402
import tp4_kv_read_probe as probe  # noqa: E402


class FakeTensor(object):
    def __init__(self, data, per_chip_heads, dtype='bfloat8_b', layout='TILE'):
        self.data = data                      # the values of every chip's heads together: [blocks, chips x heads, 64, 256]
        self.shape = (data.shape[0], per_chip_heads) + tuple(data.shape[2:])
        self.dtype, self.layout = dtype, layout


class FakeMesh(object):
    def __init__(self, chips, programs=7):
        self.chips, self.programs = chips, programs

    def get_num_devices(self):
        return self.chips

    def num_program_cache_entries(self):
        return self.programs


class FakeTTNN(object):
    """Just enough ttnn for the card check; qwen_read_blocks is the extension's contract on in-memory tensors."""
    bfloat8_b, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bfloat8_b', 'TILE', 'DRAM'
    FabricConfig = type('FabricConfig', (), {'FABRIC_1D': 'f1d', 'FABRIC_1D_RING': 'f1dr'})

    def __init__(self, chips=4, with_extension=True, corrupt=False, lose_shard=False, grow_cache=False, flat_cost=False):
        self.chips, self.corrupt, self.lose_shard, self.grow_cache, self.flat_cost = chips, corrupt, lose_shard, grow_cache, flat_cost
        self.mesh = FakeMesh(chips)
        self.calls = []
        if with_extension:
            self.qwen_read_blocks = self._read

    def MeshShape(self, *shape):
        return shape

    def set_fabric_config(self, value):
        self.calls.append(('fabric', value))

    def open_mesh_device(self, shape, **keywords):
        self.calls.append(('open', tuple(shape)))
        return self.mesh

    def close_mesh_device(self, mesh):
        self.calls.append(('close',))

    ReplicateTensorToMesh = staticmethod(lambda mesh: 'replicate')
    ConcatMeshToTensor = staticmethod(lambda mesh, dim: ('concat', dim))

    @staticmethod
    def ShardTensorToMesh(mesh, dim):
        return ('shard', dim)

    @staticmethod
    def Shape(dims):
        return tuple(dims)

    def as_tensor(self, tensor, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        return FakeTensor(torch.zeros(tensor.shape[0], tensor.shape[1] * self.chips, *tensor.shape[2:], dtype=tensor.dtype), tensor.shape[1])

    def from_torch(self, tensor, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        return FakeTensor(tensor.clone(), tensor.shape[1] // self.chips)

    def copy(self, source, destination):
        destination.data = source.data.clone()

    def deallocate(self, tensor):
        pass

    def from_device(self, tensor):
        return tensor

    def to_torch(self, tensor, mesh_composer=None):
        data = tensor.data.float()
        return data[:, : data.shape[1] // self.chips].clone() if self.lose_shard and getattr(tensor, 'host', False) else data

    def allocate_tensor_on_host(self, shape, dtype, layout, mesh):
        host = FakeTensor(torch.zeros(shape[0], shape[1] * self.chips, *shape[2:], dtype=torch.bfloat16), shape[1])
        host.host = True
        return host

    def _read(self, cache, host, blocks):
        host.data = cache.data.index_select(0, torch.as_tensor(blocks, dtype=torch.long)).clone()
        if self.corrupt:
            host.data.view(-1)[0] += 1
        if self.grow_cache:
            self.mesh.programs += 1


def parse(*extra):
    return probe.build_parser().parse_args(['--output', 'x.json', '--blocks', '256', *extra])


class CardCheckTests(unittest.TestCase):
    def test_a_correct_read_passes_every_check(self):
        ttnn = FakeTTNN()
        report = card.run(ttnn, ttnn.mesh, 4, 256, 1)
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual(sorted(name for name in report if isinstance(report[name], dict)),
                         ['one', 'run', 'scattered', 'shuffled_order', 'two_runs'])
        self.assertTrue(all(report[name]['equal'] for name in ('one', 'run', 'scattered', 'shuffled_order', 'two_runs')))
        self.assertEqual(report['program_cache_growth'], 0)

    def test_wrong_bytes_a_lost_shard_and_a_compiling_read_each_fail(self):
        for knob, text in (('corrupt', 'differs from the whole-cache read'), ('lose_shard', 'a chip shard was lost'),
                           ('grow_cache', 'grew the program cache')):
            with self.subTest(knob=knob):
                ttnn = FakeTTNN(**{knob: True})
                report = card.run(ttnn, ttnn.mesh, 4, 256, 1)
                self.assertFalse(report['ok'])
                self.assertTrue(any(text in problem for problem in report['problems']), report['problems'])

    def test_the_extension_is_found_or_the_reason_is_named(self):
        self.assertEqual(card.ensure_extension(FakeTTNN()), (None, 'present'))
        reason, source = card.ensure_extension(FakeTTNN(with_extension=False))
        self.assertIsNone(source)
        self.assertTrue(reason and 'qwen_read_blocks' in reason or 'qwen_kv_read' in reason, reason)


class ProbeTests(unittest.TestCase):
    def test_it_opens_the_one_by_four_mesh_once_runs_the_checks_and_closes_it(self):
        ttnn, lines = FakeTTNN(), []
        report = probe.run(parse('--fabric', 'FABRIC_1D_RING'), ttnn=ttnn, log=lines.append)
        self.assertEqual([call[0] for call in ttnn.calls], ['fabric', 'open', 'close'])
        self.assertEqual(ttnn.calls[0][1], 'f1dr')
        self.assertEqual(ttnn.calls[1][1], (1, 4))
        self.assertEqual((report['kind'], report['verdict'], report['chips'], report['closed']), (probe.KIND, 'PASS', 4, True))
        self.assertTrue(any(line.startswith('KV_READ_PROBE verdict=PASS chips=4 blocks=256') for line in lines), lines)
        self.assertEqual(report['extension_source'], 'present')

    def test_a_mismatch_is_a_fail_with_the_problem_on_its_own_line(self):
        lines = []
        report = probe.run(parse(), ttnn=FakeTTNN(corrupt=True), log=lines.append)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertTrue(any(line.startswith('KV_READ_PROBE problem:') and 'differs' in line for line in lines), lines)

    def test_no_extension_is_not_measured_and_opens_no_mesh(self):
        ttnn, lines = FakeTTNN(with_extension=False), []
        report = probe.run(parse(), ttnn=ttnn, log=lines.append)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertEqual(ttnn.calls, [])
        self.assertIn('KV_READ_PROBE verdict=NOT-MEASURED', lines[0])

    def test_main_writes_the_report_and_exits_by_the_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'r.json')
            self.assertEqual(probe.main(['--output', out, '--blocks', '256'], ttnn=FakeTTNN(), log=lambda text: None), 0)
            self.assertEqual(json.loads(Path(out).read_text())['verdict'], 'PASS')
            self.assertEqual(probe.main(['--output', out, '--blocks', '256'], ttnn=FakeTTNN(corrupt=True), log=lambda text: None), 1)
            self.assertEqual(probe.main(['--output', out, '--blocks', '256'], ttnn=FakeTTNN(with_extension=False), log=lambda text: None), 2)
            self.assertEqual(probe.main(['--output', out, '--blocks', '8'], ttnn=FakeTTNN(), log=lambda text: None), 2)

    def test_the_job_and_the_workflow_know_the_probe(self):
        self.assertIn('kvread', job.FABRIC_PROBES)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('kvread) script=tp4_kv_read_probe.py; report=kv-read-probe.json', workflow)
        self.assertIn('KV_READ_PROBE', workflow)
        self.assertTrue((HERE / 'tp4_kv_read_probe.py').is_file())
        self.assertEqual(probe.PRODUCTION_BLOCKS, 19968)


if __name__ == '__main__':
    unittest.main()
