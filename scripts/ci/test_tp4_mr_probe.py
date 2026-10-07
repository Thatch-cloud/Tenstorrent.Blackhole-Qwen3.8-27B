"""tp4_mr_probe on the CPU: the quad wrapper of the mesh-read probe, on the card-M probe's own fake ttnn at four chips.

Run at py 3.11: `py -3.11 -B -m unittest test_tp4_mr_probe` from scripts/ci."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'optimisation' / 'ttnn-op' / 'mr_probe'))

import c2_serving_job as job  # noqa: E402
import test_mr_probe as fakes  # noqa: E402
import tp4_mr_probe as quad  # noqa: E402


class QuadFake(fakes.FakeTTNN):
    FabricConfig = type('FabricConfig', (), {'FABRIC_1D': 'f1d', 'FABRIC_1D_RING': 'f1dr'})

    def __init__(self, chips=4, **keywords):
        super().__init__(chips=chips, **keywords)
        self.calls = []

    def set_fabric_config(self, value):
        self.calls.append(('fabric', value))

    def MeshShape(self, *shape):
        return shape

    def open_mesh_device(self, shape, **keywords):
        self.calls.append(('open', tuple(shape), keywords))
        chips = self.chips
        return type('Mesh', (), {'get_num_devices': lambda self: chips})()

    def close_mesh_device(self, mesh):
        self.calls.append(('close',))


def parse(*extra):
    return quad.build_parser().parse_args(['--output', 'x.json', '--iterations', '20', '--warmup', '2', *extra])


class RunTests(unittest.TestCase):
    def test_it_opens_the_one_by_four_mesh_under_the_fabric_config_once_and_closes_it(self):
        ttnn, lines = QuadFake(), []
        report = quad.run(parse('--fabric', 'FABRIC_1D_RING'), ttnn=ttnn, log=lines.append)
        self.assertEqual([call[0] for call in ttnn.calls], ['fabric', 'open', 'close'])
        self.assertEqual(ttnn.calls[0][1], 'f1dr')
        self.assertEqual(ttnn.calls[1][1], (1, 4))
        self.assertEqual((report['kind'], report['chips'], report['opened'], report['closed']), (quad.KIND, 4, True, True))
        self.assertIn(report['verdict'], ('GO', 'MESH-ONLY', 'NO-GO'))
        self.assertTrue(lines[0].startswith('MR_PROBE verdict=' + report['verdict'] + ' chips=4'))

    def test_four_chips_never_read_inconclusive_single_chip(self):
        report = quad.run(parse(), ttnn=QuadFake(), log=lambda text: None)
        self.assertNotEqual(report['verdict'], 'INCONCLUSIVE-SINGLE-CHIP')

    def test_an_open_failure_is_not_measured_and_closes_nothing(self):
        ttnn = QuadFake()

        def refuse(shape, **keywords):
            raise RuntimeError('no mesh')

        ttnn.open_mesh_device = refuse
        report = quad.run(parse(), ttnn=ttnn, log=lambda text: None)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertIn('no mesh', report['error'])

    def test_without_the_ring_descriptor_it_refuses_before_importing_ttnn(self):
        report = quad.run(parse(), environ={'TT_MESH_GRAPH_DESC_PATH': '/somewhere/pair.textproto'}, log=lambda text: None)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertIn('refused to open', report['error'])

    def test_main_writes_the_report_and_exits_by_the_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'mr.json'
            lines = []
            status = quad.main(['--output', str(out), '--iterations', '20', '--warmup', '2'], ttnn=QuadFake(), log=lines.append)
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(out.read_text())['kind'], quad.KIND)
            self.assertEqual(json.loads(lines[-1])['kind'], quad.KIND)

    def test_bad_arguments_are_refused(self):
        self.assertEqual(quad.main(['--output', 'x.json', '--rows', '65'], ttnn=QuadFake(), log=lambda text: None), 2)


class WiringTests(unittest.TestCase):
    def test_the_job_parser_and_the_workflow_know_the_mr_probe(self):
        self.assertIn('mr', job.FABRIC_PROBES)
        values = job.parse_env('C2_CARDS=quad\nC2_ACTIONS=reset fabric\nC2_FABRIC_PROBE=mr\nC2_IMAGE_TAG=tp4-w2-1\nC2_PROFILE=general-tp4\n')
        self.assertEqual(job.read_job(values, ['general-tp4'], root=ROOT)['fabric_probe'], 'mr')
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('mr) script=tp4_mr_probe.py; report=mr-probe.json', workflow)
        self.assertIn('FABRIC_PROBE|TP4_RS_TILE|MR_PROBE', workflow)
        self.assertTrue((HERE / 'tp4_mr_probe.py').is_file())

    def test_the_probe_reads_the_arms_of_the_card_m_probe(self):
        self.assertEqual(quad.mr_probe.run.__module__, 'mr_probe')


if __name__ == '__main__':
    unittest.main()
