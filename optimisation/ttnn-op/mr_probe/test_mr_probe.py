"""CPU tests of the MR probe: its statistics, its verdict rule, its arms on a fake ttnn, its failure containment and its harness script.

Run: python -B -m unittest discover -s optimisation/ttnn-op/mr_probe -p 'test_*.py'
"""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mr_probe as probe  # noqa: E402


class FakeTensor(object):
    def __init__(self, name, chips):
        self.name, self.chips = name, chips


class FakeTTNN(object):
    """Counts every read by kind; no clock: the arms' times are the fake clock's steps."""
    DRAM_MEMORY_CONFIG = 'dram'
    uint32, bfloat16 = 'uint32', 'bfloat16'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT = 'row', 'tile'

    def __init__(self, chips=4, break_api=()):
        self.chips, self.break_api = chips, set(break_api)
        self.reads = {'part': 0, 'mesh': 0, 'host': 0}
        self.synchronized = 0

    def check(self, name):
        if name in self.break_api:
            raise AttributeError('no %s in this runtime' % name)

    def from_torch(self, tensor, **keywords):
        return FakeTensor('t%d' % id(tensor), self.chips)

    def ReplicateTensorToMesh(self, mesh):
        return 'replicate'

    def ConcatMeshToTensor(self, mesh, dim):
        self.check('ConcatMeshToTensor')
        return ('concat', dim)

    def get_device_tensors(self, tensor):
        return [FakeTensor('%s:%d' % (tensor.name, chip), 1) for chip in range(tensor.chips)]

    def to_torch(self, tensor, mesh_composer=None):
        if mesh_composer is not None:
            self.reads['mesh'] += 1
            return torch.zeros(tensor.chips * 64)
        self.reads['host' if getattr(tensor, 'host', False) else 'part'] += 1
        return torch.zeros(64)

    def from_device(self, tensor, blocking=True):
        self.check('from_device')
        copy = FakeTensor(tensor.name, 1)
        copy.host = True
        return copy

    def synchronize_device(self, mesh):
        self.synchronized += 1


class Clock(object):
    """Every call advances by a step that depends on what the fake has read since the last call: reads cost 100 us, a mesh read 150 us."""

    def __init__(self, ttnn):
        self.ttnn, self.time, self.seen = ttnn, 0, 0

    def __call__(self):
        reads = sum(self.ttnn.reads.values())
        self.time += 1000 + 100000 * (reads - self.seen)
        self.seen = reads
        return self.time


class StatisticsTests(unittest.TestCase):
    def test_summary(self):
        found = probe.summarize([5000, 1000, 3000, 2000, 4000])
        self.assertEqual((found['n'], found['median_us'], found['min_us']), (5, 3.0, 1.0))
        self.assertEqual(probe.summarize([]), {'n': 0})

    def test_the_percentiles_bracket_the_median(self):
        found = probe.summarize(list(range(1000, 101000, 1000)))
        self.assertLessEqual(found['p10_us'], found['median_us'])
        self.assertLessEqual(found['median_us'], found['p90_us'])


def arms(**medians):
    return {name: dict(n=10, median_us=value) for name, value in medians.items()}


class VerdictTests(unittest.TestCase):
    def test_one_chip_is_inconclusive_by_construction(self):
        text, evidence = probe.verdict(1, arms(floor=20.0, serial=45.0, mesh=40.0, word=22.0))
        self.assertEqual(text, 'INCONCLUSIVE-SINGLE-CHIP')
        self.assertEqual(evidence['chips'], 1)
        self.assertEqual(evidence['serial_over_floor'], 2.25)

    def test_go_needs_the_mesh_read_to_win_by_a_quarter_and_the_word_to_beat_it(self):
        self.assertEqual(probe.verdict(4, arms(floor=20.0, serial=160.0, mesh=100.0, word=60.0))[0], 'GO')
        self.assertEqual(probe.verdict(4, arms(floor=20.0, serial=160.0, mesh=100.0, word=120.0))[0], 'MESH-ONLY')
        self.assertEqual(probe.verdict(4, arms(floor=20.0, serial=160.0, mesh=150.0, word=60.0))[0], 'NO-GO')
        self.assertEqual(probe.verdict(4, arms(floor=20.0, serial=160.0, mesh=120.0, word=60.0))[0], 'GO', 'exactly the quarter')

    def test_a_missing_arm_is_incomplete_not_a_verdict(self):
        self.assertEqual(probe.verdict(4, arms(floor=20.0, serial=160.0))[0], 'INCOMPLETE')
        self.assertEqual(probe.verdict(4, {})[0], 'INCOMPLETE')

    def test_the_word_arm_missing_is_mesh_only_when_the_mesh_wins(self):
        self.assertEqual(probe.verdict(4, arms(floor=20.0, serial=160.0, mesh=100.0))[0], 'MESH-ONLY')


class RunTests(unittest.TestCase):
    def run_arms(self, chips=4, break_api=(), iterations=20, warmup=4):
        ttnn = FakeTTNN(chips, break_api)
        results, errors = probe.run(ttnn, 'mesh', chips, iterations, warmup, 64, clock=Clock(ttnn))
        return ttnn, results, errors

    def test_every_arm_runs_the_same_number_of_timed_iterations(self):
        ttnn, results, errors = self.run_arms()
        self.assertEqual(errors, {})
        for name in probe.ARMS:
            self.assertEqual(results[name]['n'], 20, name)

    def test_the_served_loop_reads_two_parts_per_chip_and_the_word_arm_one_mesh_read(self):
        ttnn, results, errors = self.run_arms(chips=4, iterations=20, warmup=4)
        # per iteration: floor 1 part; serial 8 parts; mesh 2 mesh reads; word 1 mesh read; overlap 8 host copies; wide_serial 4 parts; wide_mesh 1
        rounds = 20 + 4 + 2
        self.assertEqual(ttnn.reads['part'], rounds * (1 + 8 + 4))
        self.assertEqual(ttnn.reads['mesh'], rounds * (2 + 1 + 1))
        self.assertEqual(ttnn.reads['host'], rounds * 8)
        self.assertEqual(ttnn.synchronized, rounds)

    def test_the_fake_clock_orders_the_arms_by_their_reads(self):
        ttnn, results, errors = self.run_arms(chips=4)
        self.assertGreater(results['serial']['median_us'], results['word']['median_us'])
        self.assertGreater(results['serial']['median_us'], results['floor']['median_us'])
        self.assertEqual(probe.verdict(4, results)[0], 'GO')

    def test_an_arm_whose_api_is_missing_is_skipped_and_the_others_are_timed(self):
        ttnn, results, errors = self.run_arms(break_api=('from_device',))
        self.assertEqual(set(errors), {'overlap'})
        self.assertIn('AttributeError', errors['overlap'])
        self.assertEqual(results['overlap'], {'n': 0})
        self.assertEqual(results['serial']['n'], 20)

    def test_a_missing_composer_skips_every_mesh_arm_and_nothing_else(self):
        ttnn, results, errors = self.run_arms(break_api=('ConcatMeshToTensor',))
        self.assertEqual(set(errors), {'mesh', 'word', 'wide_mesh'})
        for name in ('floor', 'serial', 'overlap', 'wide_serial'):
            self.assertEqual(results[name]['n'], 20, name)
        self.assertEqual(probe.verdict(4, results)[0], 'INCOMPLETE')


class MainTests(unittest.TestCase):
    def main_with(self, ttnn, chips, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'mr.json'
            module = SimpleNamespace(get_num_devices=lambda: chips, open_mesh_device=lambda shape: 'mesh',
                                     close_mesh_device=lambda mesh: None, MeshShape=lambda *shape: shape)
            for name in dir(ttnn):
                if not name.startswith('_') and not hasattr(module, name):
                    setattr(module, name, getattr(ttnn, name))
            lines = []
            with patch.dict(sys.modules, {'ttnn': module}), patch('builtins.print', side_effect=lambda *a, **k: lines.append(' '.join(map(str, a)))):
                with patch.object(probe, 'time', SimpleNamespace(perf_counter_ns=Clock(ttnn))):
                    status = probe.main(['--out', str(out), '--iterations', '20', '--warmup', '2', *extra])
            return status, lines, json.loads(out.read_text())

    def test_one_card_prints_the_verdict_line_then_the_json(self):
        ttnn = FakeTTNN(1)
        status, lines, report = self.main_with(ttnn, 1)
        self.assertEqual(status, 0)
        self.assertTrue(lines[-2].startswith('MR_PROBE verdict=INCONCLUSIVE-SINGLE-CHIP chips=1 '), lines[-2])
        self.assertEqual(json.loads(lines[-1])['kind'], probe.KIND)
        self.assertEqual(report['verdict'], 'INCONCLUSIVE-SINGLE-CHIP')
        self.assertEqual(report['chips'], 1)
        self.assertEqual(sorted(report['arms']), sorted(probe.ARMS))

    def test_a_runtime_that_cannot_open_the_mesh_is_not_measured_and_exits_2(self):
        ttnn = FakeTTNN(1)
        module = SimpleNamespace(get_num_devices=lambda: 1, MeshShape=lambda *shape: shape,
                                 open_mesh_device=lambda shape: (_ for _ in ()).throw(RuntimeError('no device')))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'ttnn': module}), \
                patch('builtins.print', side_effect=lambda *a, **k: None):
            out = Path(directory) / 'mr.json'
            status = probe.main(['--out', str(out)])
            report = json.loads(out.read_text())
        self.assertEqual(status, 2)
        self.assertEqual(report['verdict'], 'NOT-MEASURED')
        self.assertIn('no device', report['error'])

    def test_bad_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            for bad in (['--rows', '0'], ['--rows', '65'], ['--iterations', '5']):
                self.assertEqual(probe.main(['--out', str(Path(directory) / 'x.json'), *bad]), 2)


class HarnessTests(unittest.TestCase):
    def test_it_runs_the_probe_in_one_card_container_with_the_serving_hook_off(self):
        text = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        for fragment in ('-e QWEN_C2_SERVING=0 --entrypoint env "$IMAGE" -u TT_MESH_GRAPH_DESC_PATH python3 -B /bench/mr_probe.py',
                         '--network none', '--cap-drop ALL', 'qual_card_select', 'qual_refuse_holders', 'qual_card_recheck',
                         'src=$probe,dst=/bench/mr_probe.py,readonly'):
            self.assertIn(fragment, text)
        self.assertNotIn('\r', text)

    def test_it_names_no_host_registry_or_card(self):
        text = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        body = text[:text.index('# >>> qual_card.sh')] + text[text.index('# <<< qual_card.sh'):]
        for fragment in ('blackhole-', 'zot', '10.', '192.168'):
            self.assertNotIn(fragment, body)


if __name__ == '__main__':
    unittest.main()
