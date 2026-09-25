from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest

from gdn_recurrence_clock_program import instrument_pipeline, remove_pipeline
from gdn_recurrence_clock_stage import adapt


class RecurrenceProgramTests(unittest.TestCase):
    def source(self):
        return Path(__file__).with_name('gdn_shared_qk_pipeline.py').read_text(encoding='utf-8')

    def test_lossless_adapter_and_complete_probe_matrix(self):
        source = self.source()
        candidate = instrument_pipeline(source)
        self.assertEqual(remove_pipeline(candidate), source)
        with self.assertRaises(ValueError):
            instrument_pipeline(candidate)
        probe = Path(__file__).with_name('gdn-shared-recurrence-probe.py').read_text(encoding='utf-8')
        changed = adapt(probe)
        for statement in ("len(report['checks']) != 24", "len(report['immutable_checks']) != 48",
                'for seed in (1, 2, 0)', 'candidate.samples.prepare()',
                "candidate.samples.collect('replay_' + str(seed))",
                'samples=self.samples.buffer', 'len(candidate.samples.records) != 4'):
            self.assertIn(statement, changed)
        self.assertLess(changed.index('ttnn.release_trace'), changed.index('operation.close()'))
        with self.assertRaises(ValueError):
            adapt(changed)

    def test_both_chip_programs_preserve_dataflow_cb_and_compute_config(self):
        modules = []
        for source in (self.source(), instrument_pipeline(self.source())):
            namespace = {}
            exec(compile(source, 'pipeline.py', 'exec'), namespace)
            modules.append(namespace)
        operations = SimpleNamespace(
            CoreRangeSet=list, CoreRange=lambda *values: values, CoreCoord=lambda *values: values,
            CBDescriptor=SimpleNamespace, CBFormatDescriptor=SimpleNamespace,
            TileDescriptor=lambda value: value, Tile=list,
            bfloat16='bf16', float32='fp32', MeshProgramDescriptor=dict,
            DataMovementConfigDescriptor=SimpleNamespace,
            DataMovementProcessor=SimpleNamespace(RISCV_1='reader', RISCV_0='writer'),
            NOC=SimpleNamespace(RISCV_1_default=1, RISCV_0_default=0),
            ComputeConfigDescriptor=SimpleNamespace, MathFidelity=SimpleNamespace(HiFi4='HiFi4'),
            TensorAccessorArgs=lambda value: SimpleNamespace(get_compile_time_args=lambda: []),
            RuntimeArgs=lambda: defaultdict(dict),
            MeshCoordinate=lambda *values: values, MeshCoordinateRange=lambda *values: values,
            ProgramDescriptor=SimpleNamespace)
        class KernelDescriptor(SimpleNamespace):
            SourceType = SimpleNamespace(SOURCE_CODE='source')
        operations.KernelDescriptor = KernelDescriptor
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=13, y=10))
        shards = [[SimpleNamespace(buffer_address=lambda address=1000 * (index + 1) + chip: address)
            for chip in range(2)] for index in range(12)]
        kernels = {'recurrence': dict(reader='reader', writer='writer', compute='compute')}
        baseline = modules[0]['build_recurrence'](operations, mesh, shards[:11], kernels)
        candidate = modules[1]['build_recurrence'](operations, mesh, shards, kernels, 8)
        self.assertEqual(set(baseline), set(candidate))
        for coordinate in baseline:
            before, after = baseline[coordinate], candidate[coordinate]
            self.assertEqual(before.cbs, after.cbs)
            self.assertEqual(before.kernels[:2], after.kernels[:2])
            self.assertEqual(before.kernels[2].config, after.kernels[2].config)
            chip = coordinate[0][1]
            for horizontal, columns in after.kernels[2].runtime_args.items():
                for vertical, values in columns.items():
                    self.assertEqual(values, [16, 12000 + chip, int((horizontal, vertical) == (0, 0)), 8])
        shards[11][0].buffer_address = shards[0][0].buffer_address
        with self.assertRaisesRegex(ValueError, 'must not alias'):
            modules[1]['build_recurrence'](operations, mesh, shards, kernels, 8)


if __name__ == '__main__':
    unittest.main()
