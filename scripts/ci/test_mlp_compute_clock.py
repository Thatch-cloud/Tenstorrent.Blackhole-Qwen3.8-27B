from pathlib import Path
import subprocess
import unittest

from frozen_recipe_context import REVISION
from mlp_compute_clock import WORDS, MAGIC, ZONES, decode
from mlp_compute_clock_projection import instrument_projection, remove_projection


class ComputeClockTests(unittest.TestCase):
    def page(self, processor):
        words = [0xffffffff] * WORDS
        for index in range(len(ZONES)):
            start = (1 << 32) - 5 + 100 * index
            end = start + 20
            words[index * 6:index * 6 + 6] = [start & 0xffffffff, start >> 32,
                end & 0xffffffff, end >> 32, index, MAGIC ^ index ^ processor]
        return words

    def test_all_processors_and_clock_rollover(self):
        for processor in range(3):
            records = decode(self.page(processor), processor)
            self.assertEqual([record['duration_cycles'] for record in records], [20] * 6)
            self.assertEqual([record['zone'] for record in records], [zone[0] for zone in ZONES])
            self.assertEqual(records[3]['block'], 19)
            self.assertEqual(records[1]['block'], 10)

    def test_poison_wrong_processor_and_out_of_order_rejected(self):
        for words, processor in (([0xffffffff] * WORDS, 0), (self.page(1), 0),
                (self.page(0)[:-1], 0), (self.page(0), True)):
            with self.assertRaises(ValueError):
                decode(words, processor)
        words = self.page(0)
        words[6:10] = words[:4]
        with self.assertRaises(ValueError):
            decode(words, 0)
        words = self.page(0)
        words[2] += 100_000_001
        with self.assertRaises(ValueError):
            decode(words, 0)

    def test_projection_preserves_native_source_and_uses_distinct_pages(self):
        source = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/fused_1d.py'], text=True)
        candidate = instrument_projection(source)
        self.assertEqual(remove_projection(candidate), source)
        self.assertIn('int(core_x == 0 and core_y == 0)', candidate)
        self.assertIn('compute.runtime_args = compute_args', candidate)
        self.assertIn('self.compute_samples], mesh_program)', candidate)
        with self.assertRaises(ValueError):
            instrument_projection(candidate)

    def test_native_compute_transform_roundtrip(self):
        from fused_1d import fused_compute
        from mlp_compute_clock import instrument, remove

        root = Path('D:/qwen-evidence/35092212895/sources')
        path = root / 'ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp'
        if not path.exists():
            self.skipTest('Retained pinned kernel export unavailable')
        control = fused_compute(path.read_text(), pairs_per_worker=3)
        candidate = instrument(control)
        self.assertEqual(remove(candidate), control)
        for index in range(6):
            self.assertEqual(candidate.count(f'qwen_samples[{index * 6 + 5}] ='), 1)
        with self.assertRaises(ValueError):
            instrument(candidate)
        with self.assertRaises(ValueError):
            instrument(fused_compute(path.read_text(), pairs_per_worker=7))

    def test_probe_retains_numerical_matrix_and_poison_controls(self):
        from mlp_compute_clock_stage import probe_source

        source = subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/fused-batch-probe.py'], text=True)
        candidate = probe_source(source)
        self.assertIn('ComputeClockCapture as ClockCapture', candidate)
        self.assertIn('compute_samples=sample_capture.buffer', candidate)
        self.assertIn('sample_capture.reject_missing_execution()', candidate)
        self.assertIn("report['compute_clock_samples'] = sample_capture.records", candidate)
        self.assertIn('len(sample_capture.records) != 5', candidate)
        self.assertIn('committed_tg', candidate)
