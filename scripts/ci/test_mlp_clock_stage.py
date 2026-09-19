from pathlib import Path
import subprocess
import unittest

from frozen_recipe_context import REVISION
from mlp_clock_projection import instrument_projection, remove_projection
from mlp_clock_stage import probe_source, trace_source


class ClockStageTests(unittest.TestCase):
    def test_exact_frozen_sources_are_supported_without_relaxing_checks(self):
        directory = Path(__file__).resolve().parents[2]
        for name, transform in (('fused_1d.py', instrument_projection),
                ('fused-batch-probe.py', probe_source), ('fusion_trace.py', trace_source)):
            original = subprocess.check_output(['git', '-C', str(directory), 'show',
                f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
            candidate = transform(original)
            compile(candidate, name, 'exec')
            if name == 'fused_1d.py':
                self.assertEqual(remove_projection(candidate), original)
            elif name == 'fusion_trace.py':
                self.assertEqual(candidate.count('sample_capture.prepare()'), 2)
                self.assertEqual(candidate.count('sample_capture.collect('), 2)
                self.assertEqual(candidate.count('operations.execute_trace('), original.count('operations.execute_trace('))
                self.assertEqual(candidate.count('raise AssertionError'), original.count('raise AssertionError'))
            else:
                self.assertIn('sample_buffers=sample_capture.buffers', candidate)
                self.assertIn('len(sample_capture.records) != 5', candidate)
                self.assertIn("report['clock_samples'] = sample_capture.records", candidate)
                self.assertIn('for rows in (16,):', candidate)
                self.assertIn('sample_capture.reject_missing_execution()', candidate)
                self.assertIn("if options.hardware or options.timing", candidate)

    def test_trace_drift_rejected(self):
        source = Path(__file__).with_name('fusion_trace.py').read_text()
        with self.assertRaises(ValueError):
            trace_source(source.replace('blocking=True)', 'blocking=False)'))
