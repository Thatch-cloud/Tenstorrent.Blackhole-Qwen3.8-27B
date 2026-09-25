import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlp_compute_clock_hardware
from mlp_compute_clock_hardware import retained, verify_sources
from test_mlp_compute_clock_report import fixture


class ClockHardwareTests(unittest.TestCase):
    def test_stage_rejects_changed_simulator_source_before_modifying_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = b'changed probe'
            (directory / 'fused-batch-probe.py').write_bytes(source)
            (directory / 'compute-clock-candidate.json').write_text(json.dumps(
                {'sources': {'fused-batch-probe.py': hashlib.sha256(b'qualified probe').hexdigest()}}))
            with patch.object(mlp_compute_clock_hardware, 'retained', return_value=fixture()):
                with self.assertRaisesRegex(ValueError, 'differs from simulator'):
                    mlp_compute_clock_hardware.stage(directory, directory)
            self.assertEqual((directory / 'fused-batch-probe.py').read_bytes(), source)
            self.assertFalse((directory / 'compute-clock-hardware-sources.json').exists())

    def test_source_tampering_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'candidate.py').write_bytes(b'qualified source')
            manifest = {'candidate.py': hashlib.sha256(b'qualified source').hexdigest()}
            (directory / 'compute-clock-hardware-sources.json').write_text(json.dumps(manifest))
            verify_sources(directory)
            (directory / 'candidate.py').write_bytes(b'changed source')
            with self.assertRaises(ValueError):
                verify_sources(directory)
            (directory / 'compute-clock-hardware-sources.json').write_text(json.dumps({'../outside': 'none'}))
            with self.assertRaises(ValueError):
                verify_sources(directory)

    def test_retained_report_requires_hash_exit_runtime_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = json.dumps(fixture()).encode()
            (directory / 'fused-batch.json').write_bytes(raw)
            (directory / 'fused-batch.exit-status').write_text('0\n')
            (directory / 'simulator-runtime.txt').write_text('9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9\n')
            (directory / 'container-cleanup.json').write_text(json.dumps(
                dict(stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)))
            with self.assertRaises(ValueError):
                retained(directory)
            with patch.object(mlp_compute_clock_hardware, 'SIMULATOR_SHA256', hashlib.sha256(raw).hexdigest()):
                self.assertTrue(retained(directory)['passed'])
                for name, value in (('fused-batch.exit-status', '124'),
                        ('simulator-runtime.txt', 'wrong revision'), ('container-cleanup.json', '{}')):
                    path = directory / name
                    original = path.read_text()
                    path.write_text(value)
                    with self.assertRaises(ValueError):
                        retained(directory)
                    path.write_text(original)
