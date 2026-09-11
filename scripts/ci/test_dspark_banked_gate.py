import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dspark_banked_device import BankedDSparkDevice
from dspark_banked_gate import SOURCES, qualify


class BankedAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in SOURCES:
            (self.directory / name).write_bytes(name.encode())
        sources = {name: hashlib.sha256(name.encode()).hexdigest() for name in SOURCES}
        self.report = dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
            learned_model_executed=False, replay_counts=[2, 2], sources=sources, sources_after=dict(sources),
            checks=[dict(ordinal=ordinal, bank=bank, operand=operand, chip=chip, exact=True)
                for ordinal, bank in enumerate((0, 1, 1, 0)) for operand in range(10) for chip in range(2)],
            bank_checks=[dict(ordinal=ordinal, bank=bank, operand=operand, chip=chip, exact=True)
                for ordinal in (0, 1, 2, 3, 'after_close') for bank in range(2)
                for operand in range(10) for chip in range(2)],
            eager_replay_checks=[dict(position=position, tensors=6, exact=True) for position in range(31, 35)])
        self.path = self.directory / 'report.json'
        self.path.with_suffix('.exit-status').write_text('0')

    def check(self):
        self.path.write_text(json.dumps(self.report))
        return qualify(self.path, self.directory)

    def test_complete_report_is_only_synthetic_admission(self):
        self.assertIn('learned request audits still required', self.check()['scope'])

    def test_duplicate_missing_failed_and_stale_evidence_rejected(self):
        original = json.dumps(self.report)
        for field in ('checks', 'bank_checks'):
            for mutation in ('duplicate', 'missing', 'failed'):
                self.report = json.loads(original)
                if mutation == 'duplicate':
                    self.report[field][-1] = self.report[field][0]
                elif mutation == 'missing':
                    self.report[field].pop()
                else:
                    self.report[field][0]['exact'] = False
                with self.assertRaises(ValueError):
                    self.check()
        self.report = json.loads(original)
        (self.directory / SOURCES[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'sources'):
            self.check()

    def test_device_installs_two_bank_trace_once(self):
        device = SimpleNamespace(closed=False, prepared=None)
        prepared = Mock()
        with patch('dspark_banked_device.BankedDSparkProposal', return_value=prepared) as constructor:
            BankedDSparkDevice.prepare_trace(device, 10, audit=True)
            constructor.assert_called_once_with(device, 10, audit=True)
            self.assertIs(device.prepared, prepared)
            with self.assertRaises(ValueError):
                BankedDSparkDevice.prepare_trace(device, 10)
            device.prepared, device.closed = None, True
            with self.assertRaises(ValueError):
                BankedDSparkDevice.prepare_trace(device, 10)
