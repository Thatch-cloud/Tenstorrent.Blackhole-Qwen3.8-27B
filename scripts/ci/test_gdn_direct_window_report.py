import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from gdn_direct_window_device import HASHES
from gdn_direct_window_report import validate


class DirectReportTests(unittest.TestCase):
    def fixture(self):
        directory = Path(__file__).parent
        names = ('gdn_direct_window.py', 'gdn_direct_window_device.py', 'gdn_conv_windows.py',
                 'gdn_conv_windows.cpp', 'attention_batch.py', 'gdn_multitoken_conv.py')
        hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
        hashes['gdn-output-grid-probe.py'] = hashlib.sha256((directory / 'gdn-direct-window-probe.py').read_bytes()).hexdigest()
        result = dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
            projection_memory='L1',
            performance_qualified=False, native_unchanged=True, native_sources=dict(HASHES),
            sources=hashes, sources_after=dict(hashes), generated_reader_sha256=hashlib.sha256(b'reader').hexdigest())
        for name, count in (('checks', 7), ('immutable_checks', 11)):
            result[name] = [dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True)
                           for mode, seed in (('eager', 0), ('replay', 0), ('replay', 1), ('replay', 2))
                           for operand in range(count) for chip in (0, 1)]
        return result

    def test_complete_matrix_and_rejected_mutations(self):
        fixture = self.fixture()
        with patch('gdn_direct_window_report.sources', return_value=dict(reader='reader')):
            self.assertFalse(validate(fixture, Path(__file__).parent, '.')['hardware_qualified'])
            for mutation in (
                    lambda report: report['checks'].pop(),
                    lambda report: report['checks'][0].update(exact=False),
                    lambda report: report['immutable_checks'][0].update(chip=1),
                    lambda report: report.update(closed_cleanly=False),
                    lambda report: report.update(native_unchanged=False),
                    lambda report: report.update(projection_memory='DRAM'),
                    lambda report: report['native_sources'].clear(),
                    lambda report: report.update(generated_reader_sha256='0' * 64),
                    lambda report: report['sources_after'].update(extra='0' * 64)):
                report = copy.deepcopy(fixture)
                mutation(report)
                with self.assertRaises(ValueError):
                    validate(report, Path(__file__).parent, '.')
