from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from compact_score_scope import scoped_compact_scores
from compact_score_gate import REPORT_SHA256


class CompactScopeTests(unittest.TestCase):
    def test_owned_route_and_restore_on_success_or_device_failure(self):
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                sources = {name: 'qualified' for name in
                           ('compact_score_hardware_device.py', 'compact_score_hardware_markov.py')}
                for name, source in sources.items():
                    (directory / name).write_bytes(source.encode())
                native = Mock()
                native._compact_score_override = False
                scope_module = SimpleNamespace(candidate=native)
                records = [dict(token=object(), diagnostic=object()) for _ in range(15)]
                execute = Mock(side_effect=RuntimeError('device') if failed else None, return_value=records)
                modules = {'compact_score_hardware_device': SimpleNamespace(__file__=directory / 'compact_score_hardware_device.py'),
                           'compact_score_hardware_markov': SimpleNamespace(
                               __file__=directory / 'compact_score_hardware_markov.py', execute=execute)}
                with patch.dict('sys.modules', {'dspark_score_layout_scope': scope_module}), \
                        patch('compact_score_report.validate', return_value=dict(simulator_qualified=True)), \
                        patch('compact_score_scope.payloads', return_value=sources), \
                        patch('compact_score_scope.importlib.import_module', side_effect=modules.__getitem__):
                    try:
                        with scoped_compact_scores(dict(report={}, report_sha256=REPORT_SHA256), directory) as audit:
                            with self.assertRaisesRegex(ValueError, 'Nested'):
                                with scoped_compact_scores(dict(report={}, report_sha256=REPORT_SHA256), directory):
                                    self.fail('Nested scope entered')
                            result = scope_module.candidate(None, SimpleNamespace(shape=[1, 2]), None,
                                SimpleNamespace(shape=(1, 1, 15, 248320)), None, None, [])
                            self.assertIs(result, records)
                    except RuntimeError:
                        self.assertTrue(failed)
                    self.assertIs(scope_module.candidate, native)
                    self.assertTrue(audit['restored'])
                    self.assertEqual(audit['calls'], 0 if failed else 1)
                    self.assertEqual(audit['steps'], 0 if failed else 15)

    def test_missing_report_rejected(self):
        with patch.dict('sys.modules', {'dspark_score_layout_scope': SimpleNamespace()}):
            with self.assertRaises(ValueError):
                with scoped_compact_scores({}, '.'):
                    self.fail('Missing admission entered')
