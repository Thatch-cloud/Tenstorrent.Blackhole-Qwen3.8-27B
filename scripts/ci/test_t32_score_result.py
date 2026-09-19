import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from test_dspark_t32_result import complete_report


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScoreResultTests(unittest.TestCase):
    def setUp(self):
        self.gate = load('t32-score-result')
        probe = load('dspark-t32-markov-probe')
        self.report = complete_report()
        self.report.update(score_layout='fused', sources=probe.source_hashes(), sources_after=probe.source_hashes())
        native = dict(probe.BINARY_SHA256, **{probe.PACKER: probe.ORIGINAL_PACKER})
        self.report.update(native_sources=native, native_sources_after=dict(native))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'report.json'
        self.path.with_suffix('.exit-status').write_text('0\n')

    def qualify(self, report):
        self.path.write_text(json.dumps(report))
        return self.gate.qualify(self.path)

    def test_full_matrix_checks_sources_and_never_qualifies_hardware(self):
        result = self.qualify(self.report)
        self.assertEqual(result['eager_queries'], 186)
        self.assertEqual(result['replay_queries'], 248)
        self.assertFalse(result['hardware_qualified'])
        self.assertFalse(result['performance_qualified'])

    def test_native_incomplete_changed_source_and_wrong_binary_rejected(self):
        for field, value in (('score_layout', 'native'), ('replay_checks', []),
                ('sources', {}), ('sources_after', {}), ('native_sources', {})):
            report = copy.deepcopy(self.report)
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.qualify(report)

    def test_nonzero_exit_rejected_even_when_report_claims_pass(self):
        self.path.with_suffix('.exit-status').write_text('124\n')
        with self.assertRaises(ValueError):
            self.qualify(self.report)


if __name__ == '__main__':
    unittest.main()
