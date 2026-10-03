import copy
import hashlib
from pathlib import Path
import unittest

from tensor_bit_compare_gate import SOURCES, validate


class TensorBitCompareGateTests(unittest.TestCase):
    def fixture(self):
        root = Path(__file__).parent
        sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
        checks = []
        for mode, cases in (('eager', (0, 1, 2)), ('replay', (0, 1, 2, 0))):
            for width in (32, 2048):
                for ordinal, case in enumerate(cases):
                    for chip in (0, 1):
                        checks.append(dict(mode=mode, width=width, case=case, chip=chip,
                            repetition=None if mode == 'eager' else ordinal, exact=True,
                            inputs_unchanged=True, poisoned_counters_replaced=True))
        return dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator',
            sources=sources, sources_after=dict(sources), checks=checks)

    def test_complete_controls_qualify_only_comparator(self):
        result = validate(self.fixture(), Path(__file__).parent, '0')
        self.assertEqual(result['checks'], 28)
        self.assertFalse(result['publication_qualified'])

    def test_incomplete_or_invalid_controls_rejected(self):
        original = self.fixture()
        for index in range(28):
            report = copy.deepcopy(original)
            report['checks'].pop(index)
            with self.assertRaises(ValueError):
                validate(report, Path(__file__).parent, '0')
        for field in ('exact', 'inputs_unchanged', 'poisoned_counters_replaced'):
            report = copy.deepcopy(original)
            report['checks'][0][field] = False
            with self.assertRaises(ValueError):
                validate(report, Path(__file__).parent, '0')
        for field, value in (('closed_cleanly', False), ('sources_after', {}), ('stage', 'open')):
            report = copy.deepcopy(original)
            report[field] = value
            with self.assertRaises(ValueError):
                validate(report, Path(__file__).parent, '0')
        with self.assertRaises(ValueError):
            validate(original, Path(__file__).parent, '143')
