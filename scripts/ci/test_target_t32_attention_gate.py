import copy
import hashlib
from pathlib import Path
import unittest

from target_t32_attention_gate import SOURCES, validate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).parent
        hashes = {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest() for name in SOURCES}
        checks = [dict(capacity=4352, start=start, ticket=ticket, chip=chip, exact=True)
                  for ticket, start in enumerate((4096, 4113, 4320, 4096)) for chip in range(2)]
        self.report = dict(passed=True, rows=32, closed=True, backend='simulator', sources=hashes,
            sources_after=dict(hashes), checks=checks,
            mask_checks=[dict(entry, bundle=bundle) for bundle in range(3) for entry in checks],
            source_checks=[dict(capacity=4352, chip=chip, exact=True) for chip in (0, 1, 0, 1)],
            unpoisoned_replay=[dict(chip=chip, exact=True, nonfinite=0, mismatches=0) for chip in range(2)],
            stale_controls=2, mask_poison_controls=12)

    def test_complete_structure_not_full_request_qualification(self):
        result = validate(self.report, self.directory)
        self.assertEqual(result['replay_checks'], 8)
        self.assertFalse(result['full_request_qualified'])

    def test_missing_or_failed_checks(self):
        for field in ('checks', 'mask_checks', 'source_checks', 'unpoisoned_replay'):
            for missing in (False, True):
                report = copy.deepcopy(self.report)
                if missing:
                    report[field].pop()
                else:
                    report[field][0]['exact'] = False
                with self.subTest(field=field, missing=missing), self.assertRaises(ValueError):
                    validate(report, self.directory)

    def test_source_change_rejected(self):
        report = copy.deepcopy(self.report)
        report['sources']['attention_replay.py'] = '0' * 64
        report['sources_after'] = dict(report['sources'])
        with self.assertRaises(ValueError):
            validate(report, self.directory)

    def test_duplicate_bundle_cannot_replace_missing_bundle(self):
        report = copy.deepcopy(self.report)
        report['mask_checks'][-1] = dict(report['mask_checks'][7])
        with self.assertRaises(ValueError):
            validate(report, self.directory)

    def test_wrong_status_and_missing_negative_controls(self):
        for field, value in (('passed', False), ('closed', False), ('backend', 'hardware'),
                             ('stale_controls', 0), ('mask_poison_controls', 0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(dict(self.report, **{field: value}), self.directory)


if __name__ == '__main__':
    unittest.main()
