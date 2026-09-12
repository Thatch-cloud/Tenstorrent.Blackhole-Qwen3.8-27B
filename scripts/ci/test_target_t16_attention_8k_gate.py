import copy
import hashlib
from pathlib import Path
import unittest

from target_t16_attention_8k_gate import SOURCES, validate, validate_request_option


class GateTests(unittest.TestCase):
    def test_request_option_rejects_unqualified_geometry(self):
        valid = dict(rows=16, position=8192, remaining=256, replay=True, norm_batch=True,
                     native_sampling=True, group_rows=4, short_context=False)
        validate_request_option(True, **valid)
        for field, value in (('rows', 32), ('position', 8448), ('remaining', 257),
                             ('replay', False), ('norm_batch', False), ('native_sampling', False),
                             ('group_rows', 8), ('short_context', True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_request_option(True, **dict(valid, **{field: value}))

    def setUp(self):
        self.directory = Path(__file__).parent
        hashes = {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest() for name in SOURCES}
        checks = [dict(capacity=8448, start=start, ticket=ticket, chip=chip, exact=True)
                  for ticket, start in enumerate((8192, 8209, 8432, 8192)) for chip in range(2)]
        self.report = dict(passed=True, closed=True, backend='simulator', sources=hashes,
            sources_after=dict(hashes), checks=checks, mask_checks=copy.deepcopy(checks + checks),
            source_checks=[dict(capacity=8448, chip=chip, exact=True) for chip in (0, 1, 0, 1)],
            unpoisoned_replay=[dict(chip=chip, exact=True, nonfinite=0, mismatches=0) for chip in range(2)],
            stale_controls=2, mask_poison_controls=8)

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

    def test_wrong_status_and_missing_negative_controls(self):
        for field, value in (('passed', False), ('closed', False), ('backend', 'hardware'),
                             ('stale_controls', 0), ('mask_poison_controls', 0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(dict(self.report, **{field: value}), self.directory)


if __name__ == '__main__':
    unittest.main()
