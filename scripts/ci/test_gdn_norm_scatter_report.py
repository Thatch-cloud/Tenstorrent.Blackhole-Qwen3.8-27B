"""Reject incomplete or weakened isolated simulator evidence."""

import copy
import hashlib
from pathlib import Path
import unittest

from gdn_norm_scatter_report import validate


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).parent
        names = ('gdn-norm-scatter-probe.py', 'gdn_norm_scatter.py', 'gdn_vsplit_norm_batch.py',
                 'gdn_vsplit.py', 'attention_batch.py')
        hashes = {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest() for name in names}
        self.report = dict(passed=True, closed=True, backend='simulator', output_poisoned=True,
            before=hashes, after=dict(hashes), checks=[], replay_checks=[], poison_checks=[])
        for rows in (1, 16, 32):
            for case in range(2):
                for chip in range(2):
                    record = dict(rows=rows, case=case, chip=chip, exact=True, unchanged=True,
                                  padding_zero=True, written=True, finite=True)
                    self.report['checks'].append(record)
                    for replay in range(2):
                        self.report['replay_checks'].append(dict(record, replay=replay, stale_detected=True))
                self.report['poison_checks'].extend(dict(rows=rows, case=case, verified=True) for unused in range(5))

    def test_complete_structure(self):
        self.assertEqual(validate(self.report, self.directory)['replay_checks'], 24)

    def test_each_failed_boolean_is_rejected(self):
        for field, flags in (('checks', ('exact', 'unchanged', 'padding_zero', 'written', 'finite')),
                             ('replay_checks', ('exact', 'unchanged', 'padding_zero', 'written', 'stale_detected')),
                             ('poison_checks', ('verified',))):
            for flag in flags:
                report = copy.deepcopy(self.report)
                report[field][0][flag] = False
                with self.subTest(field=field, flag=flag), self.assertRaises(ValueError):
                    validate(report, self.directory)

    def test_missing_or_duplicate_evidence(self):
        for field in ('checks', 'replay_checks', 'poison_checks'):
            for duplicate in (False, True):
                report = copy.deepcopy(self.report)
                report[field].pop()
                if duplicate:
                    report[field].append(report[field][0])
                with self.subTest(field=field, duplicate=duplicate), self.assertRaises(ValueError):
                    validate(report, self.directory)

    def test_source_drift(self):
        for same_after in (False, True):
            report = copy.deepcopy(self.report)
            report['before']['gdn_norm_scatter.py'] = '0' * 64
            if same_after:
                report['after'] = dict(report['before'])
            with self.assertRaises(ValueError):
                validate(report, self.directory)

    def test_nonqualified_status(self):
        for field, value in (('passed', False), ('closed', False), ('backend', 'hardware'), ('output_poisoned', False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(dict(self.report, **{field: value}), self.directory)


if __name__ == '__main__':
    unittest.main()
