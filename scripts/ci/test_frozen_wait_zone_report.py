import unittest
from pathlib import Path
import tempfile
import csv

from frozen_wait_zone_report import EXPECTED, summarize, read_raw_trace_events, RAW_FIELDS


class WaitReportTests(unittest.TestCase):
    def test_raw_reader_preserves_identity_without_clock_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'raw.csv'
            events, _ = self.fixture()
            with path.open('w', newline='') as stream:
                stream.write('ARCH: blackhole, CHIP_FREQ[MHz]: 0, Max Compute Cores: 120\n')
                writer = csv.DictWriter(stream, fieldnames=list(RAW_FIELDS.values()))
                writer.writeheader()
                writer.writerow({column: events[0][field] for field, column in RAW_FIELDS.items()})
            self.assertEqual(list(read_raw_trace_events(path)), events[:1])

    def fixture(self):
        executions = {(chip, 42, 7, 2) for chip in (0, 1)}
        events = []
        for chip, host_id, trace_id, replay in sorted(executions):
            for name, count in EXPECTED.items():
                for core_x in range(count):
                    for phase, cycle in (('ZONE_START', 100), ('ZONE_END', 110)):
                        events.append(dict(chip=chip, host_id=host_id, trace_id=trace_id,
                            replay=replay, zone=name, core_x=core_x, core_y=0,
                            risc='NCRISC', phase=phase, cycle=cycle))
        return events, executions

    def test_complete_samples_without_throughput_claim(self):
        events, executions = self.fixture()
        report = summarize(reversed(events), executions, '')
        self.assertEqual(len(report['samples']), 20)
        self.assertTrue(all(item['cycles'] == 10 for item in report['samples']))
        self.assertIsNone(report['committed_tg'])

    def test_missing_or_duplicate_endpoint_rejected(self):
        events, executions = self.fixture()
        for candidate in (events[1:], events + events[:1]):
            with self.assertRaises(ValueError):
                summarize(candidate, executions, '')

    def test_trace_identity_cannot_be_substituted(self):
        events, executions = self.fixture()
        events[0]['replay'] = 3
        with self.assertRaises(ValueError):
            summarize(events, executions, '')

    def test_missing_chip_or_dropped_markers_rejected(self):
        events, executions = self.fixture()
        with self.assertRaises(ValueError):
            summarize(events, {(0, 42, 7, 2)}, '')
        with self.assertRaises(ValueError):
            summarize(events, executions, 'WARNING: markers were dropped')

    def test_backward_timestamps_rejected(self):
        events, executions = self.fixture()
        events[0]['cycle'] = 111
        with self.assertRaises(ValueError):
            summarize(events, executions, '')


if __name__ == '__main__':
    unittest.main()
