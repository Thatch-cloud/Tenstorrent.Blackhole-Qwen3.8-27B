from pathlib import Path
import unittest

from mlp_weight_read_order import accesses, issue_model, remove, transform


class WeightReadOrderTests(unittest.TestCase):
    def test_only_issue_order_changes_and_source_round_trips(self):
        source = Path(__file__).with_name('fused_1d_weights.cpp').read_text()
        candidate = transform(source)
        self.assertEqual(remove(candidate), source)
        for operation in ('cb_reserve_back', 'noc_async_read_tile', 'noc_async_read_barrier',
                'cb_push_back', 'cb_wait_front', 'noc_async_write_tile', 'cb_pop_front'):
            self.assertEqual(candidate.count(operation), source.count(operation))
        with self.assertRaises(ValueError):
            transform(candidate)

    def test_every_page_and_zero_padding_destination_preserved(self):
        pages = []
        for worker in range(91):
            for block in range(20):
                before, after = accesses(worker, block, False), accesses(worker, block, True)
                self.assertEqual(sorted(before, key=lambda entry: entry['destination_tile']),
                    sorted(after, key=lambda entry: entry['destination_tile']))
                self.assertEqual(sorted(entry['destination_tile'] for entry in after), list(range(48)))
                self.assertEqual(sum(entry['page'] is None for entry in after), 16 if worker == 90 else 0)
                pages.extend(entry['page'] for entry in after if entry['page'] is not None)
        self.assertEqual(sorted(pages), list(range(160 * 544)))

    def test_synchronized_eight_bank_hypothesis_and_geometry_rejection(self):
        self.assertEqual([max(counts.values()) for counts in issue_model(False)], [23] * 6)
        self.assertEqual([max(counts.values()) for counts in issue_model(True)], [12] * 6)
        self.assertTrue(all(len(counts) == 8 for counts in issue_model(True)))
        for arguments in ((91, 0, True), (0, 20, True), (True, 0, True), (0, 0, 1)):
            with self.assertRaises(ValueError):
                accesses(*arguments)
