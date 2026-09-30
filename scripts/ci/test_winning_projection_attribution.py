import copy
import unittest

from winning_projection_attribution import PREDECESSORS, separate


def fixture():
    rows = []
    for (operation, cores), (name, count) in PREDECESSORS.items():
        for repeat in range(count):
            for label, workers in ((operation, cores), ('MatmulDeviceOperation', '32'),
                    ('ReduceScatterMinimalAsyncDeviceOperation', '10')):
                start = len(rows) * 20
                rows.append({'OP NAME': label, 'CORE COUNT': workers,
                    'DEVICE KERNEL START CYCLE': str(start), 'DEVICE KERNEL END CYCLE': str(start + 10),
                    'DEVICE KERNEL DURATION [ns]': '1000'})
    return rows


class ProjectionAttributionTests(unittest.TestCase):
    def test_all_output_families_separate_without_double_counting(self):
        result = separate(list(reversed(fixture())))
        self.assertEqual({name: value['calls'] for name, value in result.items()}, dict(PREDECESSORS.values()))
        self.assertAlmostEqual(sum(value['summed_ms'] for value in result.values()), .128)

    def test_missing_or_unknown_neighborhood_rejected(self):
        for mutate in (lambda rows: rows.pop(),
                lambda rows: rows[0].update({'CORE COUNT': '98'}),
                lambda rows: rows[2].update({'OP NAME': 'Other'}),
                lambda rows: rows[1].update({'DEVICE KERNEL DURATION [ns]': 'nan'}),
                lambda rows: rows[0].update({'DEVICE KERNEL END CYCLE': '21'})):
            rows = copy.deepcopy(fixture())
            mutate(rows)
            with self.assertRaises(ValueError):
                separate(rows)


if __name__ == '__main__':
    unittest.main()
