import unittest

import torch

from dspark_layer_rounding import sample_coordinates


class LayerRoundingTests(unittest.TestCase):
    def test_sampling_keeps_failures_and_controls_separate(self):
        expected = torch.zeros(2,8)
        actual = expected.clone()
        actual[0,1:6] = 1
        result = sample_coordinates(actual,expected)
        self.assertEqual(result[:4],[('failure',(0,column)) for column in range(1,5)])
        self.assertEqual(result[4:],[('control',coordinate) for coordinate in ((0,0),(0,6),(0,7),(1,0))])

    def test_all_exact_is_not_reported_as_failing_samples(self):
        values = torch.zeros(2,2)
        self.assertEqual([kind for kind,coordinate in sample_coordinates(values,values)],['control']*4)

    def test_invalid_geometry_and_nonfinite_values_are_rejected(self):
        for actual,expected in ((torch.zeros(2),torch.zeros(2)),(torch.zeros(2,2),torch.zeros(2,3)),
                (torch.full((2,2),float('nan')),torch.zeros(2,2))):
            with self.assertRaises(ValueError):
                sample_coordinates(actual,expected)


if __name__ == '__main__':
    unittest.main()
