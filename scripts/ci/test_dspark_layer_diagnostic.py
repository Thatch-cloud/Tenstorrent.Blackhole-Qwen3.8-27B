from pathlib import Path
import tempfile
import unittest

import torch

from dspark_layer_diagnostic import error_summary, load_capture


class DSparkLayerDiagnosticTests(unittest.TestCase):
    def test_numeric_success_does_not_hide_changed_bf16_bits(self):
        expected = torch.tensor([1.,2.],dtype=torch.bfloat16)
        actual = expected.clone()
        actual[0] = 1.0078125
        result = error_summary(actual,expected)
        self.assertTrue(result['passed'])
        self.assertFalse(result['bitwise_exact'])
        self.assertEqual(result['different_bits'],1)
        self.assertGreater(result['relative_l2'],0.)

    def test_zero_reference_norm_does_not_report_a_false_zero_error(self):
        expected = torch.zeros(2,dtype=torch.bfloat16)
        self.assertEqual(error_summary(expected,expected)['relative_l2'],0.)
        self.assertIsNone(error_summary(torch.ones_like(expected),expected)['relative_l2'])

    def test_unpinned_capture_rejects_before_loading_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/'report.json'
            report.write_text('{}')
            with self.assertRaises(ValueError):
                load_capture(report,Path(directory)/'not-opened')


if __name__ == '__main__':
    unittest.main()
