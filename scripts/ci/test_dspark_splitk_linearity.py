import unittest

import torch

from dspark_splitk_linearity import COEFFICIENTS, diagnostic_scope, summarize
import dspark_attention_value_diagnostics as baseline


class LinearityTests(unittest.TestCase):
    def test_only_active_values_change_and_scope_restores(self):
        original = dict(history_value=torch.full((2, 4, 64, 128), 8192.),
            query_value=torch.full((2, 4, 32, 128), -8192.), query=torch.ones(1))
        previous = baseline.KINDS
        with diagnostic_scope():
            for kind, (oldest, last) in COEFFICIENTS.items():
                values = baseline.diagnostic_fixture(original, kind, 32, 15)
                self.assertTrue(torch.all(values['history_value'][:, :, 0] == oldest))
                self.assertTrue(torch.all(values['query_value'][:, :, 14] == last))
                self.assertTrue(torch.all(values['history_value'][:, :, 32:] == 8192))
                self.assertTrue(torch.all(values['query_value'][:, :, 15:] == -8192))
                self.assertIs(values['query'], original['query'])
        self.assertEqual(baseline.KINDS, previous)
        self.assertTrue(torch.all(original['history_value'] == 8192))

    def test_reports_deviation_without_claiming_qualification(self):
        records = [dict(kind=kind, chip=0, actual_by_head_row=[[value]]) for kind, value in
            (('oldest', .125), ('last_proposal', .75), ('contrast_scaled', -39.5))]
        result = summarize(records)
        self.assertEqual(result[0]['max_linearity_residual'], .5)
        self.assertFalse(result[0]['qualification'])


if __name__ == '__main__':
    unittest.main()
