import unittest

from dspark_vocabulary_fixture import inputs


class DSparkVocabularyFixtureTests(unittest.TestCase):
    def test_all_columns_and_query_zero_survive_packing(self):
        import torch

        for pattern in range(3):
            fixture = inputs(pattern)
            self.assertEqual(tuple(fixture['packed'].shape),(2,1,32,124160))
            reconstructed = torch.cat(fixture['packed'].chunk(2,dim=0),dim=-1)
            self.assertTrue(torch.equal(reconstructed,fixture['full_logits']))
            self.assertTrue(torch.equal(reconstructed[:,:,:7].float(),fixture['base_logits']))
            swapped = torch.cat(fixture['packed'].chunk(2,dim=0)[::-1],dim=-1)
            self.assertFalse(torch.equal(swapped,fixture['full_logits']))
            self.assertFalse(torch.equal(reconstructed[:,:,1:8].float(),fixture['base_logits']))
            self.assertTrue(torch.isfinite(reconstructed).all())

    def test_patterns_change_output_and_reject_undeclared_inputs(self):
        import torch

        first,second,third = (inputs(pattern) for pattern in range(3))
        self.assertFalse(torch.equal(first['base_logits'],second['base_logits']))
        self.assertFalse(torch.equal(first['base_logits'],third['base_logits']))
        for invalid in (-1,3,True,0.0):
            with self.assertRaises(ValueError):
                inputs(invalid)


if __name__ == '__main__':
    unittest.main()
