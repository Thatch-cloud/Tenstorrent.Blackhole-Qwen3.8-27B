import unittest

import torch

from dspark_backend_attribution import fp32_attention


class BackendAttributionTests(unittest.TestCase):
    def test_fp32_gqa_uses_all_keys_and_matches_declared_reference(self):
        generator = torch.Generator().manual_seed(382710)
        values = [torch.randn(shape,generator=generator).bfloat16() for shape in
            ((1,8,7,128),(1,2,39,128),(1,2,39,128))]
        expected = torch.nn.functional.scaled_dot_product_attention(values[0].float(),
            values[1].float().repeat_interleave(4,1),values[2].float().repeat_interleave(4,1),is_causal=False).bfloat16()
        self.assertTrue(torch.equal(fp32_attention(*values),expected))
        changed = [value.clone() for value in values]
        changed[2][:,:,0] += 4
        self.assertFalse(torch.equal(fp32_attention(*changed),expected))

    def test_bad_heads_dtypes_and_nonfinite_operands_are_rejected(self):
        for query,key,value in ((torch.zeros(1,7,7,128),torch.zeros(1,2,39,128),torch.zeros(1,2,39,128)),
                (torch.zeros(1,8,7,128),torch.zeros(1,2,39,128),torch.zeros(1,2,39,128)),
                (torch.full((1,8,7,128),float('nan'),dtype=torch.bfloat16),
                 torch.zeros(1,2,39,128,dtype=torch.bfloat16),torch.zeros(1,2,39,128,dtype=torch.bfloat16))):
            with self.assertRaises(ValueError):
                fp32_attention(query,key,value)


if __name__ == '__main__':
    unittest.main()
