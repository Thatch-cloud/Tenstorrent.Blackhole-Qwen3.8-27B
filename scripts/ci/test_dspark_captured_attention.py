import importlib.util
from pathlib import Path
import unittest

import torch

from dspark_attention import full_mask


spec = importlib.util.spec_from_file_location('captured_attention_probe',Path(__file__).with_name('dspark-attention-captured-probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class CapturedAttentionTests(unittest.TestCase):
    def fixture(self):
        values = [torch.zeros(shape,dtype=torch.bfloat16) for shape in ((2,16,32,128),(2,4,64,128),(2,4,64,128))]
        for value in values:
            value[1] = 7
        return [*values,full_mask(32,key_multiple=64)]

    def test_reference_uses_each_captured_chip_and_all_kv_heads(self):
        values = self.fixture()
        values[1][1,2] = 9
        query,keys,heads = probe.reference_inputs(values,1)
        self.assertEqual(query.dtype,torch.float32)
        self.assertTrue(torch.equal(query,values[0][1:2].float()))
        self.assertTrue(torch.equal(keys,values[1][1:2].float().repeat_interleave(4,1)))
        self.assertTrue(torch.equal(heads,values[2][1:2].float().repeat_interleave(4,1)))
        self.assertEqual(tuple(keys.shape),(1,16,64,128))

    def test_reference_rejects_wrong_chip_shape_dtype_and_mask(self):
        with self.assertRaises(ValueError):
            probe.reference_inputs(self.fixture(),True)
        for index,replacement in ((0,torch.zeros(1,16,32,128,dtype=torch.bfloat16)),
                (1,torch.zeros(2,4,64,128)),(3,torch.zeros(1,1,32,64,dtype=torch.bfloat16))):
            values = self.fixture()
            values[index] = replacement
            with self.assertRaises(ValueError):
                probe.reference_inputs(values,0)


if __name__ == '__main__':
    unittest.main()
