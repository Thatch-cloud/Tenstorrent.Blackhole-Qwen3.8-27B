import importlib.util
from pathlib import Path
import unittest

import torch


class DSparkNoiseFixtureTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location('dspark_noise_fixture',Path(__file__).with_name('dspark-noise-probe.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_seven_exact_lookup_rows_are_followed_by_25_zero_rows(self):
        module = self.module()
        packed,identifiers,expected = module.fixtures()
        self.assertEqual(tuple(packed.shape),(2,1,64,2560))
        table = torch.cat((packed[0,0],packed[1,0]),dim=-1)
        for tokens,output in zip(identifiers,expected,strict=True):
            self.assertEqual(tuple(tokens.shape),(1,7))
            self.assertEqual(tuple(output.shape),(1,1,32,5120))
            self.assertTrue(torch.equal(output[:,:,:7],table[tokens].unsqueeze(1)))
            self.assertEqual(int(torch.count_nonzero(output[:,:,7:])),0)
            self.assertTrue(bool((output[:,:,:7]>0).all()))
        self.assertEqual(module.REPLAYS,(0,1,0))
        self.assertEqual(sum(module.COUNTS.values()),44)

    def test_rank_row_and_stale_input_corruption_are_distinguishable(self):
        packed,identifiers,expected = self.module().fixtures()
        self.assertFalse(torch.equal(packed[0],packed[1]))
        self.assertFalse(torch.equal(expected[0],expected[0].roll(2560,dims=-1)))
        self.assertFalse(torch.equal(expected[0],expected[0].roll(-1,dims=2)))
        self.assertFalse(torch.equal(expected[0],expected[1]))


if __name__=='__main__':
    unittest.main()
