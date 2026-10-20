"""CPU tests of the numerics emulation (ksplit_emulation.py) and of the doc's a-priori reading of the bound it supports. Needs torch (skips without). These are properties of a MODEL of the two
arithmetics, not of the card: they pin what the waiver card says the emulation shows, and that the emulation has teeth (a wrong merge is far outside the bound).

Run from this directory:  python3 -B -m unittest test_ksplit_emulation
"""

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import torch
except ImportError:      # pragma: no cover
    torch = None

if torch is not None:
    import ksplit_emulation as emulation
    import ksplit_numerics as numerics


@unittest.skipIf(torch is None, 'torch not installed')
class EmulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.short = [emulation.measure(8, 2, seed, 1.0, torch) for seed in range(2)]
        cls.mid = [emulation.measure(64, 2, seed, 1.0, torch) for seed in range(2)]

    def test_a_correct_split_is_inside_the_context_scaled_bound(self):
        for blocks, rows in ((8, self.short), (64, self.mid)):
            bound = numerics.bound_for(blocks * emulation.BLOCK)
            for row in rows:
                self.assertEqual(numerics.within_bound(row['ksplit_vs_oneq'], bound), [], (blocks, row['ksplit_vs_oneq']))

    def test_the_split_is_no_further_from_the_float_reference_than_the_sequential_call(self):
        for rows in (self.short, self.mid):
            for row in rows:
                self.assertEqual(numerics.noise_problems(row['ksplit_vs_ref'], row['oneq_vs_ref']), [], row)

    def test_the_sequential_calls_own_noise_grows_with_the_context(self):
        short = sum(row['oneq_vs_ref']['rms_ulps'] for row in self.short) / len(self.short)
        mid = sum(row['oneq_vs_ref']['rms_ulps'] for row in self.mid) / len(self.mid)
        self.assertGreater(mid, 2.0 * short, (short, mid))
        # and the base bound alone would have failed a correct split at 8k: the reason the bound grows
        worst = max(row['ksplit_vs_oneq']['max_ulps'] for row in self.mid)
        self.assertGreater(worst, 0.5 * numerics.BOUND['max_ulps'])

    def test_the_split_gets_closer_to_the_truth_than_the_sequential_call_as_the_context_grows(self):
        ratio = lambda rows: (sum(row['ksplit_vs_ref']['rms_ulps'] for row in rows) / sum(row['oneq_vs_ref']['rms_ulps'] for row in rows))
        self.assertLess(ratio(self.mid), ratio(self.short))
        self.assertLess(ratio(self.mid), 1.0)

    def test_a_wrong_merge_weight_is_far_outside_the_bound_on_peaky_rows_and_invisible_on_flat_ones(self):
        genuine = emulation.ring_merge

        def inverted(prev_out, prev_lse, cur_out, cur_lse, torch_module):
            return genuine(prev_out, cur_lse, cur_out, prev_lse, torch_module)       # the weight the wrong way round
        bound = numerics.bound_for(64 * emulation.BLOCK)
        emulation.ring_merge = inverted
        try:
            peaky = emulation.measure(64, 2, 0, 4.0, torch)['ksplit_vs_oneq']
            flat = emulation.measure(64, 2, 0, 1.0, torch)['ksplit_vs_oneq']
        finally:
            emulation.ring_merge = genuine
        self.assertTrue(numerics.within_bound(peaky, bound), peaky)
        self.assertGreater(peaky['rms_ulps'], 5.0 * bound['rms_ulps'])
        # equal slice weights (flat softmax): the swap changes sigmoid(0.0 +- small) by nothing. The card-M matrix therefore carries the PEAKY variant, which is where a wrong weight shows.
        self.assertLess(flat['rms_ulps'], bound['rms_ulps'])

    def test_a_lost_slice_is_far_outside_the_bound(self):
        real = emulation.one_pass

        def dropping(q, k, v, blocks, torch_module, stats_bf16, fp32_accumulator=False):
            if not stats_bf16 and len(blocks) and blocks[0] > 0:                         # a later slice contributes only its first block
                blocks = range(blocks[0], blocks[0] + 1)
            return real(q, k, v, blocks, torch_module, stats_bf16, fp32_accumulator)
        emulation.one_pass = dropping
        try:
            row = emulation.measure(64, 2, 0, 1.0, torch)['ksplit_vs_oneq']
        finally:
            emulation.one_pass = real
        bound = numerics.bound_for(64 * emulation.BLOCK)
        self.assertTrue(numerics.within_bound(row, bound))
        self.assertGreater(row['rms_ulps'], 10.0 * bound['rms_ulps'])

    def test_the_slice_ranges_cover_the_blocks_once(self):
        import ksplit_planner as planner
        for blocks, slices in ((8, 2), (64, 4), (33, 4), (5, 4)):
            eff = planner.effective_slices(blocks, slices)
            covered = []
            for s in range(eff):
                begin = planner.slice_begin(blocks, s, eff)
                end = planner.slice_begin(blocks, s + 1, eff) if s + 1 < eff else blocks
                covered += list(range(begin, end))
            self.assertEqual(covered, list(range(blocks)), (blocks, slices))

    def test_the_table_prints(self):
        text = emulation.table([8], [2], 1, [1.0])
        self.assertIn('| 8 (1k) | 2 (2) | 1.0 |', text)


if __name__ == '__main__':
    unittest.main()
