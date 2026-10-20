"""CPU tests of the K-split protocol model (ks_protocol_model.py): S chains per q chunk and the slice-to-slice partial hand-off, deadlock freedom, the NoC pattern, the planted hang and the
checker's own teeth.

Run from this directory:  python3 -B -m unittest test_ksplit_protocol
"""

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / 'sdpa_prefill_chain'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ks_protocol_model as model  # noqa: E402
import ksplit_planner as planner  # noqa: E402
import pf_protocol_model as chain  # noqa: E402


class OkTests(unittest.TestCase):
    def test_every_shape_terminates_with_exact_content_and_clean_semaphores(self):
        failures, runs = model.sweep(range(25), starts=(0, 1, 2, 3, 4, 7, 8, 15, 16, 33, 64), qs=(0, 3, 7, 15), slices=(2, 3, 4), variants=('ok',))
        self.assertGreater(runs, 1000)
        self.assertEqual(failures['ok'], [], failures['ok'][:2])

    def test_the_engaged_production_shapes_at_long_context(self):
        for C, q, slices in ((1000, 7, 2), (2000, 3, 4), (260, 0, 4), (33, 7, 2), (65, 11, 2)):
            for seed in range(3):
                result = model.simulate(C, q, seed, slices, 6)
                self.assertEqual(model.check_ok(result), [], (C, q, slices, seed))
                eff = planner.effective_slices(planner.blocks_of(C * 128, q), slices)
                self.assertEqual(result.final['eff'], eff)

    def test_each_k_block_is_read_once_over_the_slices_and_forwarded_down_every_chain(self):
        for C, q, slices in ((33, 7, 2), (40, 3, 4), (16, 0, 3)):
            result = model.simulate(C, q, 1, slices, 6)
            blocks = planner.blocks_of(C * 128, q)
            eff = planner.effective_slices(blocks, slices)
            self.assertEqual(sorted(k for reads in result.reads.values() for k in reads), list(range(blocks)), 'each block is read from DRAM once per q chunk, as in oneq')
            for s in range(eff):
                begin, end = planner.slice_range(blocks, s, slices)
                self.assertEqual(result.reads[s], list(range(begin, end)))
                self.assertEqual(result.chain_writes[s], 2 * (end - begin) * (6 - 1), 'K and V forwarded down a 6-core chain, 5 links')
            for s in range(eff, slices):
                self.assertEqual((result.reads[s], result.chain_writes[s]), ([], 0), 'an idle slice moves nothing')

    def test_the_handoff_pattern(self):
        result = model.simulate(33, 7, 2, 4, 6)
        self.assertEqual(result.final['eff'], 4)
        self.assertEqual(len(result.handoffs), 3 * 6)
        self.assertEqual({(src // 6, dst // 6) for src, dst, _bytes in result.handoffs}, {(0, 1), (1, 2), (2, 3)})
        self.assertTrue(all(src % 6 == dst % 6 for src, dst, _bytes in result.handoffs), 'the same head')
        self.assertEqual({nbytes for _src, _dst, nbytes in result.handoffs}, {sum(planner.handoff_bytes())})
        self.assertEqual(result.final_payloads, {head: (0, planner.blocks_of(33 * 128, 7)) for head in range(6)})

    def test_one_slice_means_no_handoff_and_the_oneq_pattern(self):
        for C, q in ((0, 0), (0, 3), (1, 1)):                 # B < 8: a single slice takes the whole chunk
            result = model.simulate(C, q, 0, 4, 6)
            self.assertEqual(result.final['eff'], 1)
            self.assertEqual(result.handoffs, [])
            self.assertEqual(model.check_ok(result), [])
            self.assertEqual(result.reads[0], list(range(planner.blocks_of(C * 128, q))))

    def test_other_member_counts(self):
        for members in (1, 2, 3):
            failures, runs = model.sweep(range(8), starts=(8, 33), qs=(3, 7), slices=(2, 4), members=members, variants=('ok',))
            self.assertEqual(failures['ok'], [], members)

    def test_the_chain_checker_it_extends_still_passes(self):
        self.assertEqual(chain.check_ok(chain.simulate(16, 3, 1, 'ok')), [])


class HangTests(unittest.TestCase):
    def test_the_planted_hang_blocks_exactly_the_hand_off_waits(self):
        failures, runs = model.sweep(range(10), starts=(8, 16, 33, 64), qs=(3, 7), slices=(2, 3, 4), variants=('hang',))
        self.assertGreater(runs, 100)
        self.assertEqual(failures['hang'], [], failures['hang'][:2])

    def test_the_blocked_set(self):
        result = model.simulate(33, 7, 0, 2, 6, variant='hang')
        self.assertFalse(result.terminated)
        self.assertEqual(result.errors, [])
        expected = {}
        for head in range(6):
            expected['writer %d' % head] = 'handoff_credit'
            expected['reader %d' % (6 + head)] = 'handoff_valid'
            expected['consumer %d' % (6 + head)] = 'prev_full'
            expected['writer %d' % (6 + head)] = 'out_full'
        self.assertEqual(result.blocked, expected)
        self.assertEqual(model.hang_blocked(2, 6), expected)

    def test_one_slice_has_nothing_to_hang(self):
        result = model.simulate(0, 0, 0, 4, 6, variant='hang')
        self.assertTrue(result.terminated)


class TeethTests(unittest.TestCase):
    def caught(self, mutation, starts=(8, 16, 33), qs=(3, 7), slices=(2, 4), seeds=range(8)):
        failures, runs = model.sweep(seeds, starts=starts, qs=qs, slices=slices, mutation=mutation)
        return len(failures['ok']), runs, failures['ok']

    def test_valid_before_ack_is_caught(self):
        count, runs, items = self.caught('valid_before_ack')
        self.assertGreater(count, runs // 4, items[:1])

    def test_a_write_without_the_credit_is_caught(self):
        count, runs, items = self.caught('no_credit_wait')
        self.assertEqual(count, runs, 'a credit never consumed leaves the semaphore high on every run')

    def test_a_missing_dummy_pair_is_a_deadlock(self):
        count, runs, items = self.caught('dummy_missing', starts=(8, 16, 33, 64))
        self.assertGreater(count, 0)
        self.assertTrue(all('did not terminate' in problems[0] for _c, _q, _s, _seed, problems in items), items[:1])

    def test_differing_range_arithmetic_is_caught(self):
        count, runs, items = self.caught('eff_mismatch')
        self.assertEqual(count, runs)

    def test_an_unknown_variant_or_mutation_is_refused(self):
        with self.assertRaises(ValueError):
            model.simulate(8, 3, 0, 2, 6, variant='nope')
        with self.assertRaises(ValueError):
            model.simulate(8, 3, 0, 2, 6, mutation='nope')


if __name__ == '__main__':
    unittest.main()
