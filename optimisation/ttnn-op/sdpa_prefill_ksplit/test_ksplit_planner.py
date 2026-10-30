"""CPU tests of the K-split planner (ksplit_planner.py): the slice arithmetic, the slice-major core assignment, the chains, the refusals, the time model.

Run from this directory:  python3 -B -m unittest test_ksplit_planner
"""

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
CI = HERE.parents[2] / 'scripts' / 'ci'
for path in (HERE, HERE.parent / 'sdpa_prefill_oneq', CI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ksplit_planner as planner  # noqa: E402
import oneq_planner as oneq  # noqa: E402


class SliceArithmeticTests(unittest.TestCase):
    def test_effective_slices_keeps_four_blocks_in_a_slice(self):
        self.assertEqual([planner.effective_slices(b, 4) for b in (1, 3, 4, 7, 8, 11, 12, 15, 16, 100)], [1, 1, 1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([planner.effective_slices(b, 2) for b in (1, 7, 8, 9, 1000)], [1, 1, 2, 2, 2])
        self.assertEqual(planner.effective_slices(5000, 1), 1)

    def test_ranges_are_contiguous_disjoint_cover_all_blocks_and_the_last_holds_the_diagonal(self):
        for blocks in list(range(1, 70)) + [127, 128, 129, 1000, 1985, 2000]:
            for slices in (2, 3, 4):
                spans = planner.ranges(blocks, slices)
                self.assertEqual(spans[0][0], 0)
                self.assertEqual(spans[-1][1], blocks, 'the last slice ends on the diagonal block %d' % (blocks - 1))
                for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                    self.assertEqual(a1, b0)
                self.assertTrue(all(end > begin for begin, end in spans), 'every active slice has a block')
                self.assertTrue(all(end - begin >= planner.MIN_BLOCKS_PER_SLICE for begin, end in spans) or len(spans) == 1)
                sizes = [end - begin for begin, end in spans]
                self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_an_idle_slice_has_no_range(self):
        self.assertIsNone(planner.slice_range(5, 1, 4))
        self.assertEqual(planner.slice_range(5, 0, 4), (0, 5))
        self.assertIsNone(planner.slice_range(20, 4, 4))
        self.assertEqual(planner.slice_range(20, 3, 4), (15, 20))

    def test_blocks_of_matches_the_oneq_planner_block_count(self):
        for context in (0, 128, 4096, 131072):
            for q in (0, 5, 15):
                self.assertEqual(planner.blocks_of(context, q), oneq.blocks(q, context))


class SliceCountTests(unittest.TestCase):
    def test_the_tp4_rows(self):
        self.assertEqual({rows: planner.split_k_slices(rows) for rows in (128, 256, 512, 1024, 2048)}, {128: 4, 256: 4, 512: 4, 1024: 2, 2048: 1})
        self.assertEqual({rows: planner.split_k_slices(rows, cores=110) for rows in (512, 1024, 2048)}, {512: 4, 1024: 2, 2048: 1})

    def test_the_host_engages_only_the_qualified_rows_with_a_second_slice(self):
        self.assertEqual([rows for rows in (128, 256, 512, 1024, 1536, 2048) if planner.engaged_rows(rows)], [512, 1024])

    def test_the_cap_holds_when_the_grid_could_take_more(self):
        self.assertEqual(planner.split_k_slices(256, cores=130), 4)          # 12 busy cores: ten slices would fit
        self.assertEqual(planner.split_k_slices(256, cores=1300), 4)

    def test_it_agrees_with_the_cost_model(self):
        import prefill_cost_model as model
        for rows in (128, 256, 512, 1024, 2048):
            self.assertEqual(planner.split_k_slices(rows), model.split_k_slices(rows))

    def test_a_grid_without_room_does_not_split(self):
        self.assertEqual(planner.split_k_slices(1024, cores=96), 2)
        self.assertEqual(planner.split_k_slices(1024, cores=95), 1)
        self.assertEqual(planner.split_k_slices(1024, cores=48), 1)


class PlanTests(unittest.TestCase):
    def test_slice_zero_is_the_oneq_layout_and_every_slice_repeats_it(self):
        for rows, slices in ((1024, 2), (512, 4)):
            mine = planner.plan(rows=rows)
            base = oneq.plan(6, 1, rows, 128, (13, 10), oneq=True)
            self.assertEqual((mine['slices'], mine['busy'], mine['total']), (slices, base['total'] * slices, base['total']))
            for core in mine['cores'][:mine['busy']]:
                position = core['core'] % mine['total']
                self.assertEqual(core['slice'], core['core'] // mine['total'])
                self.assertEqual(core['unit'], base['units'][position][0])
            self.assertTrue(all(not core['busy'] for core in mine['cores'][mine['busy']:]))

    def test_chains_are_the_oneq_chains_of_each_slice(self):
        mine = planner.plan(rows=1024)
        base = oneq.plan(6, 1, 1024, 128, (13, 10), oneq=True)
        self.assertEqual(mine['chain_count'], base['chain_count'] * 2)
        self.assertEqual(mine['member_count'], base['member_count'] * 2)
        for (nb, kv_head, q, slice_index), members in mine['chains'].items():
            self.assertEqual(len(members), 6)
            self.assertEqual({mine['cores'][c]['slice'] for c in members}, {slice_index})
            self.assertEqual({mine['cores'][c]['unit'][2] for c in members}, {q})
            self.assertEqual({mine['cores'][c]['unit'][1] // 6 for c in members}, {kv_head})
        # slice 0's groups are the oneq groups, by member
        zero = sorted(tuple(members) for key, members in mine['chains'].items() if key[3] == 0)
        self.assertEqual(zero, sorted(tuple(members) for members in base['chains'].values()))

    def test_the_handoff_partner_is_the_next_slice_same_head_and_q_chunk(self):
        mine = planner.plan(rows=512)
        self.assertEqual(len(mine['edges']), 24 * 3)          # 24 units, S - 1 = 3 hops each
        for sender, receiver in mine['edges']:
            a, b = mine['cores'][sender], mine['cores'][receiver]
            self.assertEqual(receiver, sender + mine['total'])
            self.assertEqual((a['unit'], b['slice']), (b['unit'], a['slice'] + 1))
            self.assertEqual(a['next'], receiver)
            self.assertEqual(b['prev'], sender)
        last = [core for core in mine['cores'] if core['busy'] and core['slice'] == mine['slices'] - 1]
        self.assertTrue(all(core['next'] is None for core in last))
        self.assertTrue(all(core['prev'] is None for core in mine['cores'] if core['busy'] and core['slice'] == 0))

    def test_the_refusals(self):
        self.assertIn('needs at least 2 slices', planner.plan(rows=2048)['refusal'])
        self.assertEqual(planner.plan(rows=1920)['refusal'], planner.ENVELOPE_TEXT)                 # an odd chunk count
        self.assertIn('above the cap', planner.plan(rows=512, slices=5)['refusal'])                # S > 4 asked for
        self.assertIn('needs at least 2 slices', planner.plan(rows=1024, slices=3)['refusal'])    # three slices of 48 chunks need 144 cores
        self.assertIsNone(planner.plan(rows=1024, slices=2)['refusal'])
        self.assertIsNone(planner.plan(rows=512, slices=4)['refusal'])
        self.assertIn('oneq needs one q chunk per core', planner.plan(nqh=12, nkh=2, rows=2048)['refusal'])    # TP2: 192 chunks, refused by oneq first
        self.assertIn('needs at least 2 slices', planner.plan(nqh=12, nkh=2, rows=1024)['refusal'])
        self.assertEqual(planner.plan(rows=1024, grid=(6, 8))['slices'], 1)
        self.assertIn('needs at least 2 slices', planner.plan(rows=1024, grid=(6, 8))['refusal'])

    def test_the_11x10_grid(self):
        self.assertEqual((planner.plan(rows=1024, grid=(11, 10))['slices'], planner.plan(rows=512, grid=(11, 10))['slices']), (2, 4))
        self.assertEqual(planner.plan(rows=512, grid=(8, 8))['slices'], 2)


class TimeModelTests(unittest.TestCase):
    def test_blocks_per_core_follow_the_ranges(self):
        mine = planner.plan(rows=1024)
        context = 65536
        blocks = planner.core_blocks(mine, context)
        for core in mine['cores'][:mine['busy']]:
            span = planner.slice_range(planner.blocks_of(context, core['unit'][2]), core['slice'], 2)
            self.assertEqual(blocks[core['core']], span[1] - span[0])
        self.assertEqual(planner.critical_blocks(mine, context), max(blocks))

    def test_the_lever_is_a_win_at_long_context_and_neutral_at_none(self):
        for rows, bound in ((1024, 0.56), (512, 0.32)):
            self.assertLess(planner.layer_ratio(rows, 65536), bound)
            self.assertLess(planner.layer_ratio(rows, 253952), bound)
        self.assertEqual(planner.layer_ratio(2048, 65536), 1.0)               # not engaged
        self.assertGreaterEqual(planner.layer_ratio(512, 0), 1.0)            # B is small: one slice, the overhead of nothing is none (not faster either)
        self.assertLessEqual(planner.layer_ratio(512, 0), 1.0 + 1e-9)

    def test_a_dearer_hop_costs_more(self):
        a = planner.layer_ratio(512, 2048, hop_us=planner.HOP_US)
        b = planner.layer_ratio(512, 2048, hop_us=planner.HOP_US_PESSIMISTIC)
        self.assertLess(a, b)

    def test_the_handoff_is_80_kilobytes(self):
        self.assertEqual(planner.handoff_bytes(), (65536, 16384))


if __name__ == '__main__':
    unittest.main()
