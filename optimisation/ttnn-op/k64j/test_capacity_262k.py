"""The card harnesses keyed by served capacity (E2): K2/X7 design families, the reader's named R2 families and the verdict lines at
131,328 (unchanged) and 262,144 (the full window as a sixth family: 2,030 K2 tickets, 600 X7 comparisons).

    py -3.11 -B -m unittest test_capacity_262k      (from this directory; scripts/ci on the path)
"""

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
CI = HERE.parents[2] / 'scripts' / 'ci'
for _path in (str(HERE), str(OPS / 'k64j_probe'), str(OPS / 'sdpa_decode_qwen'), str(CI)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import extent_reader_card_b as reader_b  # noqa: E402
import k64j_card_b as card_b  # noqa: E402
import packed_any_admission as admission  # noqa: E402


def k2_comparisons(capacity, extents=None):
    extents = card_b.cb2_extents_for(capacity) if extents is None else extents
    rows = []
    for seed in card_b.K2_DESIGN_SEEDS:
        for variant in card_b.K2_DESIGN_VARIANTS:
            for start in range(card_b.K2_SWEEP[0], card_b.K2_SWEEP[1] + 1):
                rows.append(dict(kind='k2_native_vs_extent', seed=seed, variant=variant, ticket='sweep', start=start))
            for extent in extents:
                for offset in card_b.CB2_STARTS:
                    rows.append(dict(kind='k2_native_vs_extent', seed=seed, variant=variant, ticket='family', extent=extent,
                                     start=extent - card_b.K_CHUNK + offset))
    return rows


class K2CoverageTests(unittest.TestCase):
    def test_the_design_set_by_capacity(self):
        self.assertEqual(card_b.cb2_extents_for(131328), card_b.CB2_EXTENTS)
        self.assertEqual(card_b.cb2_extents_for(262144), card_b.CB2_EXTENTS + (262144,))
        self.assertEqual(card_b.CB2_EXTENTS, (2304, 4352, 16640, 65792, 131328), 'the 131k set did not move')
        self.assertIsNone(card_b.cb2_extents_for(200000))
        self.assertEqual(card_b.CAPACITY_262K, 262144)

    def test_the_ticket_counts_are_the_admissions_floors(self):
        for capacity, tickets in ((131328, 1980), (262144, 2030)):
            coverage = card_b.k2_coverage(dict(capacity=capacity, comparisons=k2_comparisons(capacity)))
            self.assertEqual((coverage['full'], coverage['covered'], coverage['design']), (True, tickets, tickets))
            self.assertEqual(admission.capacity_design(capacity)['k2_tickets'], tickets)
        # a report with no capacity is the default 131,328 one
        coverage = card_b.k2_coverage(dict(comparisons=k2_comparisons(131328)))
        self.assertEqual((coverage['full'], coverage['covered']), (True, 1980))

    def test_a_131k_run_is_short_at_262k_in_the_family_dimension(self):
        coverage = card_b.k2_coverage(dict(capacity=262144, comparisons=k2_comparisons(131328)))
        self.assertFalse(coverage['full'])
        self.assertEqual((coverage['covered'], coverage['design']), (1980, 2030))
        self.assertIn('family', coverage['short'])
        # and the 262k rows over the 131k design are a superset of it: full there
        self.assertTrue(card_b.k2_coverage(dict(capacity=131328, comparisons=k2_comparisons(262144)))['full'])

    def test_the_verdict_line_names_a_non_default_capacity_only(self):
        report = dict(decision=dict(verdict='PASS', first_differing=[], reasons=[], k2='not_run'), comparisons=[], liveness=[])
        self.assertNotIn('capacity=', card_b.verdict_line(dict(report)))
        self.assertNotIn('capacity=', card_b.verdict_line(dict(report, capacity=131328)))
        line = card_b.verdict_line(dict(report, capacity=262144))
        self.assertTrue(line.startswith('K64J_CARD verdict=PASS capacity=262144 '), line)


def reader_report(capacity, families):
    return dict(capacity=capacity, sections=list(reader_b.SECTIONS),
                r1_run={name: list(reader_b.DESIGN_RESIDUES) for name in reader_b.R1_GEOMETRIES},
                r2_families_replayed=families, variants_run=list(reader_b.VARIANTS), idle_starts_run=list(reader_b.IDLE_STARTS),
                seeds_run=list(reader_b.SEEDS))


class ReaderScopeTests(unittest.TestCase):
    def families(self, named):
        return sorted(set(named) | set(range(768, 768 + 256 * 60, 256)))

    def test_the_capacities_and_their_named_families(self):
        self.assertEqual(set(reader_b.CAPACITIES), {131328, 262144})
        self.assertEqual(reader_b.CAPACITIES[131328]['r2_named'], reader_b.R2_NAMED)
        self.assertEqual(reader_b.CAPACITIES[262144]['r2_named'], reader_b.R2_NAMED + (262144,))
        self.assertEqual(reader_b.R2_NAMED, (256, 2304, 16640, 65792, 131328))
        self.assertEqual(reader_b.CAPACITIES[262144]['r2_named'], admission.capacity_design(262144)['r2_named'])

    def test_each_capacity_is_full_on_its_own_design_and_reduced_on_the_others(self):
        self.assertEqual(reader_b.scope(reader_report(131328, self.families(reader_b.R2_NAMED))), ('full', []))
        wide = self.families(reader_b.R2_NAMED_262K)
        self.assertEqual(reader_b.scope(reader_report(262144, wide)), ('full', []))
        short = reader_b.scope(reader_report(262144, self.families(reader_b.R2_NAMED)))
        self.assertEqual(short, ('reduced', ['r2_families']))
        self.assertEqual(reader_b.scope(reader_report(200000, wide))[0], 'reduced')
        self.assertIn('capacity', reader_b.scope(reader_report(200000, wide))[1])
        self.assertIn('capacity', reader_b.scope(reader_report(None, wide))[1])
        # the wide families at the 131k capacity are still full: the window family is above its table, never replayed there
        self.assertEqual(reader_b.scope(reader_report(131328, wide))[0], 'full')

    def test_the_verdict_line_names_a_non_default_capacity_only(self):
        for capacity, named in ((131328, False), (262144, True)):
            report = dict(decision=dict(verdict='PASS', scope='full', first_differing=[], reasons=[]), comparisons=[], liveness=[],
                          r2_families_replayed=[256], capacity=capacity)
            line = reader_b.verdict_line(report)
            self.assertEqual(' capacity=%d' % capacity in line, named, line)
        report = dict(decision=dict(verdict='PASS', scope='full', first_differing=[], reasons=[]), comparisons=[], liveness=[], r2_families_replayed=[256])
        self.assertNotIn('capacity=', reader_b.verdict_line(report))


if __name__ == '__main__':
    unittest.main()
