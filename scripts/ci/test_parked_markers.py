"""Stage E, E7: the parked telemetry, and parked_markers, which reads it.

Each line format is held against its producer: the parked set built, rebound, unparked and re-parked on the census
world (test_parked_census) with its real log calls; the producers' own format constants rendered with sample values
(the coordinator's release line, the stop, single rebuilt and kept lines); the memory ledger's P7p phase and its
rebind bracket rendered by the real ledger; and the publish prewarm's own line. Then the judge of an arm's log
(parked_markers.problems): a parked arm, a flag-off arm that must show none, a stop at k < 4, a missing warm or
prewarm, an unpark.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parked_markers as markers  # noqa: E402
import serving_parked_engines as parked  # noqa: E402
from test_parked_census import World  # noqa: E402
from test_parked_engine_set import ParkedRequest, make_set  # noqa: E402


def run_cycle(world, environ=None, requests=1):
    engines = make_set(world, environ=environ)
    engines.build()
    for serial in range(requests):
        entry = engines.take()
        request = ParkedRequest(world, engines, entry, 'req%d' % serial, 300, 16)
        while request.step():
            pass
        request.finish()
    return engines


class ProducerLineTests(unittest.TestCase):
    def test_the_attach_and_rebind_lines_parse_as_the_set_wrote_them(self):
        with World() as world:
            run_cycle(world, environ={parked.AUDIT_FLAG: '1'}, requests=2)
            facts = markers.scan(world.lines)
            (built,) = facts['built']
            self.assertEqual((built['k'], built['slots']), (4, 4))
            self.assertGreater(built['attach_ms'], 0)
            self.assertGreater(built['free'], 0)
            self.assertGreater(built['largest_free'], 0)
            (warm,) = facts['warm']
            self.assertEqual((warm['rows'], warm['window']), (2048, 256))
            self.assertGreater(warm['prewarm_pairs'], 0)
            self.assertEqual(facts['prewarm'], [warm['prewarm_pairs']])
            self.assertEqual([entry['request'] for entry in facts['rebinds']], ['req0', 'req1'])
            self.assertEqual([(entry['slot'], entry['prompt'], entry['budget']) for entry in facts['rebinds']],
                             [(0, 300, 16), (0, 300, 16)])
            self.assertEqual([entry['generation'] for entry in facts['rebinds']], [2, 3])
            self.assertFalse(any(entry['single_rebuilt'] for entry in facts['rebinds']))
            # QWEN_FAST_PARKED_AUDIT=1: one peak reading per rebind, and one for the attach's 2048 warm
            self.assertEqual(len(facts['peaks']), 3)
            for peak in facts['peaks']:
                self.assertEqual(peak['held'], peak['free_before'] - peak['free_at_peak'])
                self.assertGreaterEqual(peak['held'], 0)
            self.assertEqual(facts['peaks'][0]['prompt'], 2048)
            self.assertEqual(problems_of(facts), [])

    def test_the_peak_reading_is_off_without_the_audit(self):
        with World() as world:
            run_cycle(world)
            self.assertEqual(markers.scan(world.lines)['peaks'], [])

    def test_unpark_and_repark_lines(self):
        with World() as world:
            engines = run_cycle(world, environ={parked.FAULT_FLAG: 'park'})
            self.assertEqual(engines.repark_idle(), [0])
            facts = markers.scan(world.lines)
            self.assertEqual(facts['unparked'], [dict(slot=0, reason=parked.FAULT_REASON)])
            self.assertEqual([entry['slot'] for entry in facts['reparked']], [0])
            self.assertEqual(facts['negative'], [dict(kind='fault', value='park')])
            out, _ = markers.problems(facts, True)
            self.assertEqual(len(out), 1)
            self.assertIn('1 slots unparked', out[0])
            self.assertEqual(markers.problems(facts, True, allow_unparks=True), ([], []))

    def test_negative_control_and_ballast_lines(self):
        with World() as world:
            engines = make_set(world, environ={parked.NEGATIVE_FLAG: 'drafter', parked.BALLAST_FLAG: str(64 * 2 ** 20)})
            engines.build()
            facts = markers.scan(world.lines)
            self.assertEqual(facts['negative'], [dict(kind='mode', value='drafter')])
            (ballast,) = facts['ballast']
            self.assertGreaterEqual(ballast['bytes'], 64 * 2 ** 20)
            self.assertGreater(ballast['buffers'], 0)


class FormatConstantTests(unittest.TestCase):
    """The producers' own format strings, rendered with sample values, read back by field."""

    def test_stop_single_rebuilt_kept_and_release_lines(self):
        lines = [
            parked.STOPPED_MARKER.format(2, 4, 'free+largest', 1500000000, 900000000, 1200000000),
            parked.SINGLE_REBUILT_MARKER.format(1, 'park', 88.5, 4194304, 227000000),
            parked.SINGLE_KEPT_MARKER.format(3, 'idle', 'free', 700000000, 600000000, 900000000),
            parked.UNPARKED_MARKER.format(2, 'pool slot moved: x=1'),
            parked.REPARKED_MARKER.format(2, 1234.5),
        ]
        import dflash_packed_proposal_coordinator as coordinator

        lines.append(coordinator.PARKED_RELEASED_LINE.format(slot=1, quad=1, pairs=[(0, 1), (2, 3)]))
        facts = markers.scan('2026-01-01 00:00:00 | INFO | %s' % line for line in lines)
        self.assertEqual(facts['stopped'], [dict(k=2, slots=4, short='free+largest', free=1500000000,
                                                 largest_free=900000000, need=1200000000)])
        self.assertEqual(facts['single_rebuilt'], [dict(slot=1, moment='park', ms=88.5, trace_delta=4194304,
                                                        dram_delta=227000000)])
        self.assertEqual(facts['single_kept'], [dict(slot=3, moment='idle', short='free', need=900000000)])
        self.assertEqual(facts['unparked'], [dict(slot=2, reason='pool slot moved: x=1')])
        self.assertEqual(facts['reparked'], [dict(slot=2, ms=1234.5)])
        self.assertEqual(facts['released'], [dict(slot='1', quad=True, pairs='[(0, 1), (2, 3)]')])

    def test_the_publish_prewarm_line_and_its_skip(self):
        import publish_prewarm

        publish_prewarm_line = '{} pairs={} count={} ms={:.2f} program_cache={}->{}'.format(
            publish_prewarm.MARKER, '16:1-3', 3, 12.5, 40, 43)
        skipped = '{} skipped history_rows={} (the publication shapes settle at {})'.format(
            publish_prewarm.MARKER, 512, 2048)
        facts = markers.scan([publish_prewarm_line, skipped])
        self.assertEqual(facts['prewarm'], [3])
        self.assertEqual(facts['prewarm_skipped'], [512])

    def test_the_ledger_p7p_phase_and_the_rebind_bracket(self):
        import memory_ledger
        from test_memory_ledger import FakeOperations, ledger_for

        operations = FakeOperations()
        ledger, lines, _ = ledger_for(operations)
        operations.charge(29 * 10 ** 9)
        ledger.phase('P7p', point='parked k=4')
        token = ledger.before('rebind', estimate=100 * 2 ** 20, point='req=abc')
        ledger.after(token)
        facts = markers.scan(lines)
        self.assertEqual([entry['chip'] for entry in facts['p7p']], [0, 1])
        self.assertEqual(facts['p7p'][0]['point'], 'parked k=4')
        self.assertAlmostEqual(facts['p7p'][0]['allocated_gb'], 29.0, places=2)
        self.assertGreater(facts['p7p'][0]['free_gb'], 0)
        self.assertIn('before', [entry['when'] for entry in facts['rebind_ops']])
        self.assertIn('after', [entry['when'] for entry in facts['rebind_ops']])
        self.assertTrue(lines[0].startswith('[MEMLEDGER] phase=P7p'))
        self.assertIn(memory_ledger.BEFORE_MARKER + 'rebind', ' '.join(lines))

    def test_every_producer_marker_prefix_is_the_one_the_scan_reads(self):
        self.assertEqual(markers.BUILT, parked.BUILT_MARKER)
        self.assertEqual(markers.WARM, parked.WARM_MARKER)
        self.assertTrue(markers.REBIND.startswith(parked.REBIND_MARKER))
        self.assertEqual(markers.PEAK, parked.PEAK_MARKER)
        self.assertEqual(markers.BALLAST, parked.BALLAST_MARKER)
        self.assertEqual(markers.NEGATIVE, parked.NEGATIVE_MARKER)
        self.assertTrue(parked.UNPARKED_MARKER.startswith(markers.SLOT))
        self.assertTrue(parked.REPARKED_MARKER.startswith(markers.SLOT))
        self.assertTrue(parked.STOPPED_MARKER.startswith(markers.STOPPED))
        import dflash_packed_proposal_coordinator as coordinator
        import publish_prewarm

        self.assertTrue(coordinator.PARKED_RELEASED_LINE.startswith(markers.RELEASED))
        self.assertEqual(markers.PREWARM, publish_prewarm.MARKER)

    def test_a_built_line_without_readings(self):
        facts = markers.scan(['[PINDIAG] parked engines built k=4 of 4 attach_ms=10.5 dram unavailable (no view)'])
        self.assertEqual(facts['built'], [dict(k=4, slots=4, attach_ms=10.5, unavailable='dram unavailable (no view)')])


def problems_of(facts, **options):
    out, missing = markers.problems(facts, True, **options)
    return out + missing


class JudgeTests(unittest.TestCase):
    def facts(self, **changes):
        base = dict(built=[dict(k=4, slots=4, attach_ms=1.0)], stopped=[], warm=[dict(rows=2048, ms=1.0, window=256,
                                                                                       prewarm_pairs=7)],
                    ballast=[], negative=[], rebinds=[], peaks=[], unparked=[], reparked=[], single_rebuilt=[],
                    single_kept=[], released=[], prewarm=[7], prewarm_skipped=[], p7p=[], rebind_ops=[], markers=3)
        base.update(changes)
        return base

    def test_a_clean_parked_arm_passes(self):
        self.assertEqual(problems_of(self.facts()), [])

    def test_the_flag_off_twin_shows_no_parked_marker(self):
        self.assertEqual(markers.problems(dict(self.facts(built=[], warm=[], prewarm=[])), False), ([], []))
        out, _ = markers.problems(self.facts(), False)
        self.assertEqual(len(out), 1)
        self.assertIn('built, warm', out[0])

    def test_each_failure_is_named(self):
        cases = (('no build', dict(built=[]), 'was never built'),
                 ('3 of 4', dict(built=[dict(k=3, slots=4, attach_ms=1.0)]), 'built 3 of 4'),
                 ('a stop', dict(stopped=[dict(k=2, slots=4, short='free', free=1, largest_free=1, need=2)]),
                  'stopped at k=2'),
                 ('no warm', dict(warm=[]), 'never warmed'),
                 ('warm below 2048', dict(warm=[dict(rows=1, ms=1.0, window=256, prewarm_pairs=0)]), 'not 2048'),
                 ('no prewarm', dict(prewarm=[]), 'no publish prewarm'),
                 ('prewarm skipped', dict(prewarm=[], prewarm_skipped=[512]), 'skipped at history_rows'),
                 ('an unpark', dict(unparked=[dict(slot=1, reason='x')]), '1 slots unparked'))
        for name, changes, wanted in cases:
            with self.subTest(case=name):
                out = problems_of(self.facts(**changes))
                self.assertEqual(len(out), 1, out)
                self.assertIn(wanted, out[0])

    def test_a_scan_of_nothing_is_the_empty_facts(self):
        facts = markers.scan([])
        self.assertEqual(facts['markers'], 0)
        self.assertEqual(json.loads(json.dumps(facts)), facts)


if __name__ == '__main__':
    unittest.main()
