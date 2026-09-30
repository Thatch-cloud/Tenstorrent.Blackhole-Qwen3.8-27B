"""lanes_report: the gate's reading of a lanes arm, on the logs and chunk timelines the whole-stack fakes (lanes_fakes.World) produce.

The World runs the real hook, lane gate, controller and packed step on a virtual clock at the design's round times, so the lines it
logs are the lines a served arm logs; the report must recover, from them alone, what the run did - and agree with the design's frame
arithmetic and with the client's own chunk timeline, which shares none of its parsing."""

import sys
import unittest

import lanes_report as report_module
from lanes_fakes import World
import lanes_sim
import serving_fast_lane as lanes
from serving_fast_lane import LaneConfig
from test_fast_lane_hook import make, run

SWEEP = '0@40,0.5@40,1@40,2@40,auto'


def sweep_world(standard=3, context='32k', schedule=SWEEP, seconds=40.0, **options):
    config = LaneConfig.from_environment({'QWEN_FAST_LANE_SCHEDULE': schedule}, seats=4)
    world = make('phase1', context, standard, config=config, max_tokens=12000, **options)
    run(world, seconds=seconds)
    return world


def streams_of(world):
    """Chunk timelines as the client sees them: one chunk per round with the cumulative count, the seed chunk first."""
    out = []
    for name in world.users:
        events = world.commits[name]
        out.append(dict(chunk_s=[0.0] + [stamp for stamp, _ in events],
                        chunk_tokens=[1] + [1 + sum(tokens for _, tokens in events[:index + 1]) for index in range(len(events))]))
    return out


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = sweep_world()
        cls.text = '\n'.join('2026-10-01 00:00:00.000 | INFO | [PINDIAG] ' + line for line in cls.world.lines)
        cls.report = report_module.lanes_report(cls.text, streams_of(cls.world), fast_users=[0])

    def test_no_problem_and_every_ratio_of_the_sweep_is_a_stretch_of_its_own(self):
        self.assertEqual(self.report['problems'], [])
        self.assertTrue(self.report['engaged'])
        ratios = [cell['ratio'] for cell in self.report['stretches'] if cell['read']]
        self.assertEqual(ratios, ['0', '0.5', '1', '2', 'auto'])
        self.assertTrue(all(cell['standard_users'] == 3 for cell in self.report['stretches']))

    def test_the_measured_rates_are_the_frame_arithmetic_of_the_measured_times(self):
        for cell in self.report['stretches']:
            with self.subTest(ratio=cell['ratio']):
                predicted = cell['predicted']
                self.assertAlmostEqual(cell['fast'], predicted['fast'], delta=0.03 * predicted['fast'])
                self.assertAlmostEqual(cell['standard_min'], predicted['standard'], delta=0.03 * predicted['standard'])

    def test_k_is_what_the_schedule_set_and_the_controller_stays_at_the_floor_on_auto(self):
        measured = {cell['ratio']: cell['k_measured'] for cell in self.report['stretches']}
        for ratio, k in (('0', 0.0), ('0.5', 0.5), ('1', 1.0), ('2', 2.0)):
            self.assertAlmostEqual(measured[ratio], k, delta=0.03)
        auto = [cell for cell in self.report['stretches'] if cell['ratio'] == 'auto'][0]
        self.assertGreaterEqual(auto['standard_min'], 75.0)
        self.assertGreaterEqual(auto['fast'], 150.0)
        self.assertTrue(auto['both'])

    def test_the_fast_rate_pays_for_the_standard_rate_ratio_by_ratio(self):
        fast = [cell['fast'] for cell in self.report['stretches'][:4]]
        standard = [cell['standard_min'] for cell in self.report['stretches'][:4]]
        self.assertEqual(fast, sorted(fast))
        self.assertEqual(standard, sorted(standard, reverse=True))

    def test_the_best_cell_holds_both_bars_and_is_the_fastest_that_does(self):
        (users, cell, label), = report_module.best_cells(self.report)
        self.assertEqual((users, label), (3, 'both'))
        self.assertTrue(cell['both'])
        rivals = [other['fast'] for other in self.report['stretches'] if other.get('both')]
        self.assertEqual(cell['fast'], max(rivals))

    def test_the_client_timeline_agrees_with_the_log(self):
        client = self.report['client']
        self.assertIsNotNone(client)
        auto = [cell for cell in self.report['stretches'] if cell['ratio'] == 'auto'][0]
        every = [entry['rate'] for entry in client['users'].values()]
        self.assertEqual([entry['lane'] for entry in client['users'].values()], ['fast', 'standard', 'standard', 'standard'])
        # the client window spans every ratio: its fast rate lies among the stretches' and every standard rate above the worst
        rates = [cell['fast'] for cell in self.report['stretches']]
        self.assertTrue(min(rates) * 0.95 <= every[0] <= max(rates) * 1.05, (every, rates))
        self.assertTrue(all(rate >= min(cell['standard_min'] for cell in self.report['stretches']) * 0.9 for rate in every[1:]))
        self.assertIsNotNone(auto)

    def test_the_rendering_names_every_stretch_the_best_cell_and_the_client_window(self):
        lines = report_module.verdict_lines(self.report)
        self.assertEqual(len([line for line in lines if line.startswith('lanes ratio=')]), 5)
        self.assertTrue(any(line.startswith('lanes BEST n=3') and '(both;' in line for line in lines))
        self.assertTrue(any(line.startswith('lanes client window') for line in lines))
        self.assertTrue(all('|' in line for line in lines if line.startswith('lanes ratio=')), 'the model beside each measured stretch')


class ShapeTests(unittest.TestCase):
    def test_the_live_count_is_a_stretch_boundary_and_one_to_three_users_each_read(self):
        world = make('phase1', '32k', 3, config=LaneConfig.from_environment({}, seats=4), max_tokens=12000)
        for request in world.scheduler.running:
            request.sampling_params.max_tokens = {'std2': 700, 'std1': 1500}.get(request.request_id, 12000)
        run(world, seconds=30.0)
        report = report_module.lanes_report('\n'.join(world.lines), None, fast_users=[0])
        self.assertEqual(report['problems'], [])
        counts = sorted(set(cell['standard_users'] for cell in report['stretches'] if cell['read']))
        self.assertGreaterEqual(len(counts), 2, counts)
        self.assertTrue(all(cell['live'] == cell['standard_users'] + 1 for cell in report['stretches'] if cell['read']))

    def test_a_stretch_with_too_few_packed_rounds_is_listed_unread_never_dropped(self):
        world = sweep_world(schedule='0@6,1@6,auto', seconds=3.0)
        report = report_module.lanes_report('\n'.join(world.lines), None, fast_users=[0])
        self.assertTrue(report['unread'])
        cell = report['stretches'][0]
        self.assertFalse(cell['read'])
        self.assertNotIn('fast', cell)
        self.assertTrue(any('unread' in line for line in report_module.verdict_lines(report)))

    def test_a_stall_between_rounds_ends_a_stretch_and_is_in_none(self):
        text = '\n'.join(
            '[LANE-ROUND] round=%d kind=packed block=packed members=u1,u2 live=2 ms=%s committed=12,12 switch=0 ratio=auto'
            % (index + 1, '90.0' if index != 30 else '5000.0') for index in range(60))
        text = '[LANE-ADMIT] request=a alias=u1 asked=fast granted=fast slots=0 reason=ok\n' \
               '[LANE-ADMIT] request=b alias=u2 asked=standard granted=standard slots=1,2,3 reason=ok\n' \
               '[LANE] engaged seats=4 ratio=auto k_max=3 gap_ms=250.0 floor=75.0 margin=0.05 reserve=1\n' + text
        report = report_module.lanes_report(text, None, fast_users=[0], min_rounds=20)
        cells = report['stretches']
        self.assertEqual([cell['packed_rounds'] for cell in cells], [30, 29])
        self.assertAlmostEqual(cells[0]['seconds'], 30 * 0.09, places=3)

    def test_the_prediction_is_the_design_frame(self):
        cell = report_module.predict(88.6, 55.3, 12.1, 12.1, 1.28, 1.0)
        self.assertAlmostEqual(cell['fast'], 171.0, delta=1.5)
        self.assertAlmostEqual(cell['standard'], 75.0, delta=1.5)
        self.assertIsNone(report_module.predict(88.6, None, 12.1, 12.1, 1.0))
        self.assertAlmostEqual(report_module.predict(100.0, 50.0, 10.0, 10.0, 0.0)['fast'], 100.0)


class ProblemTests(unittest.TestCase):
    ENGAGED = '[LANE] engaged seats=4 ratio=auto k_max=3 gap_ms=250.0 floor=75.0 margin=0.05 reserve=1'

    def test_an_arm_the_lanes_never_reached_is_a_problem_by_name(self):
        report = report_module.lanes_report('nothing here', None, fast_users=[0])
        self.assertIn('no "[LANE] engaged" line: the lane runtime never came up on this arm', report['problems'])
        self.assertIn('no [LANE-ROUND] line: no round ran under the lane runtime', report['problems'])

    def test_a_fast_mark_that_never_reached_the_worker_is_a_problem(self):
        text = self.ENGAGED + '\n[LANE-ADMIT] request=a alias=u1 asked=standard granted=standard slots=1,2,3 reason=ok'
        report = report_module.lanes_report(text, None, fast_users=[0])
        self.assertTrue(any('never reached the worker' in problem for problem in report['problems']), report['problems'])
        self.assertTrue(any('no request was granted the fast lane' in problem for problem in report['problems']))

    def test_a_desync_or_a_refused_attach_is_a_problem_carrying_its_own_line(self):
        text = (self.ENGAGED + '\n[LANE-DESYNC] the lane gate planned [u1] and the step scheduled [u1, u2]: the scheduler did not hide [u2]'
                '\n[LANE] refused: QWEN_FAST_SOLO_LANE is not 1')
        report = report_module.lanes_report(text, None)
        self.assertTrue(any(problem.startswith('[LANE-DESYNC]') for problem in report['problems']))
        self.assertTrue(any(problem.startswith('[LANE] refused:') for problem in report['problems']))

    def test_a_downgraded_second_fast_request_is_reported_not_a_problem(self):
        text = (self.ENGAGED + '\n[LANE-ADMIT] request=a alias=u1 asked=fast granted=fast slots=0 reason=ok'
                '\n[LANE-ADMIT] request=b alias=u2 asked=fast granted=standard slots=1,2,3 reason=fast-lane-busy')
        report = report_module.lanes_report(text, None, fast_users=[0, 1])
        self.assertEqual(report['downgrades'], ['u2'])
        self.assertFalse([problem for problem in report['problems'] if 'never reached' in problem])

    def test_an_arm_without_a_lane_runtime_reads_only_the_clients_streams(self):
        streams = [dict(chunk_s=[0.0, 1.0, 1.1, 1.2, 1.3, 1.4], chunk_tokens=[1, 13, 25, 37, 49, 60])]
        report = report_module.lanes_report('', streams, expect_lanes=False)
        self.assertEqual(report['problems'], [])
        self.assertNotIn('stretches', report)
        (entry,) = report['streams']
        self.assertAlmostEqual(entry['rate'], 120.0, delta=1.0)
        self.assertAlmostEqual(entry['tau'], 12.0, places=3)

    def test_a_stream_without_a_timeline_reads_none(self):
        self.assertEqual(report_module.stream_rates([None, dict(chunk_s=[0.0, 1.0])])[0]['rate'], None)
        self.assertIsNone(report_module.client_windows([dict(chunk_s=[0.0, 1.0, 2.0], chunk_tokens=[1, 2, 3]), None]))
        self.assertIsNone(report_module.client_windows([dict(chunk_s=[0.0, 0.5, 1.0], chunk_tokens=[1, 13, 25])]),
                          'a window shorter than the guards is not read')


if __name__ == '__main__':
    unittest.main()
