"""prefix_report: TTFT, turn time and the hit rate from the model's own markers, on CPU."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prefix_report as pr  # noqa: E402


def turn(ttft, wall, q=None, l=None, salt='s', continuation=True, ok=True, grant=None, restored=None, captured=None,
         capture_ms=None, prompt=1000, out=100):
    rows = [dict(q=q, l=l, restored_ms=restored, captured=captured or [], capture_ms=capture_ms)] if l else []
    return dict(ok=ok, ttft_s=ttft, wall_s=wall, salt=salt, continuation=continuation, prompt_tokens=prompt,
                completion_tokens=out, markers=dict(q=q if l else None, l=l, rows=rows, grant=grant))


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        values = list(range(1, 11))
        self.assertEqual(pr.percentile(values, 0.5), 5)
        self.assertEqual(pr.percentile(values, 0.9), 9)
        self.assertEqual(pr.percentile(values, 1.0), 10)
        self.assertEqual(pr.percentile([3], 0.9), 3)
        self.assertIsNone(pr.percentile([None], 0.5))


class PhaseTests(unittest.TestCase):
    def test_the_hit_rate_is_sum_q_over_sum_l_of_the_models_rows(self):
        records = [
            turn(12.0, 70.0, q=0, l=8000, continuation=False),
            turn(2.0, 60.0, q=8192, l=10000, grant=dict(h=8256, q=8192), restored=300.0, captured=[10240],
                 capture_ms=400.0),
            turn(3.0, 65.0, q=0, l=12000, grant=dict(h=10240, q=0)),
            turn(None, 5.0, ok=False),
            turn(1.0, 50.0, salt=None, q=0, l=500, continuation=True),
            turn(1.5, 55.0, q=None, l=None),
        ]
        s = pr.phase_summary(records, seconds=3600.0, rss_gb=18.2, pods=[5, 7])
        self.assertEqual((s['turns'], s['served'], s['failed']), (6, 5, 1))
        self.assertEqual(s['hit_rate'], round(8192 / 30000.0, 4), 'unsalted turns are left out')
        self.assertEqual(s['hit_rate_continuation'], round(8192 / 22000.0, 4))
        self.assertEqual((s['hit_turns'], s['marked_turns'], s['unmarked_salted_turns']), (1, 3, 1))
        self.assertEqual((s['trim_loss_tokens'], s['grants']), (64 + 10240, 2))
        self.assertEqual((s['restore_ms_p50'], s['capture_ms_p50']), (300.0, 400.0))
        self.assertEqual((s['ttft_p50'], s['ttft_continuation_p50']), (2.0, 1.5))
        self.assertEqual(s['turns_per_hour'], 5.0)
        self.assertEqual((s['rss_gb_max'], s['ci_pods']), (18.2, [5, 7]))
        line = pr.render_phase('agents-4', s)
        self.assertIn('agents-4: 5/6 turns served', line)
        self.assertIn('hit rate 0.2731', line)

    def test_an_empty_phase(self):
        s = pr.phase_summary([])
        self.assertEqual((s['turns'], s['hit_rate'], s['ttft_p50'], s['prompt_tokens_mean']), (0, None, None, None))
        self.assertNotIn('turns_per_hour', s)
        pr.render_phase('agents-1', s)

    def test_compare_phases_pairs_what_both_ran(self):
        a = dict(ttft_continuation_p50=2.0, ttft_continuation_p90=4.0, turns_per_hour=180.0)
        b = dict(ttft_continuation_p50=20.0, ttft_continuation_p90=40.0, turns_per_hour=120.0)
        out = pr.compare_phases({'agents-4': a, 'agents-5': a}, {'agents-4': b})
        self.assertEqual(out, {'agents-4': dict(ttft_p50=(2.0, 20.0), ttft_p90=(4.0, 40.0), turns_per_hour=(180.0, 120.0))})


if __name__ == '__main__':
    unittest.main()
