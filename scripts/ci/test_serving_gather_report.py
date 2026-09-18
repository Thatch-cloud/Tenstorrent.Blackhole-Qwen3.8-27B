import copy
import unittest

from serving_gather_report import ARMS, REPORT_SHA256, records, summarize


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        canary = dict(passed=True, server_exit_code=0,
            shutdown=dict(worker_closed=True, devices_closed=True, engine_forced=False))
        http = dict(passed=True, requests=[])
        events = []
        kernel = dict(candidate_sha256='test-kernel')
        for ordinal, arm in enumerate(ARMS):
            http['requests'].append(dict(ordinal=ordinal, exact=True, context=4096,
                output_tokens=list(range(122)), stream_delivery_tokens_per_second=110.0))
            blocks = [dict(position=4096 + offset * 11, rows=16, committed=11,
                draft_ms=20.0, verify_host_ms=60.0, commit_host_ms=9.0,
                outside_phases_ms=1.0, cycle_ms=90.0,
                verifier=dict(blocking_trace_host_ms=59.0)) for offset in range(11)]
            events.append(dict(stage='fast_serving_phases', finished=True, cancelled=False,
                added_device_fences=False, blocks=blocks))
            events.append(dict(stage='gather_comparison_request', ordinal=ordinal,
                arm=arm, warmup=ordinal < 2, audit=None if arm == 'control' else
                dict(restored=True, report_sha256=REPORT_SHA256, kernels=[kernel])))
        return canary, http, events, kernel

    def test_warmups_excluded_and_nested_trace_not_added(self):
        fixture = self.fixture()
        fixture[2][0]['blocks'][0].update(draft_ms=1020.0, cycle_ms=1090.0)
        result = summarize(*fixture)
        self.assertEqual(result['cycle_speedup'], 1.0)
        self.assertAlmostEqual(result['measured']['control']['cycle_tokens_per_second'], 11000 / 90)
        self.assertFalse(result['performance_qualified'])
        self.assertFalse(result['repeatable_two_percent_screen'])
        self.assertEqual([pair['cycle_speedup'] for pair in result['pairs']], [1.0, 1.0])

    def test_aggregate_win_does_not_hide_pair_regression(self):
        fixture = self.fixture()
        for block in fixture[2][6]['blocks']:
            block.update(draft_ms=1.0, cycle_ms=71.0)
        for block in fixture[2][8]['blocks']:
            block.update(draft_ms=21.0, cycle_ms=91.0)
        result = summarize(*fixture)
        self.assertGreater(result['cycle_speedup'], 1.02)
        self.assertLess(result['pairs'][1]['cycle_speedup'], 1)
        self.assertFalse(result['repeatable_two_percent_screen'])

    def test_bad_or_incomplete_evidence_rejected(self):
        for corruption in ('missing', 'scope', 'tokens', 'nan', 'nested', 'shutdown', 'acceptance'):
            with self.subTest(corruption=corruption):
                canary, http, events, kernel = copy.deepcopy(self.fixture())
                if corruption == 'missing':
                    events.pop()
                elif corruption == 'scope':
                    events[7]['audit']['restored'] = False
                elif corruption == 'tokens':
                    http['requests'][3]['output_tokens'][0] = -1
                elif corruption == 'nan':
                    events[6]['blocks'][0]['cycle_ms'] = float('nan')
                elif corruption == 'nested':
                    events[6]['blocks'][0]['verifier']['blocking_trace_host_ms'] = 61
                elif corruption == 'shutdown':
                    canary['shutdown']['engine_forced'] = True
                else:
                    events[6]['blocks'][0]['rows'] = 15
                with self.assertRaises(ValueError):
                    summarize(canary, http, events, kernel)

    def test_log_prefixes(self):
        self.assertEqual(records('noise\n(Worker) {"stage":"fast_serving_phases"}\n'),
            [dict(stage='fast_serving_phases')])


if __name__ == '__main__':
    unittest.main()
