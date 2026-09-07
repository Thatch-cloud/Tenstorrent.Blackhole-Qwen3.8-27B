from copy import deepcopy
import unittest

from full_request_pair import measure_requests, summarize_requests


class MatchedRequestTests(unittest.TestCase):
    def test_attention_comparison_keeps_norm_and_family_routing_fixed(self):
        def measure(*, attention_replay):
            return dict(self.record(attention_replay), norm_batch=True, attention_replay=attention_replay, family_routing=True)

        records, summary = measure_requests(measure, arm_key='attention_replay')
        self.assertEqual([entry['attention_replay'] for entry in records], [False, True, True, False])
        self.assertEqual(summary['arm_key'], 'attention_replay')
        self.assertEqual(summary['decode_speedup'], 1.25)
        for key in ('norm_batch', 'family_routing'):
            invalid = deepcopy(records)
            invalid[1][key] = False
            with self.assertRaisesRegex(ValueError, 'identical norm batching'):
                summarize_requests(invalid, arm_key='attention_replay')

    def record(self, enabled):
        return dict(norm_batch=enabled, prompt_tokens=[8, 9], emitted=[0, 1, 2], max_new_tokens=3,
            eos_ids=[], vocab_size=10, committed_decode_tokens=2, proposed=1, accepted=1,
            exact=True, state_exact=True, inactive_exact=True, decode_ms=80 if enabled else 100,
            engine_setup_ms=20, prefill_ms=30, setup_amortized=False, cross_request_trace_reuse=False,
            blocks=[dict(rows=2, source='lookup', accepted=1, committed=2, match_length=1,
                         position=2, input_tokens=[0, 1])])

    def test_actual_abba_order_and_separate_setup_accounting(self):
        observed = []

        def measure(*, norm_batch):
            observed.append(norm_batch)
            return self.record(norm_batch)

        requests, summary = measure_requests(measure)
        self.assertEqual(observed, [False, True, True, False])
        self.assertEqual(len(requests), 4)
        self.assertEqual(summary['decode_speedup'], 1.25)
        self.assertEqual(summary['arms']['candidate']['committed_tokens_per_second'], 25)
        self.assertEqual(summary['arms']['candidate']['post_seed_including_setup_tokens_per_second'], 20)
        self.assertEqual(summary['arms']['candidate']['prefill_setup_decode_ms'], 260)

    def test_failure_does_not_start_later_requests(self):
        observed = []

        def measure(*, norm_batch):
            observed.append(norm_batch)
            if norm_batch:
                raise RuntimeError('request failed after cleanup')
            return self.record(norm_batch)

        with self.assertRaises(RuntimeError):
            measure_requests(measure)
        self.assertEqual(observed, [False, True])

    def test_incomplete_nonfinite_or_different_work_is_rejected(self):
        baseline = [self.record(enabled) for enabled in (False, True, True, False)]
        for key, value in (('exact', False), ('state_exact', False), ('inactive_exact', False),
                           ('norm_batch', 1), ('decode_ms', 0), ('decode_ms', float('nan')),
                           ('engine_setup_ms', -1), ('committed_decode_tokens', 0),
                           ('emitted', [0, 1, 3]), ('proposed', 2), ('setup_amortized', True),
                           ('cross_request_trace_reuse', True)):
            records = deepcopy(baseline)
            records[1][key] = value
            with self.assertRaises(ValueError):
                summarize_requests(records)
        records = deepcopy(baseline)
        records[1]['blocks'][0]['input_tokens'] = [0, 3]
        with self.assertRaisesRegex(ValueError, 'routing'):
            summarize_requests(records)
        with self.assertRaises(ValueError):
            summarize_requests(baseline[:3])
