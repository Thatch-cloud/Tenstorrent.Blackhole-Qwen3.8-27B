from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from dspark_score_pair import paired_scope


class PairTests(unittest.TestCase):
    def test_same_request_control_then_candidate(self):
        calls, records = [], []
        def measure(prompt, *, audit_features, max_new_tokens):
            calls.append('request')
            return dict(exact=True, state_exact=True, inactive_exact=True,
                committed_decode_tokens=135, decode_ms=4300, prefill_ms=25000, prompt_tokens=list(prompt))
        module = SimpleNamespace(measure_dspark_request=measure)
        @contextmanager
        def candidate(isolated):
            calls.append('candidate')
            self.assertIs(isolated.measure_dspark_request, measure)
            yield
        with paired_scope(module, candidate, records, lambda record: None):
            for ordinal in range(2):
                module.measure_dspark_request([1] * 65536, audit_features=False, max_new_tokens=256)
        self.assertEqual(calls, ['request', 'candidate', 'request'])
        self.assertEqual([record['summary']['arm'] for record in records], ['control', 'score_layout'])
        self.assertEqual([record['summary']['context'] for record in records], [65536, 65536])
        self.assertIs(module.measure_dspark_request, measure)

    def test_degraded_control_retained_and_candidate_not_started(self):
        records = []
        def measure(prompt, *, audit_features, max_new_tokens):
            return dict(exact=True, state_exact=True, inactive_exact=True,
                committed_decode_tokens=135, decode_ms=17000, prefill_ms=25000, prompt_tokens=list(prompt))
        module = SimpleNamespace(measure_dspark_request=measure)
        with self.assertRaisesRegex(RuntimeError, 'Control below'):
            with paired_scope(module, lambda module: self.fail('candidate started'), records, lambda record: None):
                module.measure_dspark_request([1] * 65536, audit_features=False, max_new_tokens=256)
        self.assertEqual(len(records), 1)
        self.assertIs(module.measure_dspark_request, measure)
