from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from history_append_pair import paired_history


class HistoryPairTests(unittest.TestCase):
    def test_complete_pair_and_metrics(self):
        active, records = [], []

        @contextmanager
        def candidate():
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        def measure(prompt, *, audit_features, max_new_tokens, proposal_trace):
            return dict(prompt_tokens=list(prompt), emitted=[7] * 135 + [9], eos_ids=[9],
                max_new_tokens=256, committed_decode_tokens=135, decode_ms=3000 if active else 4000,
                prefill_ms=25000, exact=True, state_exact=True, inactive_exact=True)
        module = SimpleNamespace(measure_dspark_request=measure)
        with paired_history(module, candidate, records, lambda record: None):
            for ordinal in range(2):
                module.measure_dspark_request([1] * 65536, audit_features=False, max_new_tokens=256, proposal_trace=True)
        self.assertIs(module.measure_dspark_request, measure)
        self.assertEqual([record['summary']['committed_tg'] for record in records], [33.75, 45])
        self.assertEqual(records[1]['summary']['pp'], 2621.44)
        self.assertFalse(active)

    def test_different_prompt_rejected_before_candidate(self):
        records = []
        def initial(prompt, *, audit_features, max_new_tokens, proposal_trace):
            return dict(prompt_tokens=list(prompt), emitted=[7, 9], eos_ids=[9], max_new_tokens=256,
                committed_decode_tokens=1, decode_ms=10, prefill_ms=100, exact=True, state_exact=True, inactive_exact=True)
        module = SimpleNamespace(measure_dspark_request=initial)
        with self.assertRaisesRegex(ValueError, 'Matched prompt'):
            with paired_history(module, lambda: self.fail('candidate entered'), records, lambda record: None):
                module.measure_dspark_request([1] * 65536, audit_features=False, max_new_tokens=256, proposal_trace=True)
                module.measure_dspark_request([2] * 65536, audit_features=False, max_new_tokens=256, proposal_trace=True)
        self.assertIs(module.measure_dspark_request, initial)


if __name__ == '__main__':
    unittest.main()
