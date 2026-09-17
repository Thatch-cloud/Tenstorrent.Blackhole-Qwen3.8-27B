from contextlib import contextmanager
from types import SimpleNamespace
import unittest

from history_append_pair import history_dominated, paired_history


class HistoryPairTests(unittest.TestCase):
    def test_degraded_history_control_continues_but_unexplained_slowdown_stops(self):
        for explained in (True, False):
            with self.subTest(explained=explained):
                records, candidates = [], []

                @contextmanager
                def candidate():
                    candidates.append(True)
                    yield

                def measure(prompt, *, audit_features, max_new_tokens, proposal_trace):
                    result = dict(prompt_tokens=list(prompt), emitted=[7] * 16 + [9], eos_ids=[9],
                        max_new_tokens=256, committed_decode_tokens=16, decode_ms=1000,
                        prefill_ms=25000, exact=True, state_exact=True, inactive_exact=True,
                        blocks=[dict(position=65536, draft_ms=80, verify_readback_ms=80)])
                    if explained:
                        result['publication_diagnostics'] = dict(records=[
                            dict(stage='prepare_history', position=65536, host_ms=700)])
                    return result

                module = SimpleNamespace(measure_dspark_request=measure)
                def run_pair():
                    with paired_history(module, candidate, records, lambda record: None):
                        for ordinal in range(2):
                            module.measure_dspark_request([1] * 65536, audit_features=False,
                                max_new_tokens=256, proposal_trace=True)

                if explained:
                    run_pair()
                    self.assertEqual(len(records), 2)
                    self.assertEqual(candidates, [True])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Control below'):
                        run_pair()
                    self.assertEqual(len(records), 1)
                    self.assertEqual(candidates, [])
                self.assertTrue(records[0]['summary']['degraded'])
                self.assertIs(module.measure_dspark_request, measure)

    def test_changed_committed_output_is_rejected(self):
        records = []

        @contextmanager
        def candidate():
            yield

        def measure(prompt, *, audit_features, max_new_tokens, proposal_trace):
            return dict(prompt_tokens=list(prompt), emitted=[8 if records else 7, 9], eos_ids=[9],
                max_new_tokens=256, committed_decode_tokens=1, decode_ms=10,
                prefill_ms=25000, exact=True, state_exact=True, inactive_exact=True)

        module = SimpleNamespace(measure_dspark_request=measure)
        with self.assertRaisesRegex(ValueError, 'Identical committed output'):
            with paired_history(module, candidate, records, lambda record: None):
                for ordinal in range(2):
                    module.measure_dspark_request([1] * 65536, audit_features=False,
                        max_new_tokens=256, proposal_trace=True)
        self.assertEqual(len(records), 1)
        self.assertIs(module.measure_dspark_request, measure)

    def test_history_stall_classification_requires_matching_stage_evidence(self):
        request = dict(decode_ms=1000, blocks=[dict(position=65536, draft_ms=80, verify_readback_ms=80)],
            publication_diagnostics=dict(records=[dict(stage='prepare_history', position=65536, host_ms=700)]))
        self.assertTrue(history_dominated(request))
        request['blocks'][0]['draft_ms'] = 500
        self.assertFalse(history_dominated(request))
        request['blocks'][0]['draft_ms'] = 80
        request['publication_diagnostics']['records'][0]['position'] = 65537
        self.assertFalse(history_dominated(request))

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
