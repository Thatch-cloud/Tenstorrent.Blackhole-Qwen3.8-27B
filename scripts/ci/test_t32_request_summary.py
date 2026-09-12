from copy import deepcopy
import unittest
from unittest.mock import patch

from dspark_request_experiment import run_loaded_requests, summarize_t32_simulator


class T32RequestSummaryTests(unittest.TestCase):
    def test_loaded_schedule_checks_admission_before_model_access(self):
        with patch('t32_attention_admission.require_active', side_effect=ValueError('simulator required')) as gate:
            with self.assertRaisesRegex(ValueError, 'simulator required'):
                run_loaded_requests(*([None] * 12), {}, None, prompt=[], context={}, t32_request=True)
        gate.assert_called_once_with()

    def record(self):
        return dict(instrumented_timing=True, exact=True, state_exact=True, inactive_exact=True,
            committed_tokens_per_second=None, committed_decode_tokens=32, emitted=list(range(33)),
            blocks=[dict(rows=32)], dspark=dict(proposals=31, verifier_rows=32, full_history=True,
                audit_features=True, proposal_trace=True, committed_feature_rows=32,
                feature_checks=[dict(exact=True) for index in range(10)],
                proposal_checks=[dict(exact=True, tensors=6) for index in range(2)]))

    def test_complete_audit_is_correctness_only(self):
        summary = summarize_t32_simulator([self.record()])
        self.assertTrue(summary['full_request_simulator_exact'])
        self.assertFalse(summary['hardware_qualified'] or summary['held_out_coding_quality'])
        self.assertIsNone(summary['committed_tg'])
        self.assertIsNone(summary['pp'])

    def test_incomplete_or_timed_requests_are_not_admitted(self):
        for field, value in (('exact', False), ('state_exact', False), ('inactive_exact', False),
                ('instrumented_timing', False), ('committed_tokens_per_second', 201),
                ('committed_decode_tokens', 0), ('emitted', [1]), ('blocks', [dict(rows=16)])):
            record = self.record()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_t32_simulator([record])
        for records in ([], [self.record(), self.record()]):
            with self.assertRaises(ValueError):
                summarize_t32_simulator(records)

    def test_all_publication_and_replay_checks_are_required(self):
        for field, value in (('proposals', 15), ('verifier_rows', 16), ('full_history', False),
                ('audit_features', False), ('proposal_trace', False), ('committed_feature_rows', 31),
                ('feature_checks', []), ('proposal_checks', []),
                ('proposal_checks', [dict(exact=True, tensors=2)] * 2)):
            record = deepcopy(self.record())
            record['dspark'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_t32_simulator([record])


if __name__ == '__main__':
    unittest.main()
