from copy import deepcopy
import unittest
from unittest.mock import Mock

from dspark_sfpu_request_screen import measure_bounded, summarize_screen


def request():
    return dict(arm='scatter', instrumented_timing=True, exact=True, state_exact=True,
        inactive_exact=True, length=65536, prompt_tokens=[1] * 65536,
        emitted=[1, 2, 3], committed_decode_tokens=2, commit_only_gdn=True,
        blocks=[dict(position=65536, rows=16)],
        dspark=dict(native_attention=True, proposal_trace=True,
            proposal_checks=[dict(position=65536, exact=True, tensors=6) for _ in range(2)]),
        gdn_verify_checks=[dict(position=65536, rows=16, unchanged=True)],
        captured_publication=dict(enabled=True, checks=[dict(exact=True, tensors=20) for _ in range(2)]),
        norm_scatter_kernel=dict(restored=True, loads=[1]))


class ScreenTests(unittest.TestCase):
    def test_short_generation_preserves_allocation_and_audit_contract(self):
        measure = Mock(return_value='result')
        self.assertEqual(measure_bounded(measure, 'model', max_new_tokens=256,
            audit_features=True, captured_publication=True), 'result')
        measure.assert_called_once_with('model', max_new_tokens=16,
            audit_features=True, captured_publication=True)
        for options in (dict(max_new_tokens=32, audit_features=True),
                dict(max_new_tokens=256, audit_features=False)):
            with self.assertRaises(ValueError):
                measure_bounded(measure, **options)

    def test_screen_cannot_publish_throughput(self):
        result = summarize_screen([request()])
        self.assertTrue(result['correctness_screen_passed'])
        self.assertFalse(result['performance_qualified'])
        self.assertFalse(result['full_request_qualified'])
        self.assertIsNone(result['arms']['scatter']['committed_tg'])

    def test_missing_or_failed_audits_reject(self):
        source = request()
        mutations = [lambda value: value.update(state_exact=False),
            lambda value: value.update(instrumented_timing=False),
            lambda value: value['dspark']['proposal_checks'].pop(),
            lambda value: value['captured_publication']['checks'].pop(),
            lambda value: value['gdn_verify_checks'][0].update(unchanged=False),
            lambda value: value['norm_scatter_kernel'].update(restored=False),
            lambda value: value.update(emitted=list(range(17)))]
        for mutation in mutations:
            value = deepcopy(source)
            mutation(value)
            with self.assertRaises(ValueError):
                summarize_screen([value])


if __name__ == '__main__':
    unittest.main()
