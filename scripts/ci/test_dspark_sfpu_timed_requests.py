from copy import deepcopy
import unittest

from dspark_sfpu_timed_requests import summarize_timed
from test_dspark_sfpu_request_screen import request


class TimedRequestTests(unittest.TestCase):
    def fixture(self):
        audited = request()
        value = deepcopy(audited)
        value.update(instrumented_timing=False, emitted=[1, 2, 3] + [4] * 16 + [99],
            eos_ids=[99], committed_decode_tokens=19, prefill_ms=20000, decode_ms=1000)
        value['blocks'][0].update(input_tokens=[1], accepted=1, committed=2)
        return audited, [value, deepcopy(value)]

    def test_full_loop_rates_and_scope(self):
        audited, values = self.fixture()
        summary = summarize_timed(values, audited)
        self.assertEqual(summary['arms']['scatter']['committed_tg'], 19)
        self.assertEqual(summary['arms']['scatter']['pp'], 3276.8)
        self.assertEqual(summary['arms']['scatter']['output_budget'], 256)
        self.assertFalse(summary['performance_qualified'])

    def test_incomplete_mismatched_and_invalid_measurements_reject(self):
        mutations = [lambda value: value.update(state_exact=False),
            lambda value: value.update(instrumented_timing=True),
            lambda value: value.update(eos_ids=[]),
            lambda value: value.update(decode_ms=float('nan')),
            lambda value: value.update(prefill_ms=0),
            lambda value: value['emitted'].__setitem__(0, 88),
            lambda value: value['blocks'][0].update(accepted=0)]
        for mutation in mutations:
            audited, values = self.fixture()
            mutation(values[1])
            with self.assertRaises(ValueError):
                summarize_timed(values, audited)
        audited, values = self.fixture()
        with self.assertRaises(ValueError):
            summarize_timed(values[:1], audited)


if __name__ == '__main__':
    unittest.main()
