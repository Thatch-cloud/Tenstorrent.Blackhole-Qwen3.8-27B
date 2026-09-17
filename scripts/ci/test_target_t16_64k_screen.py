from copy import deepcopy
import unittest
from unittest.mock import Mock

from test_dspark_sfpu_request_screen import request
from target_t16_64k_screen import measure_bounded, summarize_screen


class ScreenTests(unittest.TestCase):
    def test_t16_is_reachable_without_changing_reservation(self):
        measure = Mock()
        measure_bounded(measure, max_new_tokens=256, audit_features=True)
        measure.assert_called_once_with(max_new_tokens=17, audit_features=True)
        for options in (dict(max_new_tokens=17, audit_features=True),
                dict(max_new_tokens=256, audit_features=False)):
            with self.assertRaises(ValueError):
                measure_bounded(measure, **options)

    def test_actual_folded_route_and_all_audits_required(self):
        value = request()
        value.update(target_attention_t16=True, attention_replay=True, family_routing=True, capture_count=5)
        self.assertTrue(summarize_screen([value])['correctness_screen_passed'])
        mutations = [lambda entry: entry.update(target_attention_t16=False),
            lambda entry: entry.update(capture_count=4),
            lambda entry: entry['blocks'][0].update(rows=8),
            lambda entry: entry.update(state_exact=False),
            lambda entry: entry['dspark']['proposal_checks'].pop(),
            lambda entry: entry['captured_publication']['checks'].pop()]
        for mutation in mutations:
            candidate = deepcopy(value)
            mutation(candidate)
            with self.assertRaises(ValueError):
                summarize_screen([candidate])


if __name__ == '__main__':
    unittest.main()
