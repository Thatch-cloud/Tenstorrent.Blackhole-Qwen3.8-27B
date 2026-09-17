import unittest
from unittest.mock import patch

import dspark_cached_markov_variants as variants


class CacheVariantTests(unittest.TestCase):
    def test_only_cache_differs_between_arms(self):
        candidate = dict(variants.POLICIES['publication'])
        self.assertTrue(candidate.pop('bias_cache'))
        self.assertEqual(candidate, variants.POLICIES['control'])
        self.assertTrue(candidate['gdn_shared_qk'])

    def test_requires_candidate_reset_and_rejects_cached_control(self):
        evidence = dict(released=True, reset_epoch=2,
            admission=dict(simulator_run=variants.RUN, reports=dict(variants.REPORTS)))
        value = dict(score_layout=dict(bias_cache=evidence))
        with patch.object(variants, 'shared_route') as shared:
            variants.validate_route(value, 'publication')
            shared.assert_called_with(value, 'publication')
            with self.assertRaises(ValueError):
                variants.validate_route(value, 'control')
            evidence['reset_epoch'] = 1
            with self.assertRaises(ValueError):
                variants.validate_route(value, 'publication')
