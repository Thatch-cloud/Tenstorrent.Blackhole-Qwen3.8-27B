import unittest

from dspark_markov_cache_analysis import predecessors, replay


class CacheAnalysisTests(unittest.TestCase):
    def test_cold_first_access_and_lru_eviction(self):
        result = replay([1, 2, 1, 3, 2, 1], 2)
        self.assertEqual((result['hits'], result['misses']), (1, 5))
        self.assertEqual(replay([7, 7], 0)['hits'], 0)
        self.assertEqual(replay([7, 7], 1)['hits'], 1)

    def test_payload_and_invalid_tokens(self):
        self.assertEqual(replay([], 64)['fp32_payload_bytes_per_card'], 63569920)
        for tokens, slots in (([True], 1), ([-1], 1), ([248320], 1), ([1], True), ([1], -1)):
            with self.assertRaises(ValueError):
                replay(tokens, slots)

    def test_last_generated_token_is_not_a_predecessor_in_same_block(self):
        request = dict(exact=True, state_exact=True, inactive_exact=True, instrumented_timing=False,
            proposed=15, blocks=[dict(rows=16, source='dspark', input_tokens=list(range(16)))])
        self.assertEqual(predecessors(request), list(range(15)))
        request['proposed'] = 16
        with self.assertRaises(ValueError):
            predecessors(request)


if __name__ == '__main__':
    unittest.main()
