import unittest
from unittest.mock import patch

import matched_combined_timed as candidate
from matched_combined_timed_request import validate_updates


class MatchedTimingTests(unittest.TestCase):
    def test_fresh_identity_propagates_and_restores(self):
        before = (candidate.baseline.SCREEN_RUN, candidate.baseline.qualify_score)
        with patch.dict('os.environ', QWEN_MATCHED_COMBINED='1'):
            with candidate.identity_scope():
                self.assertEqual(candidate.baseline.SCREEN_RUN, 35038298030)
                self.assertIs(candidate.baseline.qualify_score, candidate.qualify_combined)
                self.assertEqual(candidate.baseline.shared.SCREEN_SHA256, candidate.SCREEN_SHA256)
        self.assertEqual((candidate.baseline.SCREEN_RUN, candidate.baseline.qualify_score), before)

    def test_every_request_uses_writer_and_cache_hit_loader(self):
        requests = [dict(blocks=[{}]), dict(blocks=[{}, {}])]
        updates = [dict(failed=False, restored=True, committed=count, prepared=count + 1,
            discarded=1, max_touched_rows=64) for count in (1, 2)]
        warmups = [dict(committed=False), dict(committed=False)]
        loaders = [dict(restored=True, calls=320, packed_calls=64,
            materializations=0, packed_materializations=0)]
        validate_updates(requests, updates, warmups, loaders)
        updates[1]['committed'] = 1
        with self.assertRaises(ValueError):
            validate_updates(requests, updates, warmups, loaders)
        updates[1]['committed'] = 2
        loaders[0]['materializations'] = 1
        with self.assertRaises(ValueError):
            validate_updates(requests, updates, warmups, loaders)


if __name__ == '__main__':
    unittest.main()
