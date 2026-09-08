import unittest

from dflash_memory import request_kv_allocation


class DFlashMemoryTests(unittest.TestCase):
    def test_single_stream_preserves_context_and_inactive_gdn_slots(self):
        control = request_kv_allocation('0')
        self.assertEqual(control['physical_pages'], 8200)
        for drafts in ('7', '31'):
            candidate = request_kv_allocation(drafts)
            self.assertEqual(candidate['physical_pages'], 1032)
            self.assertEqual(candidate['physical_pages'] - candidate['request_pages'], 8)
            for key in ('request_pages', 'block_tokens', 'request_capacity_tokens', 'gdn_slots'):
                self.assertEqual(candidate[key], control[key])
            self.assertEqual(candidate['request_pages'] * candidate['block_tokens'], 65536)

    def test_rejects_implicit_or_unqualified_policy(self):
        for invalid in (True, 7, None, '', '8', '32'):
            with self.assertRaises(ValueError):
                request_kv_allocation(invalid)
