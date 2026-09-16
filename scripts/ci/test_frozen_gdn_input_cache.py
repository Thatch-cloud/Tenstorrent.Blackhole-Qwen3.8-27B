import unittest

from frozen_gdn_input_cache import buffers, cache_source, reader, LOOP
from gdn_shared_qk_recurrence import INPUTS
from gdn_vsplit import cb_plan


class GdnInputCacheTests(unittest.TestCase):
    def test_three_bf16_pages_and_no_compute_buffer_change(self):
        io, fp32 = cb_plan('recurrence')
        changed_io, changed_fp32 = buffers(io, fp32)
        self.assertEqual(changed_fp32, fp32)
        self.assertNotIn(31, io)
        self.assertEqual(changed_io, io | {31: 3})
        with self.assertRaises(ValueError):
            buffers(changed_io, fp32)

    def test_prefetches_only_v_beta_gate_before_loop(self):
        cache = cache_source()
        self.assertEqual(cache.count('noc.async_read('), 3)
        self.assertNotIn('noc.async_read(q_acc', cache)
        self.assertNotIn('noc.async_read(k_acc', cache)
        source = LOOP + '\n' + INPUTS + '}\n    query_cache.pop_front(4);\n'
        changed = reader(source)
        before, loop = changed.split(LOOP)
        self.assertEqual(before.count('noc.async_read('), 3)
        self.assertNotIn('gather_scalar(', loop)
        self.assertIn('gather_normalized(20, 10, token)', loop)
        self.assertIn('cache.pop_front(3)', loop)
        with self.assertRaises(ValueError):
            reader(changed)


if __name__ == '__main__':
    unittest.main()
