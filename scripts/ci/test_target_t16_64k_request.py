import unittest
from unittest.mock import patch

import target_t16_64k_request as candidate


class RequestTests(unittest.TestCase):
    def test_request_bounds(self):
        options = dict(rows=16, position=65536, remaining=255, replay=True,
            norm_batch=True, native_sampling=True, group_rows=4, short_context=False)
        candidate.validate_request_option(True, **options)
        for name, value in (('position', 8192), ('remaining', 257), ('rows', 32),
                ('native_sampling', False), ('group_rows', 8), ('remaining', True)):
            with self.assertRaises(ValueError):
                candidate.validate_request_option(True, **dict(options, **{name: value}))

    def test_capture_plan_and_restoration(self):
        import attention_request_plan
        import dspark_64k_variants
        import target_t16_attention_gate

        original = attention_request_plan.validate_ticket
        policies = dspark_64k_variants.POLICIES
        gate = target_t16_attention_gate.validate_request_option
        with patch.object(candidate, 'qualify', return_value={'hardware_qualified': True}):
            with candidate.request_scope('.'):
                plan = attention_request_plan.capture_plan(65536, 66560, 16, 255, max_verify_rows=16)
                self.assertEqual({capture.capacity for capture in plan.captures}, {None, 65792})
                self.assertEqual(plan.select(65536, 16, 255).capacity, 65792)
                self.assertTrue(dspark_64k_variants.POLICIES['scatter']['target_attention_t16'])
        self.assertIs(attention_request_plan.validate_ticket, original)
        self.assertIs(dspark_64k_variants.POLICIES, policies)
        self.assertIs(target_t16_attention_gate.validate_request_option, gate)
        with self.assertRaises(ValueError):
            original(65536, 16, 65792)


if __name__ == '__main__':
    unittest.main()
