import unittest
from unittest.mock import patch

from dspark_publication_variants import POLICIES, validate_route


class PublicationVariantTests(unittest.TestCase):
    def test_only_publication_differs(self):
        self.assertEqual(POLICIES['publication'], dict(POLICIES['control'], captured_publication=True))
        self.assertTrue(POLICIES['control']['fused_t16_mlp'])

    @patch('dspark_publication_variants.fusion_route')
    def test_audit_and_timing_routes(self, fusion):
        for audit in (True, False):
            value = dict(blocks=[{}], instrumented_timing=audit, captured_publication=dict(enabled=True,
                checks=[dict(exact=True, tensors=20)] * (2 if audit else 0)))
            validate_route(value, 'publication')
            with self.assertRaises(ValueError):
                validate_route(value, 'control')
            value['captured_publication']['checks'].append(dict(exact=False, tensors=20))
            with self.assertRaises(ValueError):
                validate_route(value, 'publication')
        validate_route({}, 'control')
        fusion.assert_called_with({}, 'fusion')


if __name__ == '__main__':
    unittest.main()
