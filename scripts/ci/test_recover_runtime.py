import unittest

from recover_runtime import IMAGE_ID, REPOSITORY, verified_reference


class RuntimeRecoveryTests(unittest.TestCase):
    def test_immutable_reference_for_exact_config(self):
        document = {'SchemaV2Manifest': {'config': {'digest': IMAGE_ID}},
            'Descriptor': {'digest': 'sha256:' + 'a' * 64}}
        self.assertEqual(verified_reference(document), REPOSITORY + '@sha256:' + 'a' * 64)

    def test_reject_changed_image_missing_digest_and_indexes(self):
        for document in ([], {},
                {'SchemaV2Manifest': {'config': {'digest': 'sha256:' + 'b' * 64}},
                    'Descriptor': {'digest': 'sha256:' + 'a' * 64}},
                {'SchemaV2Manifest': {'config': {'digest': IMAGE_ID}}},
                {'SchemaV2Manifest': {'config': {'digest': IMAGE_ID}},
                    'Descriptor': {'digest': 'latest'}}):
            with self.subTest(document=document), self.assertRaises(ValueError):
                verified_reference(document)
