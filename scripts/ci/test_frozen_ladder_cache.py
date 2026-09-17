import unittest

from frozen_context_geometry import CONTEXTS, geometry
from frozen_ladder_cache import build_identity


class LadderCacheTests(unittest.TestCase):
    def fixture(self):
        return dict(report_sha256='report', source='factory.cpp', source_before='before',
            factory_sha256='compiled-source', key_chunk_size=256, capacity=16640,
            target_tree_scratch=dict(sources={'kernel.cpp': 'kernel-hash'}, patch_sha256='patch'))

    def test_all_contexts_share_build_identity_without_changing_evidence(self):
        factory = self.fixture()
        for context in CONTEXTS:
            selected = dict(factory, capacity=geometry(context)['capacity'])
            self.assertEqual(build_identity(selected), build_identity(factory))
            self.assertEqual(selected['capacity'], context + 256)

    def test_every_other_build_input_still_invalidates_cache(self):
        factory = self.fixture()
        for field in set(factory) - {'capacity'}:
            with self.subTest(field=field):
                self.assertNotEqual(build_identity(factory), build_identity(dict(factory, **{field: 'changed'})))

    def test_unknown_field_and_invalid_capacity_fail_closed(self):
        for factory in (dict(self.fixture(), extra='unreviewed'), dict(self.fixture(), capacity=True),
                dict(self.fixture(), capacity=4096)):
            with self.assertRaises(ValueError):
                build_identity(factory)


if __name__ == '__main__':
    unittest.main()
