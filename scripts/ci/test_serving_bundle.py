import unittest

from serving_bundle import expose_singleton_pages, package


class ServingBundleTests(unittest.TestCase):
    def test_singleton_overlay_preserves_other_frozen_adaptations(self):
        source = ('def outer():\n    def inner():\n'
            '        frozen_geometry = 32768\n'
            '        singleton_pages = upload(pages, ttnn.int32)\n'
            '        return frozen_geometry\n')
        result = expose_singleton_pages(source)
        self.assertEqual(result.replace('        self.singleton_pages = singleton_pages\n', ''), source)
        with self.assertRaises(ValueError):
            expose_singleton_pages(result)
        with self.assertRaises(ValueError):
            expose_singleton_pages(source + source)

    def test_unknown_binary_inventory_rejected_before_file_access(self):
        with self.assertRaisesRegex(ValueError, 'runtime inventory'):
            package('missing', 'missing', {'cached_binary_matches': False}, 'missing-output')
