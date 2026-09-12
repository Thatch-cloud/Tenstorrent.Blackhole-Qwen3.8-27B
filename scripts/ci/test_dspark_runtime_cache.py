from pathlib import Path
import tempfile
import unittest

from dspark_runtime_cache import cache_key, inspect_entry, store_entry


class DSparkRuntimeCacheTests(unittest.TestCase):
    def test_complete_build_is_verified_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'built.so'
            binary.write_bytes(b'complete binary')
            inputs = dict(image='pinned',builders={'source':'sha'})
            self.assertIsNone(inspect_entry(root/'cache',inputs))
            manifest = store_entry(root/'cache',inputs,binary)
            self.assertEqual(inspect_entry(root/'cache',inputs),manifest)
            self.assertEqual(store_entry(root/'cache',inputs,binary),manifest)

    def test_changed_source_or_image_cannot_reuse_the_previous_build(self):
        base = dict(image='one',builders={'file':'a'})
        self.assertEqual(cache_key(base),cache_key(dict(builders={'file':'a'},image='one')))
        self.assertNotEqual(cache_key(base),cache_key(dict(image='two',builders={'file':'a'})))
        self.assertNotEqual(cache_key(base),cache_key(dict(image='one',builders={'file':'b'})))

    def test_tampered_binary_and_incomplete_entry_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'built.so'
            binary.write_bytes(b'good')
            inputs = dict(image='pinned')
            store_entry(root/'cache',inputs,binary)
            entry = root/'cache'/cache_key(inputs)
            (entry/'_ttnncpp.so').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                inspect_entry(root/'cache',inputs)
            (entry/'manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):
                inspect_entry(root/'cache',inputs)

    def test_existing_key_never_silently_replaces_a_different_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'built.so'
            binary.write_bytes(b'first')
            inputs = dict(image='pinned')
            store_entry(root/'cache',inputs,binary)
            binary.write_bytes(b'second')
            with self.assertRaises(ValueError):
                store_entry(root/'cache',inputs,binary)


if __name__=='__main__':
    unittest.main()
