import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import dspark_markov_fixture as fixture


class DSparkMarkovFixtureTests(unittest.TestCase):
    def prepare(self, root, value=0.):
        matrix = torch.full((2, 256), value, dtype=torch.bfloat16)
        data = bytes(memoryview(matrix.view(torch.uint8).numpy()))
        tensors = {name: dict(file=entry['file'], shape=[2, 256], dtype='BF16', bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest()) for name, entry in fixture.TENSORS.items()}
        for entry in tensors.values():
            (root / entry['file']).write_bytes(data)
        return matrix, tensors

    def test_loaded_bytes_are_exact_and_manifest_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, tensors = self.prepare(root, .5)
            with patch.object(fixture, 'TENSORS', tensors):
                (root / 'manifest.json').write_text(json.dumps(fixture.expected_manifest()))
                manifest, predecessor, successor = fixture.load_fixture(root)
                self.assertEqual(manifest, fixture.expected_manifest())
                self.assertTrue(torch.equal(predecessor, expected))
                self.assertTrue(torch.equal(successor, expected))

    def test_corruption_partial_file_and_unpinned_manifest_fail(self):
        for mode in ('content', 'length', 'partial', 'manifest'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                unused_matrix, tensors = self.prepare(root)
                with patch.object(fixture, 'TENSORS', tensors):
                    manifest = fixture.expected_manifest()
                    if mode == 'content':
                        (root / 'predecessor.bf16').write_bytes(b'X' * 1024)
                    elif mode == 'length':
                        (root / 'predecessor.bf16').write_bytes(b'X')
                    elif mode == 'partial':
                        (root / 'predecessor.bf16.partial').touch()
                    else:
                        manifest['revision'] = 'main'
                    (root / 'manifest.json').write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError):
                        fixture.load_fixture(root)

    def test_nonfinite_matrix_is_rejected_even_with_matching_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unused_matrix, tensors = self.prepare(root, float('nan'))
            with patch.object(fixture, 'TENSORS', tensors):
                (root / 'manifest.json').write_text(json.dumps(fixture.expected_manifest()))
                with self.assertRaises(ValueError):
                    fixture.load_fixture(root)

    def test_fetch_refuses_existing_directory_before_network(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fixture, 'read_range') as ranged:
            with self.assertRaises(ValueError):
                fixture.fetch(Path(directory))
            ranged.assert_not_called()


if __name__ == '__main__':
    unittest.main()
