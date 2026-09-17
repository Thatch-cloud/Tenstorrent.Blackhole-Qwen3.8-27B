import hashlib
import json
from pathlib import Path
import tempfile
import struct
import unittest
from unittest.mock import patch

from draft_projection_full_fixture import fetch, stream_tensor, load_projection, ensure_fixture, MODEL, REVISION, HEADER_SHA256


class FullProjectionFixtureTests(unittest.TestCase):
    def test_cache_reuse_rehashes_content_without_network(self):
        data = b'abcd'
        digest = hashlib.sha256(data).hexdigest()
        manifest = dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
            tensors={'fc.weight': dict(file='fc.bf16', shape=[2], dtype='BF16', bytes=4, sha256=digest)})
        with tempfile.TemporaryDirectory() as temporary, \
                patch('draft_projection_full_fixture.TENSORS', {'fc.weight': ([2], 'fc.bf16')}), \
                patch('draft_projection_full_fixture.TENSOR_SHA256', {'fc.weight': digest}), \
                patch('draft_projection_full_fixture.read_range') as reader:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'fc.bf16').write_bytes(data)
            self.assertEqual(ensure_fixture(root), manifest)
            (root / 'fc.bf16').write_bytes(b'bad!')
            with self.assertRaises(ValueError):
                ensure_fixture(root)
            reader.assert_not_called()
            self.assertEqual((root / 'fc.bf16').read_bytes(), b'bad!')

    def test_loader_rejects_unpinned_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(dict(model='wrong', revision='wrong', header_sha256='wrong')))
            with self.assertRaises(ValueError):
                load_projection(root)

    def test_complete_manifest_requires_both_verified_tensors(self):
        header = json.dumps({'fc.weight': dict(dtype='BF16', shape=[2, 8], data_offsets=[0, 32]),
            'hidden_norm.weight': dict(dtype='BF16', shape=[2], data_offsets=[32, 36])}).encode()
        total = 8 + len(header) + 36
        responses = [(struct.pack('<Q', len(header)), total), (header, total), (b'a' * 32, total), (b'b' * 4, total)]
        with tempfile.TemporaryDirectory() as temporary, \
                patch('draft_projection_full_fixture.HEADER_SHA256', hashlib.sha256(header).hexdigest()), \
                patch('draft_projection_full_fixture.TENSORS', {'fc.weight': ([2, 8], 'fc.bf16'), 'hidden_norm.weight': ([2], 'hidden_norm.bf16')}), \
                patch('draft_projection_full_fixture.read_range', side_effect=responses):
            manifest = fetch(Path(temporary))
            self.assertEqual(manifest['fetched_bytes'], total)
            self.assertEqual(manifest['tensors']['hidden_norm.weight']['bytes'], 4)
            self.assertEqual(json.loads((Path(temporary) / 'manifest.json').read_text()), manifest)

    def test_unpinned_header_is_rejected_before_tensor_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch('draft_projection_full_fixture.read_range',
                side_effect=[(struct.pack('<Q', 2), 100), (b'{}', 100)]) as reader:
            with self.assertRaises(ValueError):
                fetch(Path(temporary))
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_stream_is_bounded_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'fc.bf16'
            with patch('draft_projection_full_fixture.read_range', side_effect=[(b'abcd', 99), (b'ef', 99)]) as reader:
                manifest = stream_tensor(target, 10, 6, 99, chunk_bytes=4)
            self.assertEqual([(call.args[1], call.args[2]) for call in reader.call_args_list], [(10, 4), (14, 2)])
            self.assertEqual(target.read_bytes(), b'abcdef')
            self.assertEqual(manifest['sha256'], hashlib.sha256(b'abcdef').hexdigest())
            self.assertFalse(target.with_suffix('.bf16.partial').exists())
            with self.assertRaises(ValueError):
                stream_tensor(target, 10, 6, 99)

    def test_failed_stream_retains_partial_not_complete_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'fc.bf16'
            with patch('draft_projection_full_fixture.read_range', side_effect=[(b'abcd', 99), (b'ef', 100)]), self.assertRaises(ValueError):
                stream_tensor(target, 10, 6, 99, chunk_bytes=4)
            self.assertFalse(target.exists())
            self.assertEqual(target.with_suffix('.bf16.partial').read_bytes(), b'abcd')
            with self.assertRaises(ValueError):
                stream_tensor(target, 10, 6, 99)

    def test_bad_geometry_and_existing_fixture_fail_before_network(self):
        with tempfile.TemporaryDirectory() as temporary, patch('draft_projection_full_fixture.read_range') as reader:
            target = Path(temporary) / 'fc.bf16'
            for args in ((-1, 4, 99), (0, 100, 99), (0, 0, 99), (True, 4, 99)):
                with self.assertRaises(ValueError):
                    stream_tensor(target, *args)
            target.write_bytes(b'keep')
            with self.assertRaises(ValueError):
                fetch(Path(temporary))
            reader.assert_not_called()
            self.assertEqual(target.read_bytes(), b'keep')
