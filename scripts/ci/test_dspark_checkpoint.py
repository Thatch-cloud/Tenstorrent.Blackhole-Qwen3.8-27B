import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dspark_checkpoint as checkpoint


class Response(BytesIO):
    def __init__(self, data, *, status=200, size=None):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Length': str(len(data) if size is None else size)}


class DSparkCheckpointTests(unittest.TestCase):
    def test_complete_bounded_download_is_published_only_after_hash(self):
        data = b'fixture safetensors body'
        with tempfile.TemporaryDirectory() as directory, patch.object(checkpoint, 'CHUNK_BYTES', 3):
            path = Path(directory) / 'model.safetensors'
            checkpoint.copy_response(Response(data), path, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse(path.with_suffix('.safetensors.partial').exists())

    def test_bad_size_hash_and_partial_responses_never_publish(self):
        data = b'fixture'
        for response, digest in ((Response(data, status=206), hashlib.sha256(data).hexdigest()),
                (Response(data, size=3), hashlib.sha256(data).hexdigest()),
                (Response(data), '0' * 64), (Response(data[:-1], size=len(data)), hashlib.sha256(data).hexdigest()),
                (Response(data + b'x', size=len(data)), hashlib.sha256(data).hexdigest())):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.safetensors'
                with self.assertRaises(ValueError):
                    checkpoint.copy_response(response, path, len(data), digest)
                self.assertFalse(path.exists())

    def test_existing_complete_or_partial_files_are_not_overwritten(self):
        for partial in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.safetensors'
                existing = path.with_suffix('.safetensors.partial') if partial else path
                existing.write_bytes(b'preserve')
                with self.assertRaises(ValueError):
                    checkpoint.copy_response(Response(b'new'), path, 3, hashlib.sha256(b'new').hexdigest())
                self.assertEqual(existing.read_bytes(), b'preserve')

    def test_fetch_rejects_existing_or_insufficient_space_before_network(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(checkpoint.urllib.request, 'urlopen') as opened:
            path = Path(directory) / 'model.safetensors'
            path.touch()
            with self.assertRaises(ValueError):
                checkpoint.fetch(path)
            with patch.object(checkpoint.shutil, 'disk_usage') as usage:
                usage.return_value.free = checkpoint.CHECKPOINT_BYTES - 1
                with self.assertRaises(ValueError):
                    checkpoint.fetch(Path(directory) / 'another.safetensors')
            opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
