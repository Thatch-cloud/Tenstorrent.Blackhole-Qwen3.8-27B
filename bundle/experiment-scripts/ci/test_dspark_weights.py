from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

import dspark_weights as loader


@contextmanager
def fixture_weights(*, nonfinite=False):
    tensors = dict(first=torch.tensor([[.5, -1.25], [2., 3.]], dtype=torch.bfloat16),
        second=torch.tensor([float('nan') if nonfinite else .25, .75], dtype=torch.bfloat16))
    header, payload, hashes = {'__metadata__': {'format': 'pt'}}, b'', {}
    for name, tensor in tensors.items():
        data = tensor.view(torch.uint8).numpy().tobytes()
        header[name] = dict(dtype='BF16', shape=list(tensor.shape), data_offsets=[len(payload), len(payload) + len(data)])
        hashes[name] = hashlib.sha256(data).hexdigest()
        payload += data
    encoded = json.dumps(header).encode()
    blob = struct.pack('<Q', len(encoded)) + encoded + payload
    with tempfile.TemporaryDirectory() as directory, patch.multiple(loader, CHECKPOINT_BYTES=len(blob),
            HEADER_BYTES=len(encoded), HEADER_SHA256=hashlib.sha256(encoded).hexdigest(),
            CHECKPOINT_SHA256=hashlib.sha256(blob).hexdigest(), CHUNK_BYTES=3, TENSORS={}), \
            patch.object(loader, 'validate_header') as validate:
        path = Path(directory) / 'model.safetensors'
        path.write_bytes(blob)
        yield path, tensors, hashes, validate


class DSparkWeightsTests(unittest.TestCase):
    def test_stream_hashes_and_owned_tensors_survive_close(self):
        with fixture_weights() as (path, tensors, hashes, validate):
            with loader.VerifiedWeights(path) as weights:
                validate.assert_called_once()
                self.assertEqual(weights.fingerprints(), hashes)
                first = weights.tensor('first')
                second_copy = weights.tensor('first')
                self.assertNotEqual(first.data_ptr(), second_copy.data_ptr())
                second_copy.zero_()
                self.assertTrue(torch.equal(first, tensors['first']))
                observed = weights.fingerprints()
                observed.clear()
                self.assertEqual(weights.fingerprints(), hashes)
                with self.assertRaises(ValueError):
                    weights.tensor('__metadata__')
            self.assertTrue(torch.equal(first, tensors['first']))
            with self.assertRaises(ValueError):
                weights.tensor('first')

    def test_payload_header_size_and_partial_files_reject(self):
        for change in ('payload', 'header', 'size', 'partial'):
            with fixture_weights() as (path, unused_tensors, unused_hashes, unused_validate):
                data = bytearray(path.read_bytes())
                if change == 'partial':
                    path.with_suffix('.safetensors.partial').touch()
                elif change == 'size':
                    path.write_bytes(data[:-1])
                else:
                    data[-1 if change == 'payload' else 8] ^= 1
                    path.write_bytes(data)
                with self.assertRaises(ValueError):
                    loader.VerifiedWeights(path)

    def test_replacement_after_verification_fails_even_with_same_content(self):
        with fixture_weights() as (path, unused_tensors, unused_hashes, unused_validate):
            weights = loader.VerifiedWeights(path)
            try:
                replacement = path.with_name('replacement')
                replacement.write_bytes(path.read_bytes())
                os.replace(replacement, path)
                with self.assertRaises(ValueError):
                    weights.tensor('second')
            finally:
                weights.__exit__(RuntimeError, None, None)

    def test_in_place_change_is_rejected_and_nonfinite_values_are_not_loaded(self):
        with fixture_weights() as (path, unused_tensors, unused_hashes, unused_validate):
            weights = loader.VerifiedWeights(path)
            try:
                with path.open('r+b') as stream:
                    stream.seek(-1, 2)
                    stream.write(b'\0')
                with self.assertRaises(ValueError):
                    weights.tensor('second')
            finally:
                weights.__exit__(RuntimeError, None, None)
        with fixture_weights(nonfinite=True) as (path, unused_tensors, unused_hashes, unused_validate):
            with loader.VerifiedWeights(path) as weights:
                with self.assertRaises(ValueError):
                    weights.tensor('second')

    def test_final_hash_catches_unread_changes_even_if_metadata_does_not(self):
        with fixture_weights() as (path, tensors, unused_hashes, unused_validate):
            weights = loader.VerifiedWeights(path)
            with patch.object(weights, '_assert_unchanged'):
                with path.open('r+b') as stream:
                    stream.seek(-1, 2)
                    stream.write(b'\0')
                self.assertTrue(torch.equal(weights.tensor('first'), tensors['first']))
                with self.assertRaises(ValueError):
                    weights.__exit__(None, None, None)
            self.assertIsNone(weights.source)


if __name__ == '__main__':
    unittest.main()
