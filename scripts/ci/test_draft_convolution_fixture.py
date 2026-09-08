import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from draft_convolution_fixture import load_convolution, ensure_fixture, verified_tensor, MODEL, REVISION, HEADER_SHA256


class ConvolutionFixtureTests(unittest.TestCase):
    def test_tensor_conversion_is_independently_hash_checked(self):
        import torch

        data = b'\x80\x3f'
        expected = hashlib.sha256(data).hexdigest()
        self.assertEqual(verified_tensor(data, [1], expected, 'weight').item(), 1)
        with patch('torch.frombuffer', return_value=torch.tensor([2.], dtype=torch.bfloat16)):
            with self.assertRaisesRegex(ValueError, 'Loaded tensor integrity failure: weight'):
                verified_tensor(data, [1], expected, 'weight')
        with patch('torch.frombuffer') as convert:
            with self.assertRaisesRegex(ValueError, 'Input bytes changed'):
                verified_tensor(b'xx', [1], expected, 'weight')
            convert.assert_not_called()

    def test_loader_checks_bytes_not_only_manifest(self):
        data = b'\x80\x3f'
        digest = hashlib.sha256(data).hexdigest()
        manifest = dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
            checkpoint_bytes=3848817896, tensors={'test': dict(file='test.bf16', shape=[1], dtype='BF16', bytes=2, sha256=digest)})
        with tempfile.TemporaryDirectory() as temporary, \
                patch('draft_convolution_fixture.TENSORS', {'test': ([1], 'test.bf16')}), \
                patch('draft_convolution_fixture.TENSOR_SHA256', {'test': digest}):
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'test.bf16').write_bytes(data)
            self.assertEqual(load_convolution(root)[1]['test'].item(), 1)
            with patch('draft_convolution_fixture.read_range') as reader:
                self.assertEqual(ensure_fixture(root), manifest)
                reader.assert_not_called()
            (root / 'test.bf16').write_bytes(b'xx')
            with self.assertRaisesRegex(ValueError, f'test at .*test.bf16; expected {digest}, read {hashlib.sha256(b"xx").hexdigest()}'):
                load_convolution(root)

    def test_loader_rejects_other_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(dict(model=MODEL, revision='other')))
            with self.assertRaises(ValueError):
                load_convolution(root)
