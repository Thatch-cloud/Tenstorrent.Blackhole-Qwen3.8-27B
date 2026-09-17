import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from draft_mlp_fixture import load_mlp, ensure_fixture
from draft_convolution_fixture import MODEL, REVISION, HEADER_SHA256


class MlpFixtureTests(unittest.TestCase):
    def test_only_pinned_mlp_data_is_loaded_and_reused(self):
        data = b'\x80\x3f'
        digest = hashlib.sha256(data).hexdigest()
        manifest = dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
            checkpoint_bytes=3848817896, tensors={'gate': dict(file='gate.bf16', shape=[1], dtype='BF16', bytes=2, sha256=digest)})
        with tempfile.TemporaryDirectory() as temporary, \
                patch('draft_mlp_fixture.TENSORS', {'gate': ([1], 'gate.bf16')}), \
                patch('draft_mlp_fixture.TENSOR_SHA256', {'gate': digest}), \
                patch('draft_mlp_fixture.fetch') as download:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'gate.bf16').write_bytes(data)
            self.assertEqual(load_mlp(root)[1]['gate'].item(), 1)
            self.assertEqual(ensure_fixture(root), manifest)
            (root / 'gate.bf16').write_bytes(b'xx')
            with self.assertRaises(ValueError):
                load_mlp(root)
            with self.assertRaises(ValueError):
                ensure_fixture(root)
            download.assert_not_called()
