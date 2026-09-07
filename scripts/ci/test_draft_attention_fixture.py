import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from draft_attention_fixture import load_attention
from draft_convolution_fixture import MODEL, REVISION, HEADER_SHA256


class AttentionFixtureTests(unittest.TestCase):
    def test_attention_loader_uses_its_own_pinned_tensor_set(self):
        data = b'\x80\x3f'
        digest = hashlib.sha256(data).hexdigest()
        manifest = dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
            checkpoint_bytes=3848817896, tensors={'query': dict(file='query.bf16', shape=[1], dtype='BF16', bytes=2, sha256=digest)})
        with tempfile.TemporaryDirectory() as temporary, \
                patch('draft_attention_fixture.TENSORS', {'query': ([1], 'query.bf16')}), \
                patch('draft_attention_fixture.TENSOR_SHA256', {'query': digest}):
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'query.bf16').write_bytes(data)
            self.assertEqual(load_attention(root)[1]['query'].item(), 1)
            (root / 'query.bf16').write_bytes(b'xx')
            with self.assertRaises(ValueError):
                load_attention(root)
