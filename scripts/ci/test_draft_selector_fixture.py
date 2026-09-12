import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from draft_selector_fixture import TENSORS, fetch, load_selector, ensure_fixture
from draft_convolution_fixture import MODEL, REVISION, HEADER_SHA256
from draft_remaining_layers_fixture import specifications
from draft_projection_full_fixture import TENSORS as PROJECTION


class SelectorFixtureTests(unittest.TestCase):
    def test_loader_rejects_corruption_without_redownloading(self):
        data = b'\x80\x3f'
        digest = hashlib.sha256(data).hexdigest()
        manifest = dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
            checkpoint_bytes=3848817896, tensors={'norm.weight': dict(file='norm.bf16', shape=[1],
                dtype='BF16', bytes=2, sha256=digest)})
        with tempfile.TemporaryDirectory() as temporary, \
                patch('draft_selector_fixture.TENSORS', {'norm.weight': ([1], 'norm.bf16')}), \
                patch('draft_selector_fixture.TENSOR_SHA256', {'norm.weight': digest}), \
                patch('draft_selector_fixture.fetch') as download:
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'norm.bf16').write_bytes(data)
            self.assertEqual(load_selector(root)[1]['norm.weight'].item(), 1)
            self.assertEqual(ensure_fixture(root), manifest)
            (root / 'norm.bf16').write_bytes(b'xx')
            with self.assertRaises(ValueError):
                load_selector(root)
            with self.assertRaises(ValueError):
                ensure_fixture(root)
            download.assert_not_called()

    def test_selector_selection_is_bounded(self):
        self.assertEqual(len(TENSORS), 4)
        self.assertEqual(sum(2 * math.prod(shape) for shape, filename in TENSORS.values()), 256911360)
        with patch('draft_selector_fixture.fetch_subset', return_value={}) as download:
            fetch(Path('selector'))
            self.assertEqual(download.call_args.kwargs['specifications'], TENSORS)

    def test_all_fixture_families_cover_checkpoint_payload_bytes(self):
        layer_bytes = sum(2 * math.prod(shape) for shape, filename in specifications(1).values())
        projection_bytes = sum(2 * math.prod(shape) for shape, filename in PROJECTION.values())
        selector_bytes = sum(2 * math.prod(shape) for shape, filename in TENSORS.values())
        self.assertEqual(5 * layer_bytes + projection_bytes + selector_bytes + 8936, 3848817896)
