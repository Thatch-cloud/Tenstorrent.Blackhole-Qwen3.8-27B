import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from draft_remaining_layers_fixture import specifications, fetch_layers, ensure_layers, load_layer, TENSOR_SHA256
from draft_convolution_fixture import MODEL, REVISION, HEADER_SHA256


class RemainingLayerFixtureTests(unittest.TestCase):
    def test_cached_layer_is_rehashed_and_corruption_never_refetched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'layer-1').mkdir()
            (root / 'layer-1/manifest.json').write_text('{}')
            with patch('draft_remaining_layers_fixture.fetch_subset') as fetch, \
                    patch('draft_remaining_layers_fixture.verified_bytes', side_effect=ValueError('corrupt')) as verify:
                with self.assertRaisesRegex(ValueError, 'corrupt'):
                    ensure_layers(root, (1,))
                fetch.assert_not_called()
                self.assertEqual(verify.call_args.kwargs['hashes'], TENSOR_SHA256['1'])

    def test_unpinned_staging_fails_before_network(self):
        with patch('draft_remaining_layers_fixture.TENSOR_SHA256', {}), \
                patch('draft_remaining_layers_fixture.fetch_subset') as fetch:
            with self.assertRaisesRegex(ValueError, 'audited hashes'):
                ensure_layers(Path('missing'), (1,))
            fetch.assert_not_called()

    def test_all_remaining_layers_have_complete_sha256_pins(self):
        self.assertEqual(set(TENSOR_SHA256), {'1', '2', '3', '4'})
        for layer in range(1, 5):
            hashes = TENSOR_SHA256[str(layer)]
            self.assertEqual(set(hashes), set(specifications(layer)))
            for digest in hashes.values():
                self.assertRegex(digest, r'^[0-9a-f]{64}$')

    def test_unpinned_layer_is_rejected_before_reading_files(self):
        with patch('draft_remaining_layers_fixture.TENSOR_SHA256', {}), self.assertRaisesRegex(ValueError, 'not been audited'):
            load_layer(Path('missing'), 4)

    def test_loader_rehashes_pinned_content(self):
        data = b'\x80\x3f'
        digest = hashlib.sha256(data).hexdigest()
        manifest = dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
            checkpoint_bytes=3848817896, tensors={'layers.1.test': dict(file='test.bf16', shape=[1],
                dtype='BF16', bytes=2, sha256=digest)})
        with tempfile.TemporaryDirectory() as temporary, \
                patch('draft_remaining_layers_fixture.specifications', return_value={'layers.1.test': ([1], 'test.bf16')}), \
                patch('draft_remaining_layers_fixture.TENSOR_SHA256', {'1': {'layers.1.test': digest}}):
            root = Path(temporary)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'test.bf16').write_bytes(data)
            self.assertEqual(load_layer(root, 1)[1]['layers.1.test'].item(), 1)
            (root / 'test.bf16').write_bytes(b'xx')
            with self.assertRaises(ValueError):
                load_layer(root, 1)

    def test_each_layer_has_complete_unique_bounded_tensor_selection(self):
        for layer in range(1, 5):
            tensors = specifications(layer)
            self.assertEqual(len(tensors), 15)
            self.assertEqual(len({filename for shape, filename in tensors.values()}), 15)
            self.assertTrue(all(name.startswith(f'layers.{layer}.') for name in tensors))
            self.assertEqual(sum(2 * math.prod(shape) for shape, filename in tensors.values()), 665948672)

    def test_invalid_selection_fails_before_network_access(self):
        with patch('draft_remaining_layers_fixture.fetch_subset') as fetch:
            for layers in ((), (1, 1), (0,), (5,), (True,)):
                with self.assertRaises(ValueError):
                    fetch_layers(Path('unused'), layers)
            fetch.assert_not_called()

    def test_layers_are_kept_in_separate_fixture_directories(self):
        with patch('draft_remaining_layers_fixture.fetch_subset', return_value={}) as fetch:
            fetch_layers(Path('staged'), (1, 4))
            self.assertEqual([call.args[0] for call in fetch.call_args_list], [Path('staged/layer-1'), Path('staged/layer-4')])
            self.assertEqual(fetch.call_args_list[1].kwargs['specifications'], specifications(4))
