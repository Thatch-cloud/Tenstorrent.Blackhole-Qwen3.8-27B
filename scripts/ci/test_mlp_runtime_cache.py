from pathlib import Path
import unittest
from unittest.mock import patch

import mlp_runtime_cache


class MlpRuntimeCacheTests(unittest.TestCase):
    def test_inputs_bind_original_slice_and_base_binary(self):
        def digest(path):
            return mlp_runtime_cache.IMAGE_SHA256 if str(path).endswith(mlp_runtime_cache.SOURCE) else '1' * 64
        with patch.object(mlp_runtime_cache, 'digest', side_effect=digest):
            inputs = mlp_runtime_cache.inputs_for(Path('/native'), Path('/scripts'), Path('/patch'))
        self.assertEqual(inputs['slice_sha256'], mlp_runtime_cache.IMAGE_SHA256)
        self.assertEqual(inputs['base_binary_sha256'], '1' * 64)
        self.assertEqual(set(inputs['builders']), set(mlp_runtime_cache.BUILDERS))

    def test_other_slice_rejected(self):
        with patch.object(mlp_runtime_cache, 'digest', return_value='0' * 64), self.assertRaises(ValueError):
            mlp_runtime_cache.inputs_for(Path('/native'), Path('/scripts'), Path('/patch'))

    def test_not_available_without_explicit_hardware_allocation(self):
        with patch.dict(mlp_runtime_cache.os.environ, {}, clear=True), self.assertRaises(ValueError):
            mlp_runtime_cache.main()
