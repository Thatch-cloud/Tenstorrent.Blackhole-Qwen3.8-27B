from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import fused_t16_admission as admission
import test_dram_mlp_down_scope as fixtures
from test_dram_mlp_sharded import Tensor


class FusionAdmissionTests(unittest.TestCase):
    def fixture(self):
        operations, model = fixtures.DownScopeTests().fixture()
        for layer in model.layers:
            layer.feed_forward.compute_kernel_config_decode.math_approx_mode = True
            layer.feed_forward.weights.w_gate_up = Tensor((5120, 17408), 'bf4', 'dram')
        return operations, model

    def test_retained_simulator_report_and_sources(self):
        self.assertTrue(admission.qualify_simulator()['passed'])

    def test_modified_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(admission.__file__).with_name('fused-t16-target-simulator.json')
            target = Path(directory) / source.name
            shutil.copyfile(source, target)
            target.write_bytes(target.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'Reviewed'):
                admission.qualify_simulator(directory)

    def test_all_target_layers_checked_without_new_weight_copies(self):
        operations, model = self.fixture()
        with patch('packed_weight_check.compare_packed_weights', return_value='counts') as compare, patch(
                'packed_weight_check.read_comparison', return_value=[dict(chip=chip, exact=True) for chip in (0, 1)]), patch.object(
                admission, 'release_owned') as release:
            result = admission.qualify_target_weights(operations, model)
        self.assertEqual(len(result['checks']), 256)
        self.assertEqual(compare.call_count, 128)
        self.assertEqual(release.call_count, 64)
        self.assertEqual(result['extra_weight_allocations'], 0)
        operations.to_memory_config.assert_not_called()

    def test_native_math_change_fails_before_device_comparison(self):
        operations, model = self.fixture()
        model.layers[0].feed_forward.compute_kernel_config_decode.math_approx_mode = False
        with patch('packed_weight_check.compare_packed_weights') as compare:
            with self.assertRaisesRegex(ValueError, 'native target decode math'):
                admission.qualify_target_weights(operations, model)
        compare.assert_not_called()

    def test_weight_mismatch_releases_only_temporary_comparison_buffers(self):
        operations, model = self.fixture()
        with patch('packed_weight_check.compare_packed_weights', return_value='counts'), patch(
                'packed_weight_check.read_comparison', return_value=[dict(chip=0, exact=False)]), patch.object(
                admission, 'release_owned') as release:
            with self.assertRaisesRegex(AssertionError, 'layer 0'):
                admission.qualify_target_weights(operations, model)
        release.assert_called_once_with(operations, [])
