from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

import fused_t16_scope as scope
import test_fused_t16_admission as fixtures
from test_dram_mlp_sharded import Tensor


class FusedScopeTests(unittest.TestCase):
    def setup_arm(self, stack, rows=16):
        operations, model = fixtures.FusionAdmissionTests().fixture()
        operations.bfloat16, operations.TILE_LAYOUT = 'bf16', 'tile'
        operations.linear = Mock(return_value='partial')
        for layer in model.layers:
            layer.feed_forward.args.mlp_w2_decode_1d_progcfg = 'native-down'
        manifest = dict(token_rows=rows)
        stack.enter_context(patch.object(scope, 'qualify_simulator', return_value=dict(kernels=[manifest], report_sha256='qualified')))
        stack.enter_context(patch.object(scope, 'qualify_target_weights', return_value=dict(passed=True)))
        projection = Mock(return_value='hidden')
        projection.manifest = manifest
        stack.enter_context(patch.object(scope, 'FusedProjection', return_value=projection))
        stack.enter_context(patch.object(scope, 'addresses', side_effect=lambda ops, tensor: (id(tensor), id(tensor))))
        collective = Mock(return_value='reduced')
        arm_type = scope.FusedT16Arm if rows == 16 else scope.FusedT32Arm
        return operations, model, projection, collective, arm_type(operations, model, collective)

    def test_t32_scope_preserves_t16_native_fallback_and_all_layer_ownership(self):
        with ExitStack() as stack:
            operations, model, projection, collective, arm = self.setup_arm(stack, rows=32)
            value = Tensor((1, 1, 32, 5120), 'bf16', 'l1')
            value.layout = 'tile'
            with arm.install():
                for layer in model.layers:
                    self.assertEqual(layer.feed_forward.forward(value), 'reduced')
                    self.assertEqual(layer.feed_forward.forward(Tensor((1, 1, 16, 5120), 'bf16', 'l1')), 'native')
            self.assertEqual(arm.audit['rows'], 32)
            self.assertEqual(arm.audit['hits'], [1] * 64)
            self.assertEqual(arm.audit['fallbacks'], [1] * 64)
            self.assertTrue(arm.audit['restored'])
            self.assertTrue(arm.audit['native_bindings_unchanged'])
            self.assertTrue(all(call.args == ('hidden',) for call in operations.deallocate.call_args_list))

    def test_all_layers_use_t16_only_and_preserve_borrowed_inputs(self):
        with ExitStack() as stack:
            operations, model, projection, collective, arm = self.setup_arm(stack)
            value = Tensor((1, 1, 16, 5120), 'bf16', 'l1')
            value.layout = 'tile'
            with arm.install():
                for layer in model.layers:
                    self.assertEqual(layer.feed_forward.forward(value), 'reduced')
                    self.assertEqual(layer.feed_forward.forward(Tensor((1, 1, 1, 5120), 'bf16', 'l1')), 'native')
            self.assertEqual(arm.audit['hits'], [1] * 64)
            self.assertEqual(arm.audit['fallbacks'], [1] * 64)
            self.assertTrue(arm.audit['restored'])
            self.assertTrue(arm.audit['native_bindings_unchanged'])
            self.assertEqual(collective.call_count, 64)
            self.assertTrue(all(call.args == ('hidden',) for call in operations.deallocate.call_args_list))
            self.assertEqual(operations.linear.call_args.kwargs['program_config'], 'native-down')

    def test_projection_failure_restores_forward_and_releases_conversion(self):
        with ExitStack() as stack:
            operations, model, projection, collective, arm = self.setup_arm(stack)
            projection.side_effect = RuntimeError('projection failed')
            value = Tensor((1, 1, 16, 5120), 'bf16', 'sharded')
            value.layout = 'tile'
            with self.assertRaisesRegex(RuntimeError, 'projection failed'):
                with arm.install():
                    model.layers[0].feed_forward.forward(value)
            self.assertTrue(arm.audit['restored'])
            operations.deallocate.assert_called_once()
            self.assertIsNot(operations.deallocate.call_args.args[0], value)
            collective.assert_not_called()

    def test_unscoped_execution_is_rejected(self):
        with ExitStack() as stack:
            operations, model, projection, collective, arm = self.setup_arm(stack)
            with self.assertRaisesRegex(RuntimeError, 'active instance scope'):
                arm.forward(0, None)
