import ast
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from gdn_prefix import decode_projected, independent_row, prepare_token_rows, split_gated_source, validate_reused_input, validate_rows


class PrefixTests(unittest.TestCase):
    def test_only_nonzero_rows_skip_cloning(self):
        layer, packed, tokens, operations, projection = self.fixture()
        operations.get_device_tensors = lambda tensor: [SimpleNamespace(buffer_address=lambda: id(tensor))] * 2
        skipped = Mock()
        decode_projected(layer, packed, tokens, Mock(), operations, clone_skipped=skipped)
        self.assertEqual(operations.clone.call_count, 1)
        self.assertEqual(skipped.call_count, 3)
        self.assertIs(layer._project_qkvzab_raw, projection)

    def test_alias_on_either_chip_is_rejected(self):
        source, destination = object(), object()
        for addresses in ((10, 21), (11, 20)):
            operations = SimpleNamespace(get_device_tensors=lambda tensor: [
                SimpleNamespace(buffer_address=lambda value=value: value)
                for value in ((10, 20) if tensor is source else addresses)])
            with self.assertRaises(ValueError):
                independent_row(operations, source, destination)

    def test_alias_failure_restores_projection_hook(self):
        layer, packed, tokens, operations, projection = self.fixture()
        operations.get_device_tensors = lambda tensor: [SimpleNamespace(buffer_address=lambda: 7)] * 2
        skipped = Mock()
        with self.assertRaises(ValueError):
            decode_projected(layer, packed, tokens, Mock(), operations, clone_skipped=skipped)
        skipped.assert_not_called()
        self.assertIs(layer._project_qkvzab_raw, projection)

    def test_reused_input_has_single_owner_and_one_slice(self):
        for rows in (1, 2, 4, 8, 16):
            layer, packed, tokens, operations, projection = self.fixture(rows)
            reused, owned = prepare_token_rows(operations, packed, reuse=True)
            self.assertEqual(len(reused), rows)
            self.assertEqual(len(owned), 1)
            self.assertTrue(all(value is owned[0] for value in reused))
            operations.slice.assert_called_once_with(packed, (0, 0, 0), (1, 1, 5120), memory_config="L1")

    def test_control_still_prepares_every_distinct_input_row(self):
        layer, packed, tokens, operations, projection = self.fixture()
        prepared, owned = prepare_token_rows(operations, packed)
        self.assertIs(prepared, owned)
        self.assertEqual(operations.slice.call_count, 4)

    def test_reused_input_source_rejects_new_data_consumers(self):
        source = '''def forward_decode(self, x):
    if len(x.shape) == 4:
        x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
    return self._project_qkvzab_raw(x, x.shape[-2], memory)
'''
        validate_reused_input(source)
        for changed in (source.replace("return self", "hidden = ttnn.add(x, other)\n    return self"),
                        source.replace("return self", "hidden = x\n    return self")):
            with self.assertRaises(ValueError):
                validate_reused_input(changed)

    def test_split_preserves_native_core_and_removes_only_tail(self):
        source = '''def forward_decode(self, token):
    gated = self.core(token)
    partial = self._row_proj(gated, tw["out"])
    ttnn.deallocate(gated)
    partial = ttnn.reshape(partial, shape)
    out = tt_all_reduce(partial)
    return out
'''
        tree = split_gated_source(source)
        self.assertEqual(ast.unparse(tree), "def forward_gated(self, token):\n    gated = self.core(token)\n    return gated")
        for changed in (source.replace("gated, tw", "other, tw"), source + "    self.changed()\n"):
            with self.assertRaises(ValueError):
                split_gated_source(changed)

    def fixture(self, rows=4):
        packed = SimpleNamespace(shape=(1, rows, 5120))
        tokens = [SimpleNamespace(shape=(1, 1, 5120)) for _ in range(rows)]
        projection = Mock(return_value=SimpleNamespace(shape=(1, rows, 6144)))
        layer = SimpleNamespace(_project_qkvzab_raw=projection)
        operations = SimpleNamespace(L1_MEMORY_CONFIG="L1", slice=Mock(), clone=Mock(), deallocate=Mock())
        layer.forward_decode = Mock(side_effect=lambda token: layer._project_qkvzab_raw(token, 1, "L1"))
        return layer, packed, tokens, operations, projection

    def test_supported_shapes_only(self):
        for rows in (1, 2, 4, 8, 16):
            self.assertEqual(validate_rows((1, rows, 5120)), rows)
        for shape in ((1, 3, 5120), (2, 4, 5120), (1, 4, 2560), (1, 1, 4, 5120)):
            with self.assertRaises(ValueError):
                validate_rows(shape)

    def test_single_projection_then_ordered_native_steps(self):
        layer, packed, tokens, operations, projection = self.fixture()
        checkpoint = Mock()
        self.assertEqual(len(decode_projected(layer, packed, tokens, checkpoint, operations)), 4)
        projection.assert_called_once_with(packed, 4, "L1")
        self.assertEqual([call.args[0] for call in checkpoint.call_args_list], [1, 2, 3, 4])
        self.assertEqual([call.args[1][1] for call in operations.slice.call_args_list], [0, 1, 2, 3])
        self.assertIs(layer._project_qkvzab_raw, projection)
        operations.deallocate.assert_called_once_with(projection.return_value)

    def test_native_failure_restores_hook(self):
        layer, packed, tokens, operations, projection = self.fixture()
        layer.forward_decode = Mock(side_effect=RuntimeError("device error"))
        with self.assertRaises(RuntimeError):
            decode_projected(layer, packed, tokens, Mock(), operations)
        self.assertIs(layer._project_qkvzab_raw, projection)
        operations.deallocate.assert_called_once()

    def test_missing_native_hook_is_not_a_pass(self):
        layer, packed, tokens, operations, projection = self.fixture()
        layer.forward_decode = Mock(return_value=None)
        with self.assertRaisesRegex(ValueError, "not engaged"):
            decode_projected(layer, packed, tokens, Mock(), operations)
        self.assertIs(layer._project_qkvzab_raw, projection)

    def test_invalid_input_rejected_before_projection(self):
        layer, packed, tokens, operations, projection = self.fixture()
        with self.assertRaises(ValueError):
            decode_projected(layer, packed, tokens[:1], Mock(), operations)
        projection.assert_not_called()

    def test_unexpected_native_batch_rejected(self):
        layer, packed, tokens, operations, projection = self.fixture()
        layer.forward_decode = Mock(side_effect=lambda token: layer._project_qkvzab_raw(token, 2, "L1"))
        with self.assertRaises(ValueError):
            decode_projected(layer, packed, tokens, Mock(), operations)
        self.assertIs(layer._project_qkvzab_raw, projection)


if __name__ == "__main__":
    unittest.main(verbosity=2)
