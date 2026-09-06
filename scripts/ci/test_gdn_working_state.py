import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gdn_working_state import WorkingState


def native_gated(self, token):
    return recurrent_gated_delta_rule_decode_packed_ttnn(
        token, initial_state=self.rec_state, inplace_state=self.want_inplace)[0]


class WorkingStateTests(unittest.TestCase):
    def tensor(self, address):
        return SimpleNamespace(buffer_address=lambda: address)

    def fixture(self, alias=True):
        live = [self.tensor(index) for index in range(5)]
        compact = [self.tensor(index + 10) for index in range(5)]
        layer = SimpleNamespace(B=8, _stable_state=True, rec_state=live[0], conv_states=live[1:], want_inplace=True)
        active = SimpleNamespace(direct=True, gdn=layer, live=live, allocate=Mock(return_value=compact),
                                 save=Mock(), restore=Mock())
        operations = SimpleNamespace(get_device_tensors=lambda tensor: [tensor, tensor], copy=Mock(), deallocate=Mock())
        recurrence = Mock(side_effect=lambda *args, **kwargs: ("output", kwargs["initial_state"] if alias else live[0]))
        with patch.dict(native_gated.__globals__, recurrent_gated_delta_rule_decode_packed_ttnn=recurrence):
            with patch("gdn_working_state.gated_decode", return_value=SimpleNamespace(__func__=native_gated)):
                working = WorkingState(active, operations)
        return working, active, layer, operations

    def execute(self, layer, packed, tokens, checkpoint, operations, forward):
        self.assertEqual(layer.B, 1)
        outputs = []
        for index, token in enumerate(tokens):
            outputs.append(forward(token))
            checkpoint(index + 1)
        return outputs

    def test_compact_state_published_once_and_native_bindings_restored(self):
        working, active, layer, operations = self.fixture()
        conv = layer.conv_states
        checkpoint = Mock()
        with patch("gdn_working_state.decode_projected", side_effect=self.execute):
            self.assertEqual(working.decode("packed", [1, 2], checkpoint), ["output", "output"])
        self.assertEqual(working.calls, 2)
        self.assertEqual([call.args[0] for call in checkpoint.call_args_list], [0, 1, 2])
        active.save.assert_called_once_with(working.state)
        active.restore.assert_called_once_with(working.state)
        self.assertEqual(layer.B, 8)
        self.assertIs(layer.rec_state, active.live[0])
        self.assertIs(layer.conv_states, conv)

    def test_rejected_inplace_request_or_alias_never_publishes(self):
        for alias in (True, False):
            working, active, layer, operations = self.fixture(alias=alias)
            layer.want_inplace = not alias
            with patch("gdn_working_state.decode_projected", side_effect=self.execute):
                with self.assertRaises(AssertionError):
                    working.decode("packed", [1], Mock())
            active.restore.assert_not_called()
            self.assertEqual(layer.B, 8)
            self.assertIs(layer.rec_state, active.live[0])

    def test_failure_restores_bindings_without_publication(self):
        working, active, layer, operations = self.fixture()
        with patch("gdn_working_state.decode_projected", side_effect=RuntimeError("device")):
            with self.assertRaises(RuntimeError):
                working.decode("packed", [1], Mock())
        active.restore.assert_not_called()
        self.assertEqual(layer.B, 8)
        self.assertIs(layer.rec_state, active.live[0])

    def test_checkpoint_and_close_only_touch_compact_buffers(self):
        working, active, layer, operations = self.fixture()
        saved = list(working.state)
        with self.assertRaises(ValueError):
            working.save(["incomplete"])
        operations.copy.assert_not_called()
        working.save(["target"] * 5)
        self.assertEqual(operations.copy.call_count, 5)
        working.close()
        self.assertEqual([call.args[0] for call in operations.deallocate.call_args_list], saved)
        self.assertIs(layer.rec_state, active.live[0])


if __name__ == "__main__":
    unittest.main()
