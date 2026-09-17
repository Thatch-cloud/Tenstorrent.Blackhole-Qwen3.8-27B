from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from prefill_gdn_checkpoint import PrefillGDNCheckpoint


class PrefillGDNCheckpointTests(unittest.TestCase):
    def fixture(self):
        values = []
        def tensor():
            index = len(values)
            value = SimpleNamespace(shape=(1, 1, 32, 32), dtype='bf16', layout='tile',
                data=[index, index + 1000], parts=[SimpleNamespace(buffer_address=lambda index=index: index),
                    SimpleNamespace(buffer_address=lambda index=index: index + 10000)])
            values.append(value)
            return value
        layers = [SimpleNamespace(B=1, _stable_state=True, rec_state=tensor(),
            conv_states=[tensor() for unused in range(4)], conv_carry=tensor()) for unused in range(48)]
        buffers = [tensor() for unused in range(288)]
        operations = SimpleNamespace(get_device_tensors=lambda value: value.parts,
            synchronize_device=Mock(), copy=Mock(side_effect=lambda source, destination:
                setattr(destination, 'data', list(source.data))))
        return operations, layers, buffers

    def test_carry_and_all_states_restored_without_rebinding(self):
        operations, layers, buffers = self.fixture()
        checkpoint = PrefillGDNCheckpoint(operations, 'mesh', layers, buffers)
        expected = [list(value.data) for value in checkpoint.live]
        checkpoint.capture(2048)
        for value in checkpoint.live:
            value.data = [-1, -2]
        checkpoint.restore(2048)
        self.assertEqual([value.data for value in checkpoint.live], expected)
        self.assertEqual(operations.copy.call_count, 576)
        self.assertEqual(operations.synchronize_device.call_count, 4)
        self.assertEqual(layers[-1].conv_carry.data, expected[-1])

    def test_wrong_boundary_and_changed_storage_rejected(self):
        operations, layers, buffers = self.fixture()
        checkpoint = PrefillGDNCheckpoint(operations, 'mesh', layers, buffers)
        with self.assertRaises(ValueError):
            checkpoint.restore(2048)
        checkpoint.capture(2048)
        with self.assertRaises(ValueError):
            checkpoint.restore(4096)
        layers[0].rec_state.parts[0].buffer_address = lambda: -1
        with self.assertRaises(ValueError):
            checkpoint.restore(2048)

    def test_alias_and_decode_slot_state_rejected(self):
        operations, layers, buffers = self.fixture()
        buffers[0] = layers[0].rec_state
        with self.assertRaises(ValueError):
            PrefillGDNCheckpoint(operations, 'mesh', layers, buffers)
        layers[0].B = 8
        with self.assertRaises(ValueError):
            PrefillGDNCheckpoint(operations, 'mesh', layers, buffers)

    def test_partial_restore_poisoned(self):
        operations, layers, buffers = self.fixture()
        checkpoint = PrefillGDNCheckpoint(operations, 'mesh', layers, buffers)
        checkpoint.capture(2048)
        operations.copy.side_effect = RuntimeError('copy failed')
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            checkpoint.restore(2048)
        self.assertEqual(checkpoint.phase, 'failed')
        with self.assertRaises(ValueError):
            checkpoint.capture(2048)


if __name__ == '__main__':
    unittest.main()
