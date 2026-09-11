from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_native_slot_projected as candidate


class NativeSlotProjectedTests(unittest.TestCase):
    def test_composes_native_readers_and_retains_only_owned_buffers(self):
        projected = SimpleNamespace(shape=(1, 16, 8256))
        state = [SimpleNamespace(shape=(8, 24, 128, 128))] + [SimpleNamespace(shape=(1, 8, 5120)) for slot in range(4)]
        taps = [SimpleNamespace(shape=(1, 1, 5120)) for slot in range(4)]
        windows, packed, outputs = [object() for slot in range(4)], [object() for operand in range(3)], [object() for operand in range(3)]
        weights, z = object(), object()
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram', transformer=SimpleNamespace(gdn_decode_conv_gates=Mock(return_value=packed)),
            slice=Mock(return_value=z), to_memory_config=Mock(return_value=weights))
        with patch.object(candidate, 'addresses', side_effect=lambda operations, tensor: (id(tensor), id(tensor))), \
                patch.object(candidate, 'build_windows', return_value=windows) as build, \
                patch.object(candidate, 'recurrence', return_value=outputs) as recur, \
                patch.object(candidate, 'release_owned') as release:
            result = candidate.execute(operations, 'mesh', projected, state, taps, 'bias', 'decay', weights,
                root='/pinned', experimental=True)
            build.assert_called_once_with('mesh', projected, state[1:], experimental=True)
            recur.assert_called_once_with(operations, 'mesh', *packed, state[0], z=z, norm_w=weights,
                root='/pinned', experimental=True)
            self.assertEqual(result['owned'], [*windows, *packed, z, *outputs])
            self.assertIs(result['states'], outputs[1])
            self.assertIs(result['packed_conv_states'], windows)
            self.assertTrue(result['commit_only_gdn'] and result['native_slot_read'])
            release.assert_not_called()
            recur.side_effect = RuntimeError('recurrence failure')
            with self.assertRaisesRegex(RuntimeError, 'recurrence failure'):
                candidate.execute(operations, 'mesh', projected, state, taps, 'bias', 'decay', weights,
                    root='/pinned', experimental=True)
            release.assert_called_once_with(operations, [*windows, *packed, z])
