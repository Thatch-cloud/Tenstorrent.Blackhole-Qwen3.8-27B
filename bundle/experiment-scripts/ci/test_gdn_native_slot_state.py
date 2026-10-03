from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_native_slot_state as candidate


class NativeSlotStateTests(unittest.TestCase):
    def fixture(self):
        native = [object() for operand in range(5)]
        operations = SimpleNamespace(L1_MEMORY_CONFIG='l1')
        layer = SimpleNamespace(rec_state=native[0], conv_states=native[1:], B=8, _stable_state=True,
            _project_qkvzab_raw=Mock(return_value='projected'), mesh='mesh',
            tw=dict(conv_taps=[1, 2, 3, 4], dt_bias='bias', neg_exp_A='decay', norm_w='weights'))
        state = SimpleNamespace(commit_only=True, batch_conv=True, dma_windows=True, packed_checkpoints=True,
            norm_batch=True, prefix_zero_reuse=False, native_publication_bound=True, gdn=layer,
            operations=operations, norm_source_root='/pinned', calls=0, checkpoint_calls=0,
            active=SimpleNamespace(save=Mock(), restore=Mock()),
            native_addresses=[(id(value), id(value)) for value in native])
        return state

    def test_t16_keeps_native_state_borrowed_and_skips_snapshot(self):
        state = self.fixture()
        output = dict(owned=['output'], native_slot_read=True)
        with patch.object(candidate, 'addresses', side_effect=lambda operations, value: (id(value), id(value))), \
                patch.object(candidate, 'execute', return_value=output):
            result = candidate.NativeSlotState.decode(state, SimpleNamespace(shape=(1, 16, 5120)), object(), 16)
        state.active.save.assert_not_called()
        state.active.restore.assert_not_called()
        self.assertEqual(result['owned'], ['output', 'projected'])
        self.assertEqual((state.calls, state.checkpoint_calls, state.native_slot_calls), (1, 1, 1))

    def test_missing_publication_binding_rejects_before_projection(self):
        state = self.fixture()
        state.native_publication_bound = False
        with self.assertRaisesRegex(ValueError, 'publication semantics'):
            candidate.NativeSlotState.decode(state, SimpleNamespace(shape=(1, 16, 5120)), object(), 0)
        state.gdn._project_qkvzab_raw.assert_not_called()

    def test_other_widths_use_existing_adapter(self):
        state = candidate.NativeSlotState.__new__(candidate.NativeSlotState)
        packed, checkpoint = SimpleNamespace(shape=(1, 8, 5120)), object()
        with patch.object(candidate.DeviceLoopState, 'decode', return_value='control') as decode:
            self.assertEqual(state.decode(packed, checkpoint, 3), 'control')
        decode.assert_called_once_with(packed, checkpoint, 3)
