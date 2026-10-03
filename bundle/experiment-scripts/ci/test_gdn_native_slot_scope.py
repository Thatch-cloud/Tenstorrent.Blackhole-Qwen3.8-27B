from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_device_loop_state
import gdn_native_slot_scope as candidate
import verifier_engine


class NativeSlotScopeTests(unittest.TestCase):
    def fixture(self):
        return SimpleNamespace(mesh_device=SimpleNamespace(shape=(1, 2)), layers=[
            SimpleNamespace(attention=object(), is_full_attention=index >= 48) for index in range(64)])

    def test_pairs_all_layers_and_restores_bindings(self):
        model, operations = self.fixture(), object()
        arm = candidate.NativeSlotArm(operations, model)
        records = []
        def construct(active, operations, *args, **kwargs):
            entry = [object() for operand in range(5)]
            native = [object() for operand in range(5)]
            records.append(entry + [SimpleNamespace(shape=(16, 24, 128, 128))] + [object() for operand in range(4)]
                + native + [object() for operand in range(5)])
            return SimpleNamespace(entry=entry, native_addresses=[(id(value), id(value)) for value in native],
                gdn=active.gdn, native_slot_calls=2)
        with patch.object(candidate, 'qualify', return_value={'synthetic': True}), \
                patch.object(candidate, 'addresses', side_effect=lambda operations, value: (id(value), id(value))), \
                patch.object(candidate, 'NativeSlotState', side_effect=construct), \
                patch.object(candidate, 'native_prepare', return_value='paired') as prepare:
            with arm.install():
                for layer in model.layers[:48]:
                    state = gdn_device_loop_state.DeviceLoopState(SimpleNamespace(gdn=layer.attention), operations)
                    self.assertTrue(state.native_publication_bound)
                self.assertEqual(verifier_engine.prepare(model.mesh_device, records, 0), 'paired')
                prepare.assert_called_once_with(model.mesh_device, records, 0, experimental=True)
                with self.assertRaises(ValueError):
                    verifier_engine.prepare(model.mesh_device, records[:-1], 0)
        self.assertIs(gdn_device_loop_state.DeviceLoopState, candidate.DeviceLoopState)
        self.assertIs(verifier_engine.prepare, candidate.compact_prepare)
        summary = arm.summary()
        self.assertTrue(summary['restored'])
        self.assertEqual(summary['calls_by_layer'], [2] * 48)
        self.assertEqual(summary['publication_prefixes'], [0])

    def test_exception_restores_and_scope_cannot_be_reused(self):
        arm = candidate.NativeSlotArm(object(), self.fixture())
        with patch.object(candidate, 'qualify', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with arm.install():
                    raise RuntimeError('injected')
            with self.assertRaises(ValueError):
                with arm.install():
                    pass
        self.assertTrue(arm.restored)
        self.assertIs(verifier_engine.prepare, candidate.compact_prepare)
