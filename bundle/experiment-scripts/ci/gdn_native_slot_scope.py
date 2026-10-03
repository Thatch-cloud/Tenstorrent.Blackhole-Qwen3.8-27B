"""Single-request reversible target-only T16 direct-state and publication pairing."""

from contextlib import contextmanager
from pathlib import Path

from gdn_commit_dma import prepare as compact_prepare
from gdn_device_loop_state import DeviceLoopState
from gdn_multitoken_conv import addresses
from gdn_native_slot_gate import qualify
from gdn_native_slot_publication import prepare as native_prepare
from gdn_native_slot_state import NativeSlotState


class NativeSlotArm:
    def __init__(self, operations, model):
        self.operations, self.model = operations, model
        self.layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
        if len(model.layers) != 64 or len(self.layers) != 48 or tuple(model.mesh_device.shape) != (1, 2):
            raise ValueError('Complete two-chip target with 48 GDN layers required')
        self.states, self.publications = [], []
        self.used = self.restored = False
        self.evidence = None

    @contextmanager
    def install(self):
        import gdn_device_loop_state as state_module
        import verifier_engine
        if self.used or state_module.DeviceLoopState is not DeviceLoopState or verifier_engine.prepare is not compact_prepare:
            raise ValueError('Fresh scope with unchanged verifier and state bindings required')
        self.evidence = qualify(Path(__file__).parent)
        self.used = True
        def construct(active, operations, *args, **kwargs):
            if operations is not self.operations or not any(active.gdn is layer for layer in self.layers):
                raise ValueError('Only the bound target layers may use native-slot state')
            state = NativeSlotState(active, operations, *args, **kwargs)
            state.native_publication_bound = True
            self.states.append(state)
            return state
        def prepare(mesh, layers, prefix):
            if not layers or len(layers[0]) != 20:
                raise ValueError('Complete publication records required')
            if layers[0][5].shape[0] != 16:
                return compact_prepare(mesh, layers, prefix)
            if mesh is not self.model.mesh_device or len(layers) != 48:
                raise ValueError('All bound T16 target layers required')
            matched = []
            for layer in layers:
                entry = [addresses(self.operations, value) for value in layer[:5]]
                candidates = [state for state in self.states if len(state.entry) == 5
                    and [addresses(self.operations, value) for value in state.entry] == entry]
                if len(candidates) != 1:
                    raise ValueError('Publication must match one live native-slot entry')
                state = candidates[0]
                if ([addresses(self.operations, value) for value in layer[10:15]] != state.native_addresses
                        or not getattr(state, 'native_slot_calls', 0)):
                    raise ValueError('Publication must bind the native state actually verified')
                matched.append(id(state.gdn))
            if len(set(matched)) != 48:
                raise ValueError('Every target GDN layer must appear once')
            self.publications.append(prefix)
            return native_prepare(mesh, layers, prefix, experimental=True)
        state_module.DeviceLoopState, verifier_engine.prepare = construct, prepare
        try:
            yield self
        finally:
            unchanged = state_module.DeviceLoopState is construct and verifier_engine.prepare is prepare
            if state_module.DeviceLoopState is construct:
                state_module.DeviceLoopState = DeviceLoopState
            if verifier_engine.prepare is prepare:
                verifier_engine.prepare = compact_prepare
            self.restored = unchanged
            if not unchanged:
                raise RuntimeError('Native-slot scope bindings changed unexpectedly')

    def summary(self):
        counts = [sum(getattr(state, 'native_slot_calls', 0) for state in self.states if state.gdn is layer)
            for layer in self.layers]
        return dict(restored=self.restored, qualification=self.evidence, calls_by_layer=counts,
            publication_prefixes=list(self.publications), rows=16,
            scope='Target-only T16 direct native reads; paired zero-prefix publication; other widths unchanged')
