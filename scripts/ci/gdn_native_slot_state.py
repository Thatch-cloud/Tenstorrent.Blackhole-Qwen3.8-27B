"""Unselected T16 commit-only adapter; requires paired native-slot publication wiring."""

from pathlib import Path

from gdn_device_loop_state import DeviceLoopState
from gdn_multitoken_conv import addresses, release_owned
from gdn_native_slot_projected import execute


class NativeSlotState(DeviceLoopState):
    def decode(self, packed, checkpoint, prefix):
        if tuple(packed.shape) != (1, 16, 5120):
            return super().decode(packed, checkpoint, prefix)
        if not (self.commit_only and self.batch_conv and self.dma_windows
                and self.packed_checkpoints and self.norm_batch) or self.prefix_zero_reuse:
            raise ValueError('Native-slot T16 requires commit-only packed batched GDN and paired zero publication')
        if type(prefix) is not int or not 0 <= prefix <= 16:
            raise ValueError('T16 accepted prefix required')
        if not getattr(self, 'native_publication_bound', False):
            raise ValueError('Native-slot state must be explicitly bound to matching publication semantics')
        native = [self.gdn.rec_state, *self.gdn.conv_states]
        if (self.gdn.B != 8 or not self.gdn._stable_state
                or [addresses(self.operations, value) for value in native] != self.native_addresses):
            raise ValueError('Native state bindings changed')
        projected = self.gdn._project_qkvzab_raw(packed, 16, self.operations.L1_MEMORY_CONFIG)
        result = None
        try:
            result = execute(self.operations, self.gdn.mesh, projected, native,
                list(self.gdn.tw['conv_taps']), self.gdn.tw['dt_bias'], self.gdn.tw['neg_exp_A'], self.gdn.tw['norm_w'],
                root=self.norm_source_root or Path('/opt/tt-metal'), experimental=True)
            result['owned'].append(projected)
            self.calls += 1
            self.checkpoint_calls += 1
            self.native_slot_calls = getattr(self, 'native_slot_calls', 0) + 1
            return result
        except BaseException:
            release_owned(self.operations, result['owned'] if result is not None else [projected])
            raise
