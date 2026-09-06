"""Device-loop GDN adapter with isolated compact state and active-slot publication."""

from gdn_multitoken_conv import addresses, release_owned, restore_prefix, run_projected
from gdn_prefix import validate_rows
from gdn_state_copy import copy_compact


class DeviceLoopState:
    def __init__(self, active, operations, kernels):
        if not active.direct or active.gdn.B != 8 or not active.gdn._stable_state:
            raise ValueError('Audited stable B8 active snapshots required')
        self.active, self.gdn, self.operations, self.kernels = active, active.gdn, operations, kernels
        self.entry = active.allocate()
        self.state = active.allocate()
        self.calls = self.checkpoint_calls = self.skipped_clones = 0
        self.native_addresses = [addresses(operations, value) for value in active.live]

    def decode(self, packed, checkpoint, prefix):
        rows = validate_rows(tuple(packed.shape))
        if type(prefix) is not int or not 0 <= prefix <= rows:
            raise ValueError('Selected prefix must lie within the block')
        native = [self.gdn.rec_state, *self.gdn.conv_states]
        if self.gdn.B != 8 or [addresses(self.operations, value) for value in native] != self.native_addresses:
            raise ValueError('Native state binding changed')
        operations, layer = self.operations, self.gdn
        self.active.save(self.entry)
        copy_compact(self.entry, self.state)
        projected = layer._project_qkvzab_raw(packed, rows, operations.L1_MEMORY_CONFIG)
        result = None
        try:
            result = run_projected(layer.mesh, projected, self.entry[0], self.state[1:],
                list(layer.tw['conv_taps']), layer.tw['dt_bias'], layer.tw['neg_exp_A'], layer.tw['norm_w'], self.kernels)
            result['owned'].append(projected)
            restore_prefix(operations, result, self.entry, checkpoint, prefix)
            restore_prefix(operations, result, self.entry, self.state, rows)
            self.active.restore(self.state)
            self.calls += 1
            self.checkpoint_calls += 1
            return result
        except BaseException:
            release_owned(operations, result['owned'] if result is not None else [projected])
            raise

    def close(self):
        release_owned(self.operations, [*self.entry, *self.state])
        self.entry.clear()
        self.state.clear()
