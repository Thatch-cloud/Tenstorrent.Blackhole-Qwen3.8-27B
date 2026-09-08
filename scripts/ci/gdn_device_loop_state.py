"""Device-loop GDN adapter with isolated compact state and active-slot publication."""

from gdn_multitoken_conv import addresses, release_owned, restore_prefix, run_projected
from gdn_prefix import validate_rows
from gdn_state_copy import copy_compact
from gdn_batched_conv import norm_batch_enabled, run_batched_projected


class DeviceLoopState:
    def __init__(self, active, operations, kernels, compact_prologue=False, batch_conv=False, dma_windows=False,
                 packed_checkpoints=False, norm_batch=False, prefix_zero_reuse=False, defer_conv_publication=False):
        if type(defer_conv_publication) is not bool or (defer_conv_publication and not (batch_conv and dma_windows and packed_checkpoints)):
            raise ValueError('Deferred publication requires packed batched DMA checkpoints and explicit bool')
        self.defer_conv_publication = defer_conv_publication
        if type(prefix_zero_reuse) is not bool or (prefix_zero_reuse and not packed_checkpoints):
            raise ValueError('Prefix zero reuse requires explicit bool and packed checkpoints')
        self.prefix_zero_reuse = prefix_zero_reuse
        if type(norm_batch) is not bool or (norm_batch and not packed_checkpoints):
            raise ValueError('Norm batching requires explicit bool and packed checkpoints')
        if dma_windows and not batch_conv:
            raise ValueError('DMA windows require batched convolution')
        if packed_checkpoints and not dma_windows:
            raise ValueError('Packed checkpoints require DMA-built convolution windows')
        if not active.direct or active.gdn.B != 8 or not active.gdn._stable_state:
            raise ValueError('Audited stable B8 active snapshots required')
        self.active, self.gdn, self.operations, self.kernels = active, active.gdn, operations, kernels
        self.compact_prologue = compact_prologue
        self.batch_conv, self.dma_windows = batch_conv, dma_windows
        self.packed_checkpoints = packed_checkpoints
        self.norm_batch = norm_batch
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
        deferred = self.defer_conv_publication and rows > 1
        self.active.save(self.entry)
        if not deferred:
            copy_compact(self.entry, self.state)
        projected = layer._project_qkvzab_raw(packed, rows, operations.L1_MEMORY_CONFIG)
        result = None
        try:
            operation = run_batched_projected if self.batch_conv else run_projected
            result = operation(layer.mesh, projected, self.entry[0], (self.entry if deferred else self.state)[1:],
                list(layer.tw['conv_taps']), layer.tw['dt_bias'], layer.tw['neg_exp_A'], layer.tw['norm_w'], self.kernels,
                **(dict(norm_batch=True) if self.norm_batch else {}),
                **(dict(dma_windows=True) if self.dma_windows else {}),
                **(dict(packed_checkpoints=True) if self.packed_checkpoints else {}),
                **(dict(prefix_zero_reuse=True) if self.prefix_zero_reuse else {}),
                **(dict(defer_conv_publication=True) if deferred else {}),
                **(dict(conv_checkpoints=tuple(sorted({prefix, rows} - {0})), hoist_input=True) if self.compact_prologue else {}))
            result['owned'].append(projected)
            if result.get('deferred_conv_publication', False) != deferred:
                raise AssertionError('Deferred convolution publication did not engage as selected')
            if result.get('norm_batch', False) != norm_batch_enabled(rows, self.norm_batch):
                raise AssertionError('Norm-batch recurrence adapter did not engage as selected')
            if result.get('prefix_zero_reuse', False) != (self.prefix_zero_reuse and rows > 1):
                raise AssertionError('Prefix zero reuse did not engage as selected')
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
