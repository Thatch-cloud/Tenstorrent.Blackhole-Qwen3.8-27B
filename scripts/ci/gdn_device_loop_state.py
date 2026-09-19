"""Device-loop GDN adapter with isolated compact state and active-slot publication."""

from gdn_multitoken_conv import addresses, release_owned, restore_prefix, run_projected
from gdn_prefix import validate_rows
from gdn_state_copy import copy_compact
from gdn_batched_conv import norm_batch_enabled, run_batched_projected


def validate_segments(segments, checkpoint, prefix, rows, slots):
    """Either today's single sequence, or one contiguous span per packed user."""
    if segments is None:
        if slots is not None:
            raise ValueError('Per-user carried state only applies to a packed block')
        if type(prefix) is not int or not 0 <= prefix <= rows:
            raise ValueError('Selected prefix must lie within the block')
        return None, None, None
    spans = tuple(tuple(span) for span in segments)
    if (not spans or any(len(span) != 2 for span in spans) or spans[0][0] != 0 or spans[-1][1] != rows
            or any(type(value) is not int for span in spans for value in span)
            or any(start >= stop for start, stop in spans)
            or any(spans[index][1] != spans[index + 1][0] for index in range(len(spans) - 1))):
        raise ValueError('Ordered contiguous packed segment spans covering the block required')
    checkpoints = tuple(checkpoint) if isinstance(checkpoint, (list, tuple)) else None
    prefixes = tuple(prefix) if isinstance(prefix, (list, tuple)) else None
    if (checkpoints is None or prefixes is None or slots is None
            or len(checkpoints) != len(spans) or len(prefixes) != len(spans) or len(slots) != len(spans)):
        raise ValueError('One checkpoint, accepted prefix and carried state per packed user required')
    for (start, stop), accepted in zip(spans, prefixes):
        if type(accepted) is not int or not 0 <= accepted <= stop - start:
            raise ValueError('Each accepted prefix must lie within its own segment')
    return spans, checkpoints, prefixes


class DeviceLoopState:
    def __init__(self, active, operations, kernels, compact_prologue=False, batch_conv=False, dma_windows=False,
                 packed_checkpoints=False, norm_batch=False, prefix_zero_reuse=False, defer_conv_publication=False,
                 norm_source_root=None, commit_only=False):
        if type(commit_only) is not bool or (commit_only and not (batch_conv and dma_windows and packed_checkpoints)):
            raise ValueError('Commit-only GDN requires packed batched DMA histories and an explicit decision owner')
        self.commit_only = commit_only
        if norm_source_root is not None and not norm_batch:
            raise ValueError('Norm source override requires batched normalization')
        self.norm_source_root = norm_source_root
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
        self.state = [] if commit_only else active.allocate()
        self.calls = self.checkpoint_calls = self.skipped_clones = 0
        self.native_addresses = [addresses(operations, value) for value in active.live]

    def _recurrence(self, projected, rows, prefix, deferred):
        """One recurrence run over `projected`, starting from whatever the layer's
        active state currently holds. The caller puts the right user there first."""
        operations, layer = self.operations, self.gdn
        self.active.save(self.entry)
        if not deferred:
            copy_compact(self.entry, self.state)
        operation = run_batched_projected if self.batch_conv else run_projected
        result = operation(layer.mesh, projected, self.entry[0], (self.entry if deferred else self.state)[1:],
            list(layer.tw['conv_taps']), layer.tw['dt_bias'], layer.tw['neg_exp_A'], layer.tw['norm_w'], self.kernels,
            **(dict(norm_batch=True) if self.norm_batch else {}),
            **(dict(norm_source_root=self.norm_source_root) if self.norm_source_root is not None else {}),
            **(dict(dma_windows=True) if self.dma_windows else {}),
            **(dict(packed_checkpoints=True) if self.packed_checkpoints else {}),
            **(dict(prefix_zero_reuse=True) if self.prefix_zero_reuse else {}),
            **(dict(defer_conv_publication=True) if deferred else {}),
            **(dict(conv_checkpoints=tuple(sorted({prefix, rows} - {0})), hoist_input=True) if self.compact_prologue else {}))
        if result.get('deferred_conv_publication', False) != deferred:
            raise AssertionError('Deferred convolution publication did not engage as selected')
        if result.get('norm_batch', False) != norm_batch_enabled(rows, self.norm_batch):
            raise AssertionError('Norm-batch recurrence adapter did not engage as selected')
        if result.get('prefix_zero_reuse', False) != (self.prefix_zero_reuse and rows > 1):
            raise AssertionError('Prefix zero reuse did not engage as selected')
        return result

    def decode(self, packed, checkpoint, prefix, *, segments=None, slots=None):
        """Run the block, optionally as several users packed along the row axis.

        Packed, the row axis carries one segment per user and the recurrence must
        restart from that user's own state at every boundary. Otherwise user B
        continues user A's recurrence, which is wrong for every row of B and leaves
        A advanced by the whole block.

        What does NOT change is where the weights are read. The input projection
        `_project_qkvzab_raw` runs ONCE across all rows here, and `finish_output`
        runs the single output projection over the concatenated rows afterwards.
        Only the recurrence, which is elementwise and carries no weights, runs once
        per segment. That is the whole point of packing: one pass over the weights
        serving every user.
        """
        rows = validate_rows(tuple(packed.shape))
        spans, checkpoints, prefixes = validate_segments(segments, checkpoint, prefix, rows, slots)
        if self.commit_only and rows == 1:
            raise ValueError('Commit-only GDN requires a multirow retained decision; T1 uses the native path')
        native = [self.gdn.rec_state, *self.gdn.conv_states]
        if self.gdn.B != 8 or [addresses(self.operations, value) for value in native] != self.native_addresses:
            raise ValueError('Native state binding changed')
        operations, layer = self.operations, self.gdn
        deferred = (self.defer_conv_publication or self.commit_only) and rows > 1
        if spans is not None:
            return self._decode_packed(packed, rows, spans, checkpoints, prefixes, slots, deferred)
        projected = layer._project_qkvzab_raw(packed, rows, operations.L1_MEMORY_CONFIG)
        result = None
        try:
            result = self._recurrence(projected, rows, prefix, deferred)
            result['owned'].append(projected)
            if not self.commit_only:
                restore_prefix(operations, result, self.entry, checkpoint, prefix)
                restore_prefix(operations, result, self.entry, self.state, rows)
                self.active.restore(self.state)
            result['commit_only_gdn'] = self.commit_only
            self.calls += 1
            self.checkpoint_calls += 1
            return result
        except BaseException:
            release_owned(operations, result['owned'] if result is not None else [projected])
            raise

    def _decode_packed(self, packed, rows, spans, checkpoints, prefixes, slots, deferred):
        operations, layer = self.operations, self.gdn
        if self.commit_only:
            raise ValueError('Commit-only GDN keeps no per-segment prefix state; packing needs the retained path')
        projected = layer._project_qkvzab_raw(packed, rows, operations.L1_MEMORY_CONFIG)
        results, owned, outputs = [], [projected], []
        try:
            for (start, stop), slot, point, accepted in zip(spans, slots, checkpoints, prefixes):
                width = stop - start
                # The layer active state becomes this user before its segment runs, so
                # the recurrence starts where that user left off rather than where the
                # user packed above it ended.
                self.active.restore(slot)
                piece = operations.slice(projected, (0, start, 0), (1, stop, projected.shape[-1]))
                owned.append(piece)
                result = self._recurrence(piece, width, accepted, deferred)
                owned.extend(result['owned'])
                restore_prefix(operations, result, self.entry, point, accepted)
                # and this user carried state advances by its own rows only
                restore_prefix(operations, result, self.entry, slot, width)
                outputs.append(result['output'])
                results.append(result)
            # The layer's active state is left holding the LAST segment's user. That is
            # safe only because every packed segment restores its own slot before it
            # runs; an unpacked decode on the same layer afterwards would inherit that
            # user, so packed and unpacked blocks must not be mixed within a request.
            output = outputs[0] if len(outputs) == 1 else operations.concat(outputs, dim=1)
            if len(outputs) > 1:
                owned.append(output)
            combined = dict(results[0])
            combined.update(output=output, owned=owned, segments=tuple(spans),
                            segment_results=tuple(results), commit_only_gdn=self.commit_only)
            self.calls += 1
            self.checkpoint_calls += len(spans)
            return combined
        except BaseException:
            release_owned(operations, owned)
            raise

    def close(self):
        release_owned(self.operations, [*self.entry, *self.state])
        self.entry.clear()
        self.state.clear()
