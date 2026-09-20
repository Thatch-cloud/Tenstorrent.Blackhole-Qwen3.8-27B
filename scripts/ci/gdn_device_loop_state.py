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
                 norm_source_root=None, commit_only=False, users=None):
        if type(commit_only) is not bool or (commit_only and not (batch_conv and dma_windows and packed_checkpoints)):
            raise ValueError('Commit-only GDN requires packed batched DMA histories and an explicit decision owner')
        self.commit_only = commit_only
        if users is not None and (type(users) is not int or users < 1 or not commit_only):
            raise ValueError('Per-user entries serve the deferred packed decode, whose decisions the retained block owns')
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
        # Deferred packed decode: each user's block-start state must survive its own
        # segment, because that user's commit rebuilds any accepted prefix from ITS
        # entry and ITS histories once the readback has chosen one. A single shared
        # entry would hold only the last user's. Allocated here, before any trace.
        self.segment_entries = None if users is None else tuple(active.allocate() for unused in range(users))
        self.entry = [] if users is not None else active.allocate()
        self.state = [] if commit_only else active.allocate()
        self.segment_results = None
        self.calls = self.checkpoint_calls = self.skipped_clones = 0
        self.native_addresses = [addresses(operations, value) for value in active.live]

    def _recurrence(self, projected, rows, prefix, defer_publication, entry=None):
        """One recurrence run over `projected`, starting from whatever the layer's
        active state currently holds. The caller puts the right user there first."""
        operations, layer = self.operations, self.gdn
        entry = self.entry if entry is None else entry
        self.active.save(entry)
        if not defer_publication:
            copy_compact(entry, self.state)
        operation = run_batched_projected if self.batch_conv else run_projected
        result = operation(layer.mesh, projected, entry[0], (entry if defer_publication else self.state)[1:],
            list(layer.tw['conv_taps']), layer.tw['dt_bias'], layer.tw['neg_exp_A'], layer.tw['norm_w'], self.kernels,
            **(dict(norm_batch=True) if self.norm_batch else {}),
            **(dict(norm_source_root=self.norm_source_root) if self.norm_source_root is not None else {}),
            **(dict(dma_windows=True) if self.dma_windows else {}),
            **(dict(packed_checkpoints=True) if self.packed_checkpoints else {}),
            **(dict(prefix_zero_reuse=True) if self.prefix_zero_reuse else {}),
            **(dict(defer_conv_publication=True) if defer_publication else {}),
            **(dict(conv_checkpoints=tuple(sorted({prefix, rows} - {0})), hoist_input=True) if self.compact_prologue else {}))
        if result.get('deferred_conv_publication', False) != defer_publication:
            raise AssertionError('Deferred convolution publication did not engage as selected')
        if result.get('norm_batch', False) != norm_batch_enabled(rows, self.norm_batch):
            raise AssertionError('Norm-batch recurrence adapter did not engage as selected')
        if result.get('prefix_zero_reuse', False) != (self.prefix_zero_reuse and rows > 1):
            raise AssertionError('Prefix zero reuse did not engage as selected')
        return result

    def decode(self, packed, checkpoint, prefix, *, segments=None, slots=None, deferred=False):
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

        `deferred` is the packed serving step: the accepted prefixes are known only
        after the verify readback, so nothing is restored or advanced here. Every
        user keeps its own entry, `segment_results` holds one result per segment in
        segment order, and the retained block's `commit_user` later writes each
        user's accepted prefix into that user's carry.
        """
        rows = validate_rows(tuple(packed.shape))
        spans, checkpoints, prefixes = validate_segments(segments, checkpoint, prefix, rows, slots)
        if type(deferred) is not bool or (deferred and spans is None):
            raise ValueError('Deferred decisions apply to a packed block; a single sequence decides at decode')
        if self.commit_only and rows == 1:
            raise ValueError('Commit-only GDN requires a multirow retained decision; T1 uses the native path')
        native = [self.gdn.rec_state, *self.gdn.conv_states]
        if self.gdn.B != 8 or [addresses(self.operations, value) for value in native] != self.native_addresses:
            raise ValueError('Native state binding changed')
        operations, layer = self.operations, self.gdn
        defer_publication = (self.defer_conv_publication or self.commit_only) and rows > 1
        if spans is not None:
            return self._decode_packed(packed, rows, spans, checkpoints, prefixes, slots, defer_publication, deferred)
        projected = layer._project_qkvzab_raw(packed, rows, operations.L1_MEMORY_CONFIG)
        result = None
        try:
            result = self._recurrence(projected, rows, prefix, defer_publication)
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

    def _decode_packed(self, packed, rows, spans, checkpoints, prefixes, slots, defer_publication, deferred):
        operations, layer = self.operations, self.gdn
        if self.commit_only and not deferred:
            raise ValueError('Commit-only GDN keeps no per-segment prefix state; packing needs the retained path')
        entries = self.segment_entries
        if deferred and (entries is None or len(entries) != len(spans)):
            raise ValueError('Deferred packed decode needs one entry per packed user, allocated at construction')
        projected = layer._project_qkvzab_raw(packed, rows, operations.L1_MEMORY_CONFIG)
        results, owned, outputs = [], [projected], []
        try:
            for index, ((start, stop), slot, point, accepted) in enumerate(zip(spans, slots, checkpoints, prefixes)):
                width = stop - start
                # The layer active state becomes this user before its segment runs, so
                # the recurrence starts where that user left off rather than where the
                # user packed above it ended.
                self.active.restore(slot)
                piece = operations.slice(projected, (0, start, 0), (1, stop, projected.shape[-1]))
                owned.append(piece)
                result = self._recurrence(piece, width, accepted, defer_publication,
                                          None if entries is None else entries[index])
                owned.extend(result['owned'])
                if not deferred:
                    restore_prefix(operations, result, self.entry, point, accepted)
                    # and this user carried state advances by its own rows only
                    restore_prefix(operations, result, self.entry, slot, width)
                # Deferred, the carry is left exactly as this segment found it: the
                # commit DMA writes the accepted prefix into it once known, and a
                # prefix of zero then has nothing to write, which is only true because
                # nothing advanced the carry here.
                outputs.append(result['output'])
                results.append(result)
            # The layer's active state is left holding the LAST segment's user. That is
            # safe only because every packed segment restores its own slot before it
            # runs; an unpacked decode on the same layer afterwards would inherit that
            # user, so packed and unpacked blocks must not be mixed within a request.
            output = outputs[0] if len(outputs) == 1 else operations.concat(outputs, dim=1)
            if len(outputs) > 1:
                owned.append(output)
            pieces = tuple(results)
            combined = dict(results[0])
            combined.update(output=output, owned=owned, segments=tuple(spans),
                            segment_results=pieces, commit_only_gdn=self.commit_only)
            self.segment_results = pieces
            self.calls += 1
            self.checkpoint_calls += len(spans)
            return combined
        except BaseException:
            release_owned(operations, owned)
            raise

    def close(self):
        entries = list(self.segment_entries or ())
        release_owned(self.operations, [*self.entry, *self.state, *[value for entry in entries for value in entry]])
        self.entry.clear()
        self.state.clear()
        for entry in entries:
            entry.clear()
        self.segment_results = None
