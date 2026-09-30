"""Device-loop GDN adapter with isolated compact state and active-slot publication."""

from gdn_multitoken_conv import addresses, release_owned, restore_prefix, run_projected
from gdn_prefix import validate_rows
from gdn_state_copy import batch_enabled, copy_compact, copy_compact_batch
from gdn_batched_conv import norm_batch_enabled, run_batched_projected
from gdn_user_batch import enabled as user_batch_enabled, min_users as user_batch_min_users
from gdn_user_batch_conv import run_user_batched_projected
from verify_trace_t1 import cut as verify_t1_cut, note as verify_t1_note

TILE = 32


def project_qkvzab_by_tile(operations, layer, packed, rows):
    """The block's one input-projection weight pass, at one native call within a tile or
    one call per 32-row tile beyond it, joined on the row axis.

    Beyond one tile, `layer._project_qkvzab_raw` takes the model's prefill 2D branch
    (tp_common.sharded_decode_matmul's `seq > TILE_SIZE` arm): DRAM output, ~13x slower
    at M=64 than the cost model. Every 32-row tile stays inside `S <= TILE_SIZE`, so it
    takes the same one-tile 1D decode arm the 2-user M1 block already runs (L1 output).
    The projection is a row-independent linear map, so two 32-row calls concatenated on
    the row axis are bit for bit one call over the same rows - token-exactness holds -
    at the cost of reading the projection weight twice instead of once per 64-row block
    (the accepted trade, same as the MLP and attention two-tile forms).

    `_project_qkvzab_raw` consumes and frees whatever tensor it is given (`packed` is
    never present in any owned/release list across every caller in this file or
    gdn_prefix.decode_projected, so the native op must be freeing it). Once `packed` is
    only ever handed over as tile slices, nothing else frees the whole tensor, so it is
    freed here right after the slices are cut - the same discipline
    two_tile_decode.TwoTileConcatHeads and TwoTileMLPForward use for their own inputs.

    NATIVE M3 (Lever N M3native graft, lever_n_m3native_patch.patch_gdn_tp). Once the
    graft is mounted, `layer.args` carries `attn_wo_decode_1d_progcfg_64` and
    `_project_qkvzab_raw`'s own gate (gdn/tp.py) widens to `S <= 2 * TILE_SIZE` and
    selects `gdn_qkvz_decode_1d_progcfg_64` (per_core_M 2) above one tile - so the cap
    here raises to 64 and the whole packed block goes through in ONE native call
    instead of two tiled ones. `layer` here has no `.args` in every CPU fixture that
    predates this graft, so the check goes through `getattr` and defaults to the
    original one-tile cap without it.
    """
    l1 = operations.L1_MEMORY_CONFIG
    native_m3 = hasattr(getattr(layer, 'args', None), 'attn_wo_decode_1d_progcfg_64')
    cap = 64 if native_m3 else TILE
    if rows <= cap:
        return layer._project_qkvzab_raw(packed, rows, l1)
    width = packed.shape[-1]
    slices = []
    try:
        for first in range(0, rows, TILE):
            slices.append(operations.slice(packed, (0, first, 0), (1, first + TILE, width)))
    except BaseException:
        for piece in slices:
            operations.deallocate(piece)
        raise
    operations.deallocate(packed)
    tiles = []
    try:
        for index, piece in enumerate(slices):
            try:
                tiles.append(layer._project_qkvzab_raw(piece, TILE, l1))
            except BaseException:
                for remaining in slices[index + 1:]:
                    operations.deallocate(remaining)
                raise
        return operations.concat(tiles, dim=1)
    finally:
        for value in tiles:
            operations.deallocate(value)


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


def resident_piece(operations, piece, owned):
    """One user's slice of the block projection, interleaved in L1: the placement the
    qualified T16 direct-window candidate is admitted on (gdn_direct_window_scope.run
    demands `projected.memory_config() == ttnn.L1_MEMORY_CONFIG` of every piece).

    Within one tile the model's decode projection is already L1 (the 1D decode matmul's
    output placement, gdn/tp.py:1036-1042) and every slice inherits it, so nothing moves
    and the slice is owned as before. Beyond one tile (the 64-row M3 block), the block's
    own projection (project_qkvzab_by_tile) now runs the same one-tile decode arm per
    32-row tile and joins the L1 halves, so every 16-row slice inherits L1 there too and
    this stays a no-op. The DRAM branch below is kept as the general fallback for
    whatever placement `_project_qkvzab_raw` actually returns - run 35504864400 (image
    v49) stopped at the scope's check when the model's own prefill branch (since
    replaced here) returned DRAM at M = 64: the slice was copied to L1 and the DRAM
    slice freed, same values, only the placement the kernel reads changes. The scope's
    check stays as it is; it is the qualification.
    """
    if piece.memory_config() == operations.L1_MEMORY_CONFIG:
        owned.append(piece)
        return piece
    try:
        resident = operations.to_memory_config(piece, operations.L1_MEMORY_CONFIG)
    finally:
        operations.deallocate(piece)
    owned.append(resident)
    return resident


class DeviceLoopState:
    def __init__(self, active, operations, kernels, compact_prologue=False, batch_conv=False, dma_windows=False,
                 packed_checkpoints=False, norm_batch=False, prefix_zero_reuse=False, defer_conv_publication=False,
                 norm_source_root=None, commit_only=False, users=None, user_batch=None, user_batch_min=None):
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
        # QWEN_FAST_GDN_USER_BATCH: one recurrence/norm launch for every packed user
        # instead of one pair per user (docs/gdn-user-batch-launches.md). Default OFF,
        # and only ever consulted by the deferred packed path - an explicit bool here
        # is the test seam, so a CPU fixture never reads the process environment.
        if user_batch is None:
            user_batch = user_batch_enabled()
        if type(user_batch) is not bool:
            raise ValueError('Explicit bool user-batched GDN option required')
        if user_batch and not (batch_conv and dma_windows and packed_checkpoints
                               and (defer_conv_publication or commit_only)):
            raise ValueError('User-batched GDN requires packed batched DMA histories and deferred publication')
        self.user_batch = user_batch
        # QWEN_FAST_GDN_USER_BATCH_MIN_USERS: how many packed users a block must carry
        # before the batched launch engages. Above the batched path's own cap it never
        # engages, which is the off switch that keeps the rest of an arm identical.
        if user_batch_min is None:
            user_batch_min = user_batch_min_users()
        if type(user_batch_min) is not int or type(user_batch_min) is bool or user_batch_min < 0:
            raise ValueError('User-batched GDN threshold must be a non-negative int')
        self.user_batch_min = user_batch_min
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

    def _recurrence(self, projected, rows, prefix, defer_publication, entry=None, slot=None):
        """One recurrence run over `projected`, starting from whatever the layer's
        active state currently holds. The caller puts the right user there first.

        `slot` is the packed path's launch reduction: when the caller already holds
        this segment's compact carried state in hand and has NOT restored it into
        the native active buffer, passing it here fetches `entry` with one
        compact-to-compact `copy_compact` launch instead of the two-launch round
        trip (`active.restore` then `active.save`) through the native buffer.
        Omitted - the single-user branch and the packed path's last segment both
        omit it - `entry` is read from the native buffer's current row 0, which is
        then the only source of truth."""
        operations, layer = self.operations, self.gdn
        entry = self.entry if entry is None else entry
        if slot is None:
            self.active.save(entry)
        else:
            copy_compact(slot, entry)
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
        (project_qkvzab_by_tile) runs across all rows here - one native call within a
        tile, one call per 32-row tile beyond it, joined on the row axis, never one per
        packed user - and `finish_output` runs the single output projection over the
        concatenated rows afterwards. Only the recurrence, which is elementwise and
        carries no weights, runs once per segment. That is the whole point of packing:
        the weight passes stay tied to tile count, not user count.

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
        projected = project_qkvzab_by_tile(operations, layer, packed, rows)
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
        # Below the threshold the flag is a no-op and the per-user path runs untouched,
        # so the configuration checks below must not fire on a block that never batches.
        batched = self.user_batch and len(spans) >= self.user_batch_min
        if batched and not (deferred and defer_publication):
            raise ValueError('User-batched GDN serves the deferred packed decode, which publishes nothing here')
        projected = project_qkvzab_by_tile(operations, layer, packed, rows)
        results, owned, outputs = [], [projected], []
        last = len(spans) - 1
        try:
            if batched:
                results = self._recurrence_user_batched(projected, spans, slots, entries, owned)
                for result in results:
                    owned.extend(result['owned'])
                    outputs.append(result['output'])
                return self._finish_packed(spans, results, outputs, owned)
            for index, ((start, stop), slot, point, accepted) in enumerate(zip(spans, slots, checkpoints, prefixes)):
                width = stop - start
                entry = None if entries is None else entries[index]
                if index == last:
                    # The layer active state becomes this user before its segment runs,
                    # so the recurrence starts where that user left off rather than
                    # where the user packed above it ended. A real restore, kept for
                    # the LAST segment only: the native buffer is documented (below)
                    # to be left holding the LAST segment's user once the round ends.
                    self.active.restore(slot)
                    recurrence_slot = None
                else:
                    # Every other segment's carried state moves straight into its own
                    # entry (one copy_compact launch): the recurrence reads `entry`,
                    # never the native buffer, and no segment before the last needs
                    # the native buffer to hold anything in particular, so writing
                    # `slot` into it first and reading it straight back out again -
                    # a full round trip through native state for a value already in
                    # hand - bought nothing.
                    recurrence_slot = slot
                piece = resident_piece(operations,
                                       operations.slice(projected, (0, start, 0), (1, stop, projected.shape[-1])), owned)
                result = self._recurrence(piece, width, accepted, defer_publication, entry, slot=recurrence_slot)
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
            # safe only because the LAST segment always restores its own slot into the
            # native buffer before it runs (every earlier segment's carried state moves
            # straight into its entry and never touches the native buffer at all); an
            # unpacked decode on the same layer afterwards would inherit that user, so
            # packed and unpacked blocks must not be mixed within a request.
            return self._finish_packed(spans, results, outputs, owned)
        except BaseException:
            release_owned(operations, owned)
            raise

    def _recurrence_user_batched(self, projected, spans, slots, entries, owned):
        """Every packed user's recurrence and fused norm/gate in ONE launch.

        The per-segment state moves are exactly the ones the per-user path makes, in the
        same order: each earlier segment's carried state goes straight into its own entry
        with one `copy_compact`, and the LAST segment restores its slot into the native
        buffer and saves it back out, which is what leaves the native buffer holding the
        last user - the documented invariant the unpacked path and `commit_user` rely on.
        Only the four launches that followed them become one.

        Nothing is published here: this path is the deferred packed decode, so no entry
        is written into the shared working state and no carry is advanced. That is also
        why hoisting the state moves ahead of the launch is sound - the per-user path's
        one shared write (`copy_compact(entry, self.state)`) does not happen at all when
        publication is deferred, and a hoisted version of it would hand every user the
        last user's convolution state.

        Under `gdn_state_copy.batch_enabled()` the earlier segments' compact moves also
        become one launch instead of one each (three of them at four packed users, on
        every one of the 48 GDN layers). They are already independent - each reads one
        user's own slot and writes that user's own entry, none of them touches the
        native buffer, and no two share a source or a destination - so collapsing them
        moves the same bytes to the same places. The LAST segment stays separate: it
        goes through the native buffer, whose full-slot geometry cannot share a page
        layout with a compact-to-compact transfer. Ordering against it does not matter
        for the same reason the segments do not order against each other.

        Under QWEN_FAST_VERIFY_T1=1 none of these moves run: the launch reads every carry
        in place. Read from the code, the invariant above has no reader - commit_user's DMA
        only WRITES native slot 0 (gdn_commit_dma.cpp), reads the entry only at prefix 0
        (never published), and verifier_engine clears residency around every packed step,
        so an unpacked decode restores its own carry first (verify_trace_t1, #3).
        QWEN_FAST_VERIFY_T1_SKIP=last_carry keeps the last segment's restore/save (and its
        entry read) while the others still read in place; =direct_carry keeps every move.
        """
        operations, layer = self.operations, self.gdn
        last = len(spans) - 1
        # QWEN_FAST_VERIFY_T1 (#3, verify_trace_t1): the launch reads each user's carry
        # directly. The entry below is a byte copy of that carry which only this launch
        # reads, and nothing in the verify trace writes a carry, so the recurrence and the
        # windows read the same bytes from the carry's own address and all five state moves
        # per layer go. The entries stay allocated for the commit DMA's list (read there only
        # at prefix 0, which never publishes); native slot 0 is no longer written here, and
        # nothing trusts it after a packed step (verify_trace_t1's docstring has the readers).
        direct = verify_t1_cut('direct_carry')
        direct_last = direct and verify_t1_cut('last_carry')
        pending, batched = [], [] if batch_enabled() else None
        for index, ((start, stop), slot) in enumerate(zip(spans, slots)):
            entry = entries[index]
            source = entry
            if index == last:
                if direct_last:
                    source = slot
                else:
                    self.active.restore(slot)
                    self.active.save(entry)
            elif direct:
                source = slot
            elif batched is not None:
                batched.append((slot, entry))
            else:
                copy_compact(slot, entry)
            piece = resident_piece(operations,
                                   operations.slice(projected, (0, start, 0), (1, stop, projected.shape[-1])), owned)
            pending.append((piece, source[0], source[1:]))
        # Every batched entry is read by the launch below, never before it, so deferring
        # the moves to here is exactly as early as they need to be.
        if batched:
            copy_compact_batch(batched)
        if direct:
            verify_t1_note('direct_carry')
        if direct_last:
            verify_t1_note('last_carry')
        results = run_user_batched_projected(layer.mesh, pending, list(layer.tw['conv_taps']),
            layer.tw['dt_bias'], layer.tw['neg_exp_A'], layer.tw['norm_w'], self.kernels, operations,
            prefix_zero_reuse=self.prefix_zero_reuse)
        if len(results) != len(spans):
            raise AssertionError('Batched GDN returned a result per packed user')
        for result, (start, stop) in zip(results, spans):
            if not result.get('user_batched', False) or not result.get('deferred_conv_publication', False):
                raise AssertionError('User-batched GDN did not engage as selected')
            if result.get('norm_batch', True) or result['states'].shape[0] != stop - start:
                raise AssertionError('User-batched GDN did not return this user own prefix geometry')
        return results

    def _finish_packed(self, spans, results, outputs, owned):
        """Join the segments' outputs and publish one block result, whichever path ran."""
        operations = self.operations
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

    def close(self):
        entries = list(self.segment_entries or ())
        release_owned(self.operations, [*self.entry, *self.state, *[value for entry in entries for value in entry]])
        self.entry.clear()
        self.state.clear()
        for entry in entries:
            entry.clear()
        self.segment_results = None
