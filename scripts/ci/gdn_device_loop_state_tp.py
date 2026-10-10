"""gdn_device_loop_state.DeviceLoopState with the packed block's split and merge as one DMA launch each (tp4/vglue V2).

gdn_device_loop_state.py is held unedited by the K5 evidence (test_gdn_seq_block.py), and the two places that cost a GDN
layer 4.9 ms per replay at four users are inside it: `_recurrence_user_batched` cuts the (1, 64, W) block projection into one
piece per user (a tile Slice for users 0 and 2, an untilize / slice / tilize round trip for users 1 and 3) and
`_finish_packed` joins the four gated outputs (four untilizes, a concat, a tilize). The pinned `_decode_packed` calls both
through `self`, so this twin subclasses the pinned class and overrides those two methods; tp_addresses.install() binds it
in place of the pinned class at QWEN_FAST_TP=4 only (model_batch imports DeviceLoopState lazily).

QWEN_FAST_TP4_GDN_GLUE unset or 0: every method is the pinned one (nothing here runs). QWEN_FAST_TP4_GDN_GLUE=1: the block's
pieces come from one gdn_rows_dma_tp launch and the merged output from another; the state moves, the launch that follows and
the result dictionaries are the pinned ones. A block or a placement the launches are not written for takes the pinned path,
logged as FALLBACK and counted (nothing is guessed).

QWEN_FAST_TP4_GDN_BLOCK_CONV=1 (needs the glue flag) additionally runs the layer's convolution and gates as one block launch
(gdn_block_conv_tp) instead of one per user.

QWEN_FAST_GDN_PAIR_SLICE=1 (V2 off): the packed decode's odd-user half-tile slices share one row-major conversion of the projection
instead of one each (gdn_pair_slice_tp); QWEN_FAST_GDN_DISPATCH_DIAG=1 logs each layer's launch order (a host log, no op changes).
Both go through `_decode_packed`, which hands the pinned body a proxy for `self.operations` for that one call; with both off it is
the pinned method.

QWEN_FAST_TP4_VGLUE_AUDIT=1 (gate arms only): beside each launch this builds what the served path would have produced and holds
it, outside `owned`, for tp4_vglue.audit_round to compare after the replay (the served path is never the one whose outputs the
layer uses).
"""

import os

import gdn_block_conv_tp
import gdn_pair_slice_tp
import gdn_device_loop_state as pinned
import gdn_rows_dma_tp as rows_dma
import tp4_vglue
import verify_trace_t2
from tp_addresses import release_owned
from gdn_state_copy import batch_enabled, copy_compact, copy_compact_batch
from gdn_user_batch_conv import run_user_batched_projected
from verify_trace_t1 import cut as verify_t1_cut, note as verify_t1_note

USERS = 4
PIECE_ROWS = rows_dma.USER_ROWS
# The octo block's users: eight rows each, a quarter tile. The glue kernels move half tiles only, so such spans take the served path and say so (tp4_vglue.EIGHT_ROW_SERVED).
OCTO_PIECE_ROWS = 8
OCTO_USERS = 8
EIGHT_ROW_REASON = 'eight-row users (the octo block): the glue kernels move 16-row half tiles, the served path runs'
GLUE8_FLAG = 'QWEN_FAST_OCTO_GLUE8'


def glue8_requested():
    """QWEN_FAST_OCTO_GLUE8 is set to anything but '0' (octo_glue8.enabled says whether it is valid). Read here without importing the module, so with the flag off nothing of the
    quarter-tile movers is imported and every path below is the one that ran before they existed."""
    return os.environ.get(GLUE8_FLAG, '0') != '0'


def _pinned_class():
    """The pinned class, whichever way this module was imported (tp_addresses.install rebinds the pinned module's own name to
    this class, so a module first imported after install must not take itself for its base)."""
    base = pinned.DeviceLoopState
    if getattr(base, '__module__', None) == __name__:
        base = base.__mro__[1]
    return base


class DeviceLoopState(_pinned_class()):
    def glue_problem(self, projected, spans):
        """Why the split and merge launches cannot serve this block (None when they can)."""
        operations = self.operations
        shape = tuple(projected.shape)
        users = len(spans)
        if users == OCTO_USERS and [tuple(span) for span in spans] == [(user * OCTO_PIECE_ROWS, (user + 1) * OCTO_PIECE_ROWS) for user in range(users)]:
            if not self.eight_native(spans):
                return EIGHT_ROW_REASON
            # QWEN_FAST_OCTO_GLUE8: the quarter-tile mover serves the octo block's eight-row users.
            import gdn_rows_dma8_tp

            if len(shape) != 3 or shape[0] != 1 or shape[1] != users * OCTO_PIECE_ROWS:
                return 'block projection %r is not (1, %d, W)' % (shape, users * OCTO_PIECE_ROWS)
            return gdn_rows_dma8_tp.problem([projected], operations)
        if users < 2 or [tuple(span) for span in spans] != [(user * PIECE_ROWS, (user + 1) * PIECE_ROWS) for user in range(users)]:
            return 'spans %r are not contiguous 16-row users from row 0' % (list(spans),)
        if len(shape) != 3 or shape[0] != 1 or shape[1] != users * PIECE_ROWS:
            return 'block projection %r is not (1, %d, W)' % (shape, users * PIECE_ROWS)
        reason = rows_dma.problem([projected], operations)
        return reason

    def eight_native(self, spans):
        """Whether these spans are the octo block's and QWEN_FAST_OCTO_GLUE8 makes its glue sites native (False, without importing anything, for every other block and for the flag off)."""
        if len(spans) != OCTO_USERS or not glue8_requested():
            return False
        import octo_glue8

        return octo_glue8.is_octo_spans(spans) and octo_glue8.enabled()

    def eight_native_users(self, users):
        """The same question over a packed block's (piece, ...) users: eight pieces of eight rows."""
        if len(users) != OCTO_USERS or not glue8_requested():
            return False
        import octo_glue8

        return all(len(user[0].shape) == 3 and tuple(user[0].shape)[1] == OCTO_PIECE_ROWS for user in users) and octo_glue8.enabled()

    def note_glue_fallback(self, site, reason):
        if reason == EIGHT_ROW_REASON:
            # The octo block's known state, not a lever that saved nothing by accident: its own marker, its own counter.
            line = '%s site=%s reason=%s' % (tp4_vglue.EIGHT_ROW_SERVED, site, reason)
            if line not in self.glue_fallbacks:
                self.glue_fallbacks.add(line)
                tp4_vglue.log_line(line)
            tp4_vglue.note('gdn_%s_eight_row_served' % site)
            return
        line = '%s site=%s reason=%s' % (tp4_vglue.FALLBACK, site, reason)
        if line not in self.glue_fallbacks:
            self.glue_fallbacks.add(line)
            tp4_vglue.log_line(line)
        tp4_vglue.note('gdn_%s_fallback' % site)

    glue_fallbacks = set()

    def split_pieces(self, projected, spans, owned):
        """The per-user pieces of the block projection from one launch: (1, 16, W) TILE, L1 interleaved (the placement
        resident_piece leaves every piece in), rows 0-15 the user's rows (canonical for users 1 and 3, as the served round trip
        leaves them) and rows 16-31 zero. Owned exactly as the served pieces are."""
        operations, mesh = self.operations, self.gdn.mesh
        width = projected.shape[-1]
        pieces = []
        eight = self.eight_native(spans)
        mover, piece_rows = (rows_dma, PIECE_ROWS) if not eight else (self.eight_mover(), OCTO_PIECE_ROWS)
        try:
            for user in range(len(spans)):
                pieces.append(operations.empty((1, piece_rows, width), dtype=operations.bfloat16,
                                               layout=operations.TILE_LAYOUT, device=mesh,
                                               memory_config=operations.L1_MEMORY_CONFIG))
            mover.launch(mesh, [projected], pieces, mover.split_pieces(len(spans), width))
        except BaseException:
            for piece in pieces:
                operations.deallocate(piece)
            raise
        owned.extend(pieces)
        return pieces

    def eight_mover(self):
        import gdn_rows_dma8_tp

        return gdn_rows_dma8_tp

    def served_pieces(self, projected, spans, owned):
        """The pinned path's pieces (a Slice each, resident in L1): what the audit compares the launch's with."""
        operations = self.operations
        return [pinned.resident_piece(operations,
                                      operations.slice(projected, (0, start, 0), (1, stop, projected.shape[-1])), owned)
                for start, stop in spans]

    def _decode_packed(self, packed, rows, spans, checkpoints, prefixes, slots, defer_publication, deferred):
        call = gdn_pair_slice_tp.wrap(self.operations, tp4_vglue.enabled(tp4_vglue.GDN_GLUE))
        if call is None:
            return super()._decode_packed(packed, rows, spans, checkpoints, prefixes, slots, defer_publication, deferred)
        real = self.operations
        self.operations = call.operations
        try:
            result = super()._decode_packed(packed, rows, spans, checkpoints, prefixes, slots, defer_publication, deferred)
        except BaseException:
            call.abort()
            raise
        finally:
            self.operations = real
            call.close()
        call.finish(result)
        return result

    def _recurrence_user_batched(self, projected, spans, slots, entries, owned):
        if not tp4_vglue.enabled(tp4_vglue.GDN_GLUE):
            return super()._recurrence_user_batched(projected, spans, slots, entries, owned)
        reason = self.glue_problem(projected, spans)
        if reason is not None:
            if reason != EIGHT_ROW_REASON and self.eight_native(spans):
                import octo_glue8

                octo_glue8.note_fallback('split', reason)
            else:
                self.note_glue_fallback('split', reason)
            return super()._recurrence_user_batched(projected, spans, slots, entries, owned)
        try:
            return self._recurrence_glued(projected, spans, slots, entries, owned)
        except rows_dma.Unsupported as error:
            # Raised before any launch of ours ran and before any state move, so the pinned path starts clean.
            if self.eight_native(spans):
                import octo_glue8

                octo_glue8.note_fallback('split', str(error))
            else:
                self.note_glue_fallback('split', str(error))
            return super()._recurrence_user_batched(projected, spans, slots, entries, owned)

    def _recurrence_glued(self, projected, spans, slots, entries, owned):
        """pinned._recurrence_user_batched with the per-user slices replaced by one split launch. The state moves, their
        order, the T1 notes and the launch are the pinned body's, statement for statement."""
        operations, layer = self.operations, self.gdn
        last = len(spans) - 1
        pieces = self.split_pieces(projected, spans, owned)
        audit = None
        if tp4_vglue.audit_enabled():
            audit = self.hold_served_pieces(projected, spans, pieces)
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
            pending.append((pieces[index], source[0], source[1:]))
        if batched:
            copy_compact_batch(batched)
        if direct:
            verify_t1_note('direct_carry')
        if direct_last:
            verify_t1_note('last_carry')
        results = self.run_users(projected, pieces, pending)
        if len(results) != len(spans):
            raise AssertionError('Batched GDN returned a result per packed user')
        for result, (start, stop) in zip(results, spans):
            if not result.get('user_batched', False) or not result.get('deferred_conv_publication', False):
                raise AssertionError('User-batched GDN did not engage as selected')
            if result.get('norm_batch', True) or result['states'].shape[0] != stop - start:
                raise AssertionError('User-batched GDN did not return this user own prefix geometry')
        tp4_vglue.note('gdn_glue')
        if len(spans) == OCTO_USERS and self.eight_native(spans):
            import octo_glue8

            octo_glue8.note_engaged('split')
        if audit:
            for result, entries_ in zip(results, audit):
                result['vglue_audit'] = entries_
        return results

    def run_users(self, projected, pieces, pending):
        """The per-user convolution and gates, then the one recurrence launch: the pinned call, plus (QWEN_FAST_TP4_GDN_BLOCK_CONV)
        the block stage that makes the four users' conv, beta, g and z with one conv_gates launch."""
        layer, operations = self.gdn, self.operations
        taps = list(layer.tw['conv_taps'])
        options, staged = {}, []
        if tp4_vglue.enabled(tp4_vglue.GDN_BLOCK_CONV):
            if verify_trace_t2.cut('windows'):
                eight = self.eight_native_users(pending)
                if eight:
                    # QWEN_FAST_OCTO_GLUE8: V1 over the quarter-tile mover (gdn_block_conv8_tp); a decline is the glue8 FALLBACK line.
                    import gdn_block_conv8_tp
                    import octo_glue8

                    stage_module = gdn_block_conv8_tp
                    decline = lambda reason: octo_glue8.note_fallback('block_conv', reason)
                else:
                    stage_module = gdn_block_conv_tp
                    decline = lambda reason: self.note_glue_fallback('block_conv', reason)

                def block_stage(groups, windows):
                    found = stage_module.stage(
                        layer.mesh, projected, groups, windows, taps, layer.tw['dt_bias'], layer.tw['neg_exp_A'], operations,
                        note_fallback=decline)
                    if found is not None:
                        staged.append(found)
                        tp4_vglue.note('gdn_block_conv')
                        if eight:
                            octo_glue8.note_engaged('block_conv')
                    return found

                options['block_stage'] = block_stage
            else:
                self.note_glue_fallback('block_conv', 'the T2 packed windows cut is off (QWEN_FAST_VERIFY_T2)')
        results = run_user_batched_projected(layer.mesh, pending, taps,
            layer.tw['dt_bias'], layer.tw['neg_exp_A'], layer.tw['norm_w'], self.kernels, operations,
            prefix_zero_reuse=self.prefix_zero_reuse, **options)
        if staged and staged[0].entries:
            for result, entries in zip(results, staged[0].entries):
                result['vglue_block_audit'] = entries
        return results

    def hold_served_pieces(self, projected, spans, pieces):
        """QWEN_FAST_TP4_VGLUE_AUDIT: DRAM copies of each launch piece and of the served piece (a Slice, resident, as
        served), per user, held outside `owned` (freed with the retained record). The pieces themselves are freed inside the
        trace once the layer is done, so what is compared after the replay is the copy."""
        operations = self.operations
        held, scratch, entries = [], [], []
        try:
            served = self.served_pieces(projected, spans, scratch)
            for user, (mine, theirs) in enumerate(zip(pieces, served)):
                copies = [operations.clone(value, memory_config=operations.DRAM_MEMORY_CONFIG) for value in (mine, theirs)]
                held.extend(copies)
                entries.append([dict(label='piece user %d' % user, mine=copies[0], served=copies[1])])
        except BaseException:
            release_owned(operations, held)
            raise
        finally:
            release_owned(operations, scratch)
        return entries

    def _finish_packed(self, spans, results, outputs, owned):
        if not tp4_vglue.enabled(tp4_vglue.GDN_GLUE) or len(outputs) < 2:
            return super()._finish_packed(spans, results, outputs, owned)
        reason = self.merge_problem(outputs)
        eight = reason != EIGHT_ROW_REASON and self.eight_outputs(outputs)
        if reason is not None:
            if eight:
                import octo_glue8

                octo_glue8.note_fallback('merge', reason)
            else:
                self.note_glue_fallback('merge', reason)
            return super()._finish_packed(spans, results, outputs, owned)
        operations = self.operations
        try:
            merged = self.merge_outputs(outputs)
        except rows_dma.Unsupported as error:
            if eight:
                import octo_glue8

                octo_glue8.note_fallback('merge', str(error))
            else:
                self.note_glue_fallback('merge', str(error))
            return super()._finish_packed(spans, results, outputs, owned)
        owned.append(merged)
        audit = self.hold_served_merge(outputs, merged) if tp4_vglue.audit_enabled() else None
        combined = super()._finish_packed(spans, results, [merged], owned)
        if audit:
            combined['vglue_merge_audit'] = audit
        tp4_vglue.note('gdn_merge')
        if eight:
            import octo_glue8

            octo_glue8.note_engaged('merge')
        return combined

    def eight_outputs(self, outputs):
        """Whether these outputs are the octo block's eight (1, 8, N) ones and QWEN_FAST_OCTO_GLUE8 makes their join native."""
        if len(outputs) != OCTO_USERS or not glue8_requested():
            return False
        import octo_glue8

        shapes = {tuple(output.shape) for output in outputs}
        return len(shapes) == 1 and next(iter(shapes))[:2] == (1, OCTO_PIECE_ROWS) and octo_glue8.enabled()

    def merge_problem(self, outputs):
        shapes = {tuple(output.shape) for output in outputs}
        if len(shapes) == 1 and next(iter(shapes))[:2] == (1, OCTO_PIECE_ROWS) and len(outputs) == OCTO_USERS:
            if self.eight_outputs(outputs):
                return self.eight_mover().problem(outputs, self.operations)
            return EIGHT_ROW_REASON
        if len(shapes) != 1 or next(iter(shapes))[:2] != (1, PIECE_ROWS):
            return 'outputs %r are not equal (1, 16, N) tensors' % (sorted(shapes),)
        return rows_dma.problem(outputs, self.operations)

    def merge_outputs(self, outputs):
        """The (1, 16 * users, N) block from one launch, in interleaved DRAM (where the served concat with no memory_config
        leaves its result: measured on the card, the users' outputs are L1 and the join is not), every user canonical as the
        served untilize / concat / tilize leaves it. The output projection compiles per input placement, so the placement is
        part of the contract."""
        operations = self.operations
        width = outputs[0].shape[-1]
        mover, piece_rows = (self.eight_mover(), OCTO_PIECE_ROWS) if self.eight_outputs(outputs) else (rows_dma, PIECE_ROWS)
        merged = operations.empty((1, piece_rows * len(outputs), width), dtype=operations.bfloat16,
                                  layout=operations.TILE_LAYOUT, device=self.gdn.mesh,
                                  memory_config=operations.DRAM_MEMORY_CONFIG)
        try:
            mover.launch(self.gdn.mesh, list(outputs), [merged], mover.merge_outputs(len(outputs), width))
        except BaseException:
            operations.deallocate(merged)
            raise
        return merged

    def hold_served_merge(self, outputs, merged):
        """The served join (concat) of the same outputs and a copy of the launch's block, both in DRAM and held outside
        `owned`: the pair the audit compares, with the two placements the layer would have seen (the launch's block is what the
        layer consumes and is freed after the output projection, so what is compared after the replay is the copy; the served
        concat is freed at once so the audit holds no extra memory across the trace)."""
        operations = self.operations
        dram = operations.DRAM_MEMORY_CONFIG
        served = operations.concat(list(outputs), dim=1)
        held = []
        try:
            placement = (merged.memory_config(), served.memory_config())
            held.append(operations.clone(merged, memory_config=dram))
            held.append(operations.clone(served, memory_config=dram))
        except BaseException:
            release_owned(operations, held)
            raise
        finally:
            operations.deallocate(served)
        return [dict(label='merged output', mine=held[0], served=held[1], placement=placement)]
