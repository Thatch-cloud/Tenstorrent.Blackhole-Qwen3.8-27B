"""extent_attention_replay_tp.PackedExtentReplayReader with the block's attention fold as two launches (tp4/vglue V3a).

extent_attention_replay_tp.py is held unedited by the S2 four-card evidence (packed_any_evidence_tp4.json pins its sha256: CB2b
qualified those bytes, and packed_any_admission refuses the traffic profile on any other), so the V3a hook cannot live in it.
This twin subclasses the reader and overrides `__call__` only; tp_addresses.install() binds it in place of the reader at
QWEN_FAST_TP=4 only (model_batch reaches the class through the module alias, which is the four-card twin module).

QWEN_FAST_TP4_ATTN_FOLD unset or 0: `__call__` is the pinned reader's, called through `super()` with the same arguments (nothing
here runs). QWEN_FAST_TP4_SDPA=multi (sdpa_multi_tp) replaces the per-user launches with one launch for the block (and
QWEN_FAST_TP4_SDPA_AUDIT=1 runs both and compares them); `multi` is None otherwise and every method here is the one without it. QWEN_FAST_TP4_SDPA (sdpa_long_tp) is read once, in `__init__`, after the pinned constructor has built and qualified the readers.
QWEN_FAST_TP4_ATTN_FOLD=1: the per-segment query slices, fold DMAs, stacking concats, result slices, inverse folds and
concats become attention_block_fold_tp's two launches; a query or a placement the launches are not written for takes the pinned
path, logged as FALLBACK and counted (nothing is guessed).
"""

from contextlib import contextmanager

import attention_block_fold_tp
import extent_attention_replay_tp as base_module
import sdpa_long_tp
import tp4_vglue
from tp_addresses import addresses, release_owned


def _pinned_class():
    """The reader class of extent_attention_replay_tp, whichever way this module was imported (tp_addresses.install rebinds that
    module's own name to this class, so a module first imported after install must not take itself for its base)."""
    base = base_module.PackedExtentReplayReader
    if getattr(base, '__module__', None) == __name__:
        base = base.__mro__[1]
    return base


class PackedExtentReplayReader(_pinned_class()):
    # QWEN_FAST_TP4_SDPA=multi (sdpa_multi_tp.MultiBlock, set by sdpa_long_tp.apply in the constructor): None unless that
    # configuration is on, and then every path below that reads it is the pinned reader's.
    multi = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # QWEN_FAST_TP4_SDPA (sdpa_long_tp): flag unset or off, this returns at once and nothing is touched.
        try:
            sdpa_long_tp.apply(self)
        except BaseException:
            self.close()
            raise

    def __call__(self, query, keys, values, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        multi = self.multi
        if multi is not None:
            # QWEN_FAST_TP4_SDPA=multi: ONE launch for every user of the block. Under QWEN_FAST_TP4_SDPA_AUDIT the served per-user
            # launches run beside it (served_call, this method's own path with the flag off) and the two block outputs are
            # compared bit for bit in the trace; the multi output is the one returned.
            served = None
            if multi.audit:
                def served():
                    return self.served_call(query, keys, values, page_table_tensor=page_table_tensor,
                                            cur_pos_tensor=cur_pos_tensor, **kwargs)
            return multi.call(query, keys, values, scale=kwargs['scale'], memory_config=kwargs['memory_config'], served=served)
        return self.served_call(query, keys, values, page_table_tensor=page_table_tensor, cur_pos_tensor=cur_pos_tensor, **kwargs)

    def served_call(self, query, keys, values, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        """The served __call__ (V3a's two launches under QWEN_FAST_TP4_ATTN_FOLD, else the pinned reader's), unchanged."""
        if tp4_vglue.enabled(tp4_vglue.ATTN_FOLD):
            self.check_open()
            if tuple(query.shape) != (1, self.rows, base_module.head_rows(), 256):
                raise ValueError('Packed extent query geometry changed')
            chunks = self.fold_chunks()
            reason = attention_block_fold_tp.problem(query, chunks, kwargs['memory_config'], self.operations)
            if reason is None:
                return self.call_block_folded(query, keys, values, chunks, page_table_tensor=page_table_tensor,
                                              cur_pos_tensor=cur_pos_tensor, **kwargs)
            self.note_fold_fallback(reason)
        return super().__call__(query, keys, values, page_table_tensor=page_table_tensor, cur_pos_tensor=cur_pos_tensor, **kwargs)

    def shared_masks(self, expected_calls):
        """Flag off (multi None): the pinned reader's context manager itself. Multi: the multi launch's own scope (which refreshes its
        stacked table, cur_pos and mask once per forward) and, under the audit, the pinned scope around it, so the served launches
        beside it find their per-user masks refreshed and their call budget booked."""
        if self.multi is None:
            return super().shared_masks(expected_calls)
        return self.multi_masks(expected_calls)

    @contextmanager
    def multi_masks(self, expected_calls):
        multi = self.multi
        if multi.audit:
            with super().shared_masks(expected_calls):
                with multi.scope(expected_calls):
                    yield
        else:
            with multi.scope(expected_calls):
                yield

    def rebound_reason(self):
        """None, or why the multi launch must not run: a segment now holds other positions, table or cur_pos buffers than the launch's programs
        were built on. packed_verifier.verify asks on the host before EVERY replay (a replay re-issues the device program and never comes back to
        Python, so this is the only place a rebinding after the attach can be seen outside the scope entry; the timed arms run no SDPA audit).
        A few identity comparisons. None when the flag is off."""
        multi = self.multi
        return None if multi is None else multi.rebound()

    def sdpa_audit_round(self, round_number):
        """QWEN_FAST_TP4_SDPA_AUDIT: after a replay, the multi launch's counters (packed_verifier calls this beside the other audits).
        0 and no effect unless the multi audit is on."""
        multi = self.multi
        if multi is None or not multi.audit:
            return 0
        return multi.audit_round(round_number)

    def close(self):
        multi = self.multi
        if multi is not None:
            multi.close()
        super().close()

    def fold_chunks(self):
        """QWEN_FAST_TP4_ATTN_FOLD: the block's SDPA bundles in dispatch order (attention_block_fold_tp.Chunk)."""
        return attention_block_fold_tp.chunks_of(
            self.segments, [[bundle for bundle, pages, mask, config in reader.metadata] for reader in self.readers])

    fold_fallbacks = set()

    def note_fold_fallback(self, reason):
        if reason not in self.fold_fallbacks:
            self.fold_fallbacks.add(reason)
            tp4_vglue.log_line('%s site=attention reason=%s' % (tp4_vglue.FALLBACK, reason))
        tp4_vglue.note('attn_fold_fallback')

    def call_block_folded(self, query, keys, values, chunks, *, page_table_tensor=None, cur_pos_tensor=None, **kwargs):
        """__call__ with the per-segment query slices, fold DMAs, stacking concats, result slices, inverse folds and
        concats replaced by attention_block_fold_tp's two launches (QWEN_FAST_TP4_ATTN_FOLD). Each segment reader keeps
        its bookkeeping exactly as ExtentSegmentReader.__call__ does it (validate, mask refresh or shared-mask budget,
        calls, failed), and every SDPA call is the served one: the same stacked query bytes, keys, values, page table,
        cur_pos word, mask, scale, program config and output memory config."""
        operations = self.operations
        scale, memory_config = kwargs['scale'], kwargs['memory_config']
        owned, results = [], []
        protected = {addresses(operations, value) for value in (query, keys, values)}
        try:
            for reader in self.readers:
                reader.validate(reader.start)
                if reader.mask_scope is None:
                    reader.refresh()
                elif reader.calls >= reader.mask_scope:
                    raise AssertionError('Shared-mask forward exceeded its attention call budget')
            stacked = attention_block_fold_tp.fold_in(self.mesh, query, chunks, owned)
            entries = [(entry, positions) for reader in self.readers
                       for entry, positions in zip(reader.metadata, reader.cur_pos, strict=True)]
            for stack, ((bundle, pages, mask, config), positions) in zip(stacked, entries, strict=True):
                result = operations.transformer.paged_scaled_dot_product_attention_decode(stack, keys, values,
                    page_table_tensor=pages, cur_pos_tensor=positions, is_causal=False, attn_mask=mask, scale=scale,
                    program_config=config, memory_config=memory_config)
                owned.append(result)
                results.append(result)
            output = attention_block_fold_tp.fold_out(self.mesh, results, chunks, memory_config, owned)
            protected.add(addresses(operations, output))
            for reader in self.readers:
                reader.calls += 1
            self.calls += 1
            tp4_vglue.note('attn_fold')
            return output
        except BaseException:
            for reader in self.readers:
                reader.failed = True
            raise
        finally:
            release_owned(operations, [value for value in owned if addresses(operations, value) not in protected])
