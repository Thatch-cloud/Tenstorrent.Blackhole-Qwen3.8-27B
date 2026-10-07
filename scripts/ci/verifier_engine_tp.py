"""verifier_engine.VerifierEngine (the sequential engine) at any served width.

verifier_engine.py is one of serving_bundle.package's eight critical staged-source files (test_quad_draft holds it
untouched: the frozen bundle's inventory sha is checked against the checkout's bytes), and its sequential verify readback
requires exactly two chip-local outputs. So the four-card engine is a subclass with that one method, verify, restated with the
chip count from tp_shapes (the pair's message is unchanged: 'Two chip-local outputs required'), and proposal_rows, which
QWEN_FAST_BUDGET_CAP widens (below); every other method is inherited. publish is inherited too, with QWEN_FAST_SEQ_STAGE_LOG's begin and end lines around it.
serving_request_factory.device_components builds this class at four cards and the pair's at the pair.

ENGINE REUSE (QWEN_FAST_PARKED_ENGINES, serving_parked_engines; default off): the parked phase of the engine lives HERE, not in the
pair's file. park_refusal, park, rebind_refusal and rebind are Stage E's (the TP2 line's verifier_engine.py carries them), restated for this
class; the module functions set_replay_count, check_replay_mark and reset_retained are Stage E's too. close() releases a parked engine's
carry traces (D2: the inherited close takes only idle, preparing and failed engines' carry traces), verify() notes R2's replay mark (D1:
this class restates the whole verify, so the pair's mark is never set), publish() checks it, and proposal_rows() is restricted to the
widths a cold engine of the same request would capture (R4). Without the flag nothing here runs: an engine is never parked, request_widths
stays None and every answer is what it was.
"""

import os
import time
from contextlib import ExitStack

from attention_batch import capture_operation
from force_argmax import sample_rows
from gdn_multitoken_conv import release_owned
from serving_fast_request import budget_cap_enabled
import trace_census
import verify_trace_t1
import verifier_engine as pair_module
from verifier_engine import VERIFY_WIDTHS, VerifierEngine as PairVerifierEngine, capture_widths
from verifier_inputs import stage_inputs
import tp_shapes

# QWEN_FAST_REQUEST_SHARD_ARGMAX=1 (default off, read once per engine at construction): the request engine's rows=1/2/4 verify traces pick
# tokens with the packed block's per-chip shard argmax (verify_trace_t1.sample_shards / combine_shards) instead of the pinned sampler, so they
# bake no sampler AllGather, untilize or ArgMax. Engaged only where verify_trace_t1.shard_sampling_problem(sampler) is None and the logits
# are the bf16 TILE vocab shard; otherwise the pinned sampler runs as today and the reason is logged once.
# QWEN_FAST_REQUEST_SHARD_AUDIT=1 (beside the arm): also run the pinned sampler in the trace, compare every row with the combined ids, log
# a mismatch, and serve the pinned sampler's ids.
SHARD_FLAG = 'QWEN_FAST_REQUEST_SHARD_ARGMAX'
SHARD_AUDIT_FLAG = 'QWEN_FAST_REQUEST_SHARD_AUDIT'
ENGAGED_MARKER = '[PINDIAG] request shard argmax engaged'
FALLBACK_MARKER = '[PINDIAG] request shard argmax kept the pinned sampler'
AUDIT_MARKER = '[PINDIAG] request shard argmax audit'
AUDIT_MISMATCH = '[PINDIAG] request shard argmax audit mismatch'
_LOGGED = set()

# QWEN_FAST_TP4_TRACED_PUBLISH=1 (default off, read once per engine at construction): the sequential step's per-step GDN carry save
# (and the restore a resident-engine switch pays) replay one captured trace each instead of 48 eager launches (16.4 ms median host
# enqueue a lone step, run N4b). The traces are captured once, inside the engine's own build, right after its verify and commit traces:
# the carry slots are pool storage allocated at attach (allocate_carry, before any capture), the copy kernels run eagerly once before
# the capture (the save is the build's own seed; the restore is an identity warm, carry == slot zero by construction), and the captured
# programs allocate no persistent buffer, so nothing is created after a capture that a replay could overwrite. An engine whose carry or
# helpers do not fit (not the direct DMA copy) declines with a logged reason and stays eager. The bytes moved, their addresses and
# their order are the eager loop's.
# QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT=1 (beside the arm): after every traced copy the eager loop runs too, and the destination is read
# back before and after on every chip and compared bit for bit ('[TPUB-AUDIT]'); a mismatch is logged and the eager bytes stand.
TRACED_FLAG = 'QWEN_FAST_TP4_TRACED_PUBLISH'
TRACED_AUDIT_FLAG = 'QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT'
TRACED_ENGAGED = '[TPUB] carry traces engaged'
TRACED_DECLINED = '[TPUB] carry traces declined'
TRACED_AUDIT = '[TPUB-AUDIT]'


def traced_publish_enabled(environ=None):
    return (os.environ if environ is None else environ).get(TRACED_FLAG) == '1'


def traced_audit_enabled(environ=None):
    return traced_publish_enabled(environ) and (os.environ if environ is None else environ).get(TRACED_AUDIT_FLAG) == '1'


def carry_trace_problem(helpers, carry):
    """Why the carry copies cannot be traced, or None: every helper must be the direct DMA copy (the ttnn slice path allocates
    inside the copy) and every layer must hold a complete carry slot."""
    if not helpers or len(carry) != len(helpers):
        return 'carry holds %d layers for %d helpers' % (len(carry), len(helpers))
    for index, (helper, slot) in enumerate(zip(helpers, carry)):
        if getattr(helper, 'direct', False) is not True:
            return 'layer %d helper is not the direct copy' % index
        if len(slot) != len(helper.live):
            return 'layer %d carry holds %d tensors for %d live' % (index, len(slot), len(helper.live))
    return None


def _bits_equal(left, right):
    import torch

    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(left.element_size())
    if width is None:
        return bool(torch.equal(left, right))
    return bool(torch.equal(left.contiguous().view(width), right.contiguous().view(width)))


def carry_log_line(message, **values):
    from verifier_engine import carry_log

    carry_log(message, **values)


def shard_arm_enabled(environ=None):
    return (os.environ if environ is None else environ).get(SHARD_FLAG) == '1'


def shard_audit_enabled(environ=None):
    return shard_arm_enabled(environ) and (os.environ if environ is None else environ).get(SHARD_AUDIT_FLAG) == '1'


def _log_once(key, message):
    if key not in _LOGGED:
        _LOGGED.add(key)
        verify_trace_t1.log_line(message)


def logits_problem(operations, logits, rows):
    """Why sample_shards cannot take these logits, or None: the pre-gather (1, 1, rows, vocab shard) bf16 TILE tensor, and no
    QWEN_FAST_TP4_VGLUE_AUDIT beside the gathered shard maxima (its ttnn.max reference is consumed only by the packed block)."""
    shape = tuple(logits.shape)
    if len(shape) != 4 or shape[:3] != (1, 1, rows) or shape[3] != tp_shapes.vocab_shard():
        return 'logits shape %r is not the (1, 1, %d, %d) vocab shard' % (shape, rows, tp_shapes.vocab_shard())
    dtype, layout = getattr(logits, 'dtype', None), getattr(logits, 'layout', None)
    if dtype != operations.bfloat16 or layout != operations.TILE_LAYOUT:
        return 'logits are %r %r, not bf16 TILE' % (dtype, layout)
    import tp4_vglue

    if tp4_vglue.audit_enabled() and tp4_vglue.enabled(tp4_vglue.SHARD_VALUES):
        return 'QWEN_FAST_TP4_VGLUE_AUDIT audits the shard maxima in the packed block only'
    return None


# Engine reuse: R2's replay ledger (QWEN_FAST_PARKED_AUDIT=1, serving_parked_engines.ReplayLedger): a zero-argument callable giving the
# process's trace-replay count, or None. With it, verify notes the count right after its own replay and publish asserts that no other
# trace replayed before its commit trace reads what that verify wrote (check_replay_mark). None - the default, and the only value without
# the audit - runs neither check. The ledger installs from serving_runtime whenever the audit is on, with or without parked engines, so
# the flag-off control arm of the audit twin measures the same ordering (design K2).
_replay_count = None


def set_replay_count(counter):
    """Install (a zero-argument callable) or remove (None) the replay ledger; returns the previous one."""
    global _replay_count
    if counter is not None and not callable(counter):
        raise ValueError('The replay ledger must be a zero-argument callable or None')
    previous, _replay_count = _replay_count, counter
    return previous


def check_replay_mark(engine):
    """R2: no trace replayed between this engine's verify and now. Clears the mark."""
    mark, engine.replay_mark = getattr(engine, 'replay_mark', None), None
    if mark is None or _replay_count is None:
        return
    count = _replay_count()
    if count != mark:
        raise AssertionError('R2: %d other trace replay(s) ran between the verify of request %s and its publication'
                             % (count - mark, str(engine.session.request_id)[:48]))


def reset_retained(retained):
    """A retained GDN block's decision flags back to the values a fresh engine's hold after its capture (gdn_records.
    RetainedGDNBlock.__init__; the capture appends records and decides nothing), so the rebound engine's first verify and commit take
    a fresh engine's path. The records stay: they are the trace's. replay_epoch is a counter and stays."""
    retained.selected_prefix = None
    retained.decisions = {}
    retained.replay_ready = retained.poisoned = retained.fence_owed = False
    retained.commit_serial = 0
    retained.replay_fence, retained.replay_fence_ms = None, 0.0


class VerifierEngine(PairVerifierEngine):
    def __init__(self, *args, sampler=None, **options):
        # read before the base constructor: it captures every width's trace (operation) from inside __init__
        self.request_shard = self.request_shard_audit = False
        self.request_shard_problem = None
        if shard_arm_enabled():
            import tp4_sampdraft

            if tp4_sampdraft.enabled(tp4_sampdraft.SHARD_ARGMAX):
                # S1 belongs to the packed block: it reserves the scratch before capture and its audit reads and frees the held
                # reference every round. This engine runs neither, so the combination is refused rather than half-engaged.
                raise ValueError('%s is a lever of the packed block sampler; it does not run in the request engine, whose own shard arm '
                                 '(%s) is on: unset one of them' % (tp4_sampdraft.SHARD_ARGMAX, SHARD_FLAG))
            self.request_shard_problem = (verify_trace_t1.shard_sampling_problem(sampler) if sampler is not None
                                          else 'there is no device sampler')
            self.request_shard = self.request_shard_problem is None
            self.request_shard_audit = self.request_shard and shard_audit_enabled()
            if self.request_shard_problem is not None:
                _log_once(('sampler', self.request_shard_problem), '%s: %s' % (FALLBACK_MARKER, self.request_shard_problem))
        # read before the base constructor too: its last step is save_carry, where the traces are captured
        self.carry_trace_arm = traced_publish_enabled()
        self.carry_traces = None if self.carry_trace_arm else False
        self.carry_trace_audit = traced_audit_enabled()
        self.carry_audit_checked = self.carry_audit_mismatches = 0
        # Engine reuse (R4): the widths a COLD engine of the request this engine is rebound to would have captured. None - every engine
        # that was built for its request - is `widths`. Read by proposal_rows, serves and bucket_key.
        self.request_widths = None
        self.replay_mark = None
        super().__init__(*args, sampler=sampler, **options)
        # What the captures hold. `widths` is what the request may ask: the same, until a rebind narrows it to a cold engine's (R4).
        self.captured_widths = tuple(getattr(self, 'widths', None) or ())

    def operation(self, fixture, *, hidden_capture=None, feature_capture=None):
        """The inherited (logits, pinned ids); under the arm (logits, shard ids, shard maxima) or, audited, (logits, shard ids, shard
        maxima, pinned ids). verify reads the tuple's length."""
        if not self.request_shard:
            return super().operation(fixture, hidden_capture=hidden_capture, feature_capture=feature_capture)
        logits = None
        try:
            with ExitStack() as captures:
                for capture in (hidden_capture, feature_capture):
                    if capture is not None:
                        captures.enter_context(capture.capture())
                logits = fixture.run(sharded_logits=True)
            problem = logits_problem(self.operations, logits, fixture.rows)
            if problem is not None:
                _log_once(('logits', problem), '%s: %s' % (FALLBACK_MARKER, problem))
                return logits, sample_rows(self.sampler, logits, fixture.rows, self.operations, native_rows=self.native_sampling_rows)
            ids, values = verify_trace_t1.sample_shards(self.operations, logits, fixture.rows)
            _log_once(('engaged', fixture.rows, self.request_shard_audit), '%s rows=%d audit=%d' % (
                ENGAGED_MARKER, fixture.rows, int(self.request_shard_audit)))
            if not self.request_shard_audit:
                return logits, ids, values
            try:
                reference = sample_rows(self.sampler, logits, fixture.rows, self.operations, native_rows=self.native_sampling_rows)
            except BaseException:
                release_owned(self.operations, [ids, values])
                raise
            return logits, ids, values, reference
        except BaseException:
            if logits is not None:
                self.operations.deallocate(logits)
            raise

    def shard_predictions(self, output, rows):
        """The ticket's ids from a shard-argmax output: each chip's (id, max) read back and folded on the host exactly as
        packed_verifier.PackedVerifierEngine.shard_predictions does (verify_trace_t1.combine_shards). Audited, every row is compared
        with the pinned sampler's id from the same replay, a mismatch is logged, and the pinned sampler's ids are what is served."""
        chips = tp_shapes.chip_count()
        id_parts = self.operations.get_device_tensors(output[1])
        value_parts = self.operations.get_device_tensors(output[2])
        if len(id_parts) != chips or len(value_parts) != chips:
            raise AssertionError('%s chip-local outputs required' % tp_shapes.count_word())
        chip_ids = [self.operations.to_torch(part).reshape(-1)[:rows] for part in id_parts]
        chip_values = [self.operations.to_torch(part).reshape(-1)[:rows] for part in value_parts]
        if any(len(value) != rows for value in (*chip_ids, *chip_values)):
            raise AssertionError('Missing target prediction rows')
        if len(output) < 4:
            return verify_trace_t1.combine_shards(chip_ids, chip_values).tolist()
        pinned = self.operations.to_torch(self.operations.get_device_tensors(output[3])[0]).reshape(-1)[:rows].tolist()
        try:
            combined = verify_trace_t1.combine_shards(chip_ids, chip_values).tolist()
        except Exception as error:
            # audited, the pinned ids are what is served: a fold that cannot run (a shard id out of range) is a finding, not a failed request
            verify_trace_t1.log_line('%s rows=%d combine failed: %r' % (AUDIT_MISMATCH, rows, error))
            return pinned
        differing = [row for row, (mine, kept) in enumerate(zip(combined, pinned)) if mine != kept]
        if differing:
            verify_trace_t1.log_line('%s rows=%d differing=%s shard=%s sampler=%s' % (
                AUDIT_MISMATCH, rows, differing[:8], [combined[row] for row in differing[:8]], [pinned[row] for row in differing[:8]]))
        else:
            _log_once(('audit', rows), '%s exact=True rows=%d' % (AUDIT_MARKER, rows))
        return pinned

    def copy_carry(self, operation, source):
        """The inherited eager copies, unless QWEN_FAST_TP4_TRACED_PUBLISH has captured this engine's carry traces: then one replay.
        The first save, which the base constructor makes last, runs eagerly and captures them (only while the engine is still being
        built; an engine past its build never captures)."""
        traces = getattr(self, 'carry_traces', False)
        if traces is False:
            return super().copy_carry(operation, source)
        if traces is None:
            super().copy_carry(operation, source)
            if operation == 'save' and getattr(self, 'phase', None) == 'preparing':
                self.capture_carry_traces()
            return
        if self.slot_addresses() != self.carry_addresses:
            raise ValueError('Carried GDN state moved under the engine')
        logging = os.environ.get('QWEN_FAST_CARRY_LOG') == '1'
        request, layers = str(self.session.request_id)[:48], len(self.carry)
        if logging:
            carry_log_line('[CARRY] op={op} request={request}{origin} layers={layers} begin traced=1', op=operation,
                request=request, origin='' if source is None else ' from=' + source, layers=layers)
        started = time.perf_counter()
        if self.carry_trace_audit:
            self.audited_carry_copy(operation)
        else:
            self.operations.execute_trace(self.mesh, traces[operation], cq_id=0, blocking=False)
        enqueued = time.perf_counter()
        if logging:
            self.operations.synchronize_device(self.mesh)
            carry_log_line('[CARRY] op={op} request={request} layers={layers} enqueue_ms={enqueue:.1f} fence_ms={fence:.1f} traced=1',
                op=operation, request=request, layers=layers, enqueue=(enqueued - started) * 1000,
                fence=(time.perf_counter() - enqueued) * 1000)

    def eager_carry_copy(self, operation):
        for helper, slot in zip(self.helpers, self.carry, strict=True):
            getattr(helper, operation)(slot)

    def capture_carry_traces(self):
        problem = carry_trace_problem(self.helpers, self.carry)
        request = str(self.session.request_id)[:48]
        if problem is not None:
            self.carry_traces = False
            verify_trace_t1.log_line('%s request=%s: %s' % (TRACED_DECLINED, request, problem))
            return
        operations, traces = self.operations, {}
        try:
            # The save just ran; the restore has not. An identity warm (carry was seeded from slot zero a moment ago and nothing
            # ran between), so every program is compiled before its capture and the live bytes are unchanged.
            self.eager_carry_copy('restore')
            operations.synchronize_device(self.mesh)
            for operation in ('save', 'restore'):
                traces[operation], unused = capture_operation(operations, self.mesh,
                    lambda operation=operation: self.eager_carry_copy(operation))
        except BaseException:
            for trace in traces.values():
                operations.release_trace(self.mesh, trace)
            self.carry_traces = False
            raise
        operations.synchronize_device(self.mesh)
        self.carry_traces = traces
        verify_trace_t1.log_line('%s request=%s layers=%d audit=%d' % (TRACED_ENGAGED, request, len(self.carry),
                                                                      int(self.carry_trace_audit)))

    def carry_destination(self, operation):
        if operation == 'save':
            return [value for slot in self.carry for value in slot]
        return [value for helper in self.helpers for value in helper.live]

    def carry_readback(self, tensors):
        return [[self.operations.to_torch(part) for part in self.operations.get_device_tensors(tensor)] for tensor in tensors]

    def audited_carry_copy(self, operation):
        """The trace, then the eager loop over the same destination: bytes read back between and after must be equal on every chip.
        The eager bytes stand either way."""
        destination = self.carry_destination(operation)
        self.operations.execute_trace(self.mesh, self.carry_traces[operation], cq_id=0, blocking=False)
        self.operations.synchronize_device(self.mesh)
        traced = self.carry_readback(destination)
        self.eager_carry_copy(operation)
        self.operations.synchronize_device(self.mesh)
        eager = self.carry_readback(destination)
        differing = sum(1 for left, right in zip(traced, eager, strict=True) for a, b in zip(left, right, strict=True)
                        if not _bits_equal(a, b))
        total = sum(len(parts) for parts in traced)
        self.carry_audit_checked += total
        self.carry_audit_mismatches += differing
        verify_trace_t1.log_line('%s op=%s checked=%d mismatches=%d' % (TRACED_AUDIT, operation, total, differing))

    def close(self):
        if getattr(self, 'phase', None) == 'parked':
            # D2: a parked engine is idle by construction (park refuses anything else), so it closes as an idle one: the carry traces below
            # are released with it, and the inherited close, which refuses 'parked', takes it from there.
            self.phase = 'idle'
        traces = getattr(self, 'carry_traces', False)
        if isinstance(traces, dict) and getattr(self, 'phase', None) in ('idle', 'preparing', 'failed'):
            self.operations.synchronize_device(self.mesh)
            for trace in traces.values():
                self.operations.release_trace(self.mesh, trace)
            self.carry_traces = False
        super().close()

    # -- engine reuse: the parked phase (serving_parked_engines; QWEN_FAST_PARKED_ENGINES) -----------------------------------------
    def park_refusal(self):
        """Why this engine cannot park, or None. A parked engine keeps its captures for the process, so only an idle one with no ticket,
        sequential captures and sound retained blocks may."""
        if self.phase != 'idle':
            return 'engine phase %s' % self.phase
        if self.pending is not None or self.pending_key is not None:
            return 'a pending ticket'
        if getattr(self, 'replay_plan', None) is not None or getattr(self, 'target_attention_t16', False):
            return 'captures that are not sequential'
        for key, bucket in self.buckets.items():
            retained = getattr(bucket.get('fixture'), 'retained', None)
            if retained is not None and retained.poisoned:
                return 'the retained block of bucket %r is poisoned' % (key,)
        return None

    def park(self):
        """Fence, then park this engine for its next rebind - phase 'parked', no session, not resident - and return None; or return why it
        cannot park (park_refusal), changing nothing, and its owner closes it as today. The fence is close()'s. A block in flight is refused
        as close() refuses it; an engine that is already parked or closed raises (the owner guards that: ParkedEngineSet.park_engine)."""
        if self.phase in ('parked', 'closed'):
            raise ValueError('Only a live engine can park; this one is %s' % self.phase)
        if self.phase not in ('idle', 'preparing', 'failed'):
            raise ValueError('Finish or abort the pending verifier block before closing')
        self.operations.synchronize_device(self.mesh)
        reason = self.park_refusal()
        if reason is not None:
            return reason
        self.phase, self.pending, self.pending_key = 'parked', None, None
        self.session = None
        self.replay_mark = None
        if pair_module._resident is self:
            pair_module._resident = None
        return None

    def rebind_refusal(self, position, remaining, pages_shape):
        """Why this engine cannot be rebound to a request at `position` with `remaining` tokens to decode over a page table of
        `pages_shape`, or None - HOST ONLY: nothing is read from or written to the device, so the caller asks it BEFORE it takes a slot or
        writes anything, and a refusal costs a cold build and nothing else. The constructor's own checks on the request geometry, in its
        order, and the capture subset (design 5.9)."""
        if self.phase != 'parked':
            return 'engine phase %s, not parked' % self.phase
        if len(pages_shape) != 2 or pages_shape[0] != 1:
            return 'one request page table required'
        if tuple(pages_shape) != tuple(self.pages.shape):
            return 'the parked engine was captured over a %r page table; the request brings %r' % (tuple(self.pages.shape),
                                                                                                   tuple(pages_shape))
        try:
            widths = capture_widths(position, pages_shape[1] * 64, 16, remaining, self.capture_rows)
        except ValueError as failure:
            return str(failure)
        if not set(widths) <= set(self.captured_widths):
            return 'the request needs widths %r; the parked engine captured %r' % (widths, self.captured_widths)
        return None

    def rebind(self, session, pages):
        """Bind this parked engine to a request at its prefilled frontier, as the constructor binds a fresh one but capturing nothing
        (design 3.4). The constructor's host checks run first, in its order and with its messages, and a refusal leaves the engine parked
        with nothing written (rebind_refusal is the same checks before a session exists). Then: every fixture's page tables rewritten in
        full with the request's table (serving_parked_engines.write_page_tables, VerifierPageBinding.refresh's write), a stale packed
        adoption dropped, every bucket's first verify back on the unreplayed path with its retained block's flags reset (reset_retained),
        the initial snapshot saved from native slot 0 - which holds the request's prefilled state, adopted before this - and the carry
        seeded from it (_resident = self). The widths become a cold engine's for this request (R4): the captures keep (1, 2, 4), the
        engine asks and serves only what the request's own constructor would have captured."""
        import serving_parked_engines as reuse

        if self.phase != 'parked':
            raise ValueError('Only a parked engine can be rebound; this one is %s' % self.phase)
        if session.phase != 'idle' or session.pending is not None or session.finished or len(self.helpers) != 48:
            raise ValueError('An unfinished prefilled request and all native GDN helpers are required')
        if len(pages.shape) != 2 or pages.shape[0] != 1:
            raise ValueError('One request page table required')
        if tuple(pages.shape) != tuple(self.pages.shape):
            raise ValueError('The parked engine was captured over a %r page table; the request brings %r'
                             % (tuple(self.pages.shape), tuple(pages.shape)))
        widths = capture_widths(session.position, pages.shape[1] * 64, session.verifier_rows,
                                session.max_new_tokens - len(session.emitted), self.capture_rows)
        if not set(widths) <= set(self.captured_widths):
            raise ValueError('The request needs widths %r; the parked engine captured %r' % (widths, self.captured_widths))
        bindings = reuse.page_table_bindings(self)
        started = time.perf_counter()
        session.begin_preparation(session.request_id)
        self.session, self.position = session, session.position
        self.pages.copy_(pages)
        self.phase = 'preparing'
        pair_module.note_prefill()
        try:
            # The gate's 'pages' negative control swaps the writer for a no-op, for this call only (an instance attribute, popped by its owner).
            self.__dict__.get('page_table_writer', reuse.write_page_tables)(self.operations, self.mesh, bindings, self.pages)
            for key in [key for key in self.buckets if isinstance(key, tuple) and key[:1] == ('packed',)]:
                del self.buckets[key]
            for bucket in self.buckets.values():
                bucket['first'] = True
                if bucket['fixture'].retained is not None:
                    reset_retained(bucket['fixture'].retained)
            if not self.__dict__.get('keep_captured_widths'):
                # R4. (The gate's 'widths' negative control keeps the captured widths, for this call only.)
                self.widths = self.request_widths = tuple(widths)
            self.replay_mark = None
            for helper, snapshot in zip(self.helpers, self.initial, strict=True):
                helper.save(snapshot)
            self.validate_bindings()
            self.save_carry()
            self.rebind_ms = (time.perf_counter() - started) * 1000
            session.finish_preparation(session.request_id)
            self.phase = 'idle'
        except BaseException:
            self.phase = 'failed'
            session.fail_preparation(session.request_id)
            raise

    def proposal_rows(self, packed_rows=None):
        """QWEN_FAST_BUDGET_CAP: while any budget is left the engine answers its widest capture the
        scheduler will still schedule, not the widest that fits the remaining tokens: a request at its
        last 1-3 tokens drafts four rows and GreedySession.commit cuts the emission at the budget, so
        ordinary traffic never replays a one- or two-row capture (only an engine built without a
        four-row width, a budget under four at admission, and the last rows of the context do). A round
        the packed block serves answers the block's rows from one token left. Flag off, or a replay
        plan without the block's hint (its own width selection), the inherited answer.

        vLLM schedules a request at most max_model_len - 1 - position tokens (scheduler.py caps
        num_new_tokens there), so that bound, from the request page table (ceil(max_model_len / 64)
        pages, serving_runtime), keeps a full-width ticket the scheduler will offer whole."""
        remaining = self.session.max_new_tokens - len(self.session.emitted)
        if not budget_cap_enabled() or remaining < 1 or (packed_rows is None and getattr(self, 'replay_plan', None) is not None):
            return super().proposal_rows(packed_rows)
        if packed_rows is not None:
            if (type(packed_rows) is not int or packed_rows not in VERIFY_WIDTHS
                    or packed_rows > self.session.verifier_rows):
                raise ValueError('The packed block rows must be a supported verify width within the session verifier rows')
            return packed_rows
        schedulable = self.pages.shape[1] * 64 - 1 - self.session.position
        fits = [rows for rows in self.widths if rows <= schedulable]
        return max(fits) if fits else super().proposal_rows()

    def bucket_key(self, ticket):
        """The inherited key; a rebound engine (request_widths set) refuses a ticket whose width a cold engine of this request would not have
        captured (R4: it would replay a different, wider trace than the control's). serves() reports False for it."""
        key = super().bucket_key(ticket)
        if self.request_widths is not None and len(ticket.tokens) not in self.request_widths:
            raise ValueError('A rebound engine serves the widths %r of its request, not %d rows' % (self.request_widths, len(ticket.tokens)))
        return key

    def publish(self, prefix):
        if _replay_count is not None:
            bucket = self.buckets.get(self.pending_key)
            if bucket is not None and bucket.get('packed') is None:
                # R2: a sequential commit trace reads what this engine's own verify wrote; nothing may have replayed in between.
                check_replay_mark(self)
        if not trace_census.stage_log_enabled():
            return super().publish(prefix)
        request, rows = self.session.request_id, len(self.pending.tokens) if self.pending is not None else 'n/a'
        trace_census.stage(request, rows, 'publish')
        try:
            super().publish(prefix)
        except BaseException:
            trace_census.stage(request, rows, 'publish', 'failed')
            raise
        trace_census.stage(request, rows, 'publish', 'end')

    def verify(self, ticket):
        self.session.check_ticket(self.session.request_id, ticket)
        key = self.bucket_key(ticket)
        if self.phase != 'idle' or self.pending is not None or ticket.position != self.position or key not in self.buckets:
            raise ValueError('An idle engine and its next supported request ticket are required')
        self.phase, self.pending = 'verifying', ticket
        self.pending_key = key
        bucket = self.buckets[key]
        # QWEN_FAST_SEQ_STAGE_LOG: a flushed 'begin' line before each stage, so a stall names the stage it is in.
        marking = trace_census.stage_log_enabled()
        request, rows, first = self.session.request_id, len(ticket.tokens), ' first=%d' % bool(bucket.get('first'))

        def mark(name):
            if marking:
                trace_census.stage(request, rows, name, extra=first)
        try:
            binding_started = time.perf_counter()
            mark('validate')
            self.validate_bindings()
            carry_started = time.perf_counter()
            mark('restore')
            restored = self.restore_carry()
            started = time.perf_counter()
            mark('stage_inputs')
            stage_inputs(bucket['fixture'], ticket.tokens, ticket.position)
            staged = time.perf_counter()
            trace_ms = 0.0
            def operation():
                nonlocal trace_ms
                trace_started = time.perf_counter()
                result = self.operations.execute_trace(self.mesh, bucket['trace'], cq_id=0, blocking=True)
                trace_ms += (time.perf_counter() - trace_started) * 1000
                return result
            mark('execute_trace')
            if marking and bucket['first']:
                trace_census.first_replay(self, request, rows)
            if bucket['first'] or bucket['fixture'].retained is None:
                operation()
                mark('sync')
                self.operations.synchronize_device(self.mesh)
            else:
                bucket['fixture'].retained.replay(operation)
                mark('sync')
            if _replay_count is not None:
                # D1: the pair's verify notes R2's mark right after its replay; this class restates the whole verify, so it notes it here.
                self.replay_mark = _replay_count()
            if getattr(self, 'attention_audit', False) and bucket['fixture'].replay_reader is not None:
                bucket['fixture'].replay_reader.audit.check(ticket.position, ticket.tokens)
            replay_finished = time.perf_counter()
            mark('readback')
            if len(bucket['output']) > 2:
                predictions = self.shard_predictions(bucket['output'], len(ticket.tokens))
            else:
                logits, ids = bucket['output']
                tensor = logits if ids is None else ids
                parts = self.operations.get_device_tensors(tensor)
                if len(parts) != tp_shapes.chip_count():
                    raise AssertionError('%s chip-local outputs required' % tp_shapes.count_word())
                host = self.operations.to_torch(parts[0])
                predictions = (host.reshape(len(ticket.tokens), self.model.args.vocab_size).float().argmax(dim=-1)
                               if ids is None else host.reshape(-1)[:len(ticket.tokens)]).tolist()
            finished = time.perf_counter()
            if len(predictions) != len(ticket.tokens):
                raise AssertionError('Missing target prediction rows')
            bucket['first'] = False
            self.phase = 'verified'
            if os.environ.get('QWEN_FAST_PROFILE_DUMP_EVERY'):
                # The TP4 op profile's read-back cadence (packed_verifier.dump_device_profiler_every): a sequential
                # step is a verify replay too. Imported here so an unprofiled run never touches the module.
                from packed_verifier import dump_device_profiler_every
                dump_device_profiler_every(self.operations, self.mesh)
            return predictions, dict(input_ms=(staged - started) * 1000,
                verify_readback_ms=(finished - staged) * 1000,
                binding_validation_ms=(carry_started - binding_started) * 1000,
                carry_restore_ms=(started - carry_started) * 1000, carry_restored=restored,
                singleton_position_uploads=getattr(bucket['fixture'], 'last_singleton_uploads', None),
                blocking_trace_host_ms=trace_ms,
                replay_checks_sync_ms=(replay_finished - staged) * 1000 - trace_ms,
                output_readback_host_ms=(finished - replay_finished) * 1000)
        except BaseException:
            self.phase = 'failed'
            self.session.fail_verification(self.session.request_id, ticket)
            raise
