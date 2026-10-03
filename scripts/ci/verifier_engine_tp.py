"""verifier_engine.VerifierEngine (the sequential engine) at any served width.

verifier_engine.py is one of serving_bundle.package's eight critical staged-source files (test_quad_draft holds it
untouched: the frozen bundle's inventory sha is checked against the checkout's bytes), and its sequential verify readback
requires exactly two chip-local outputs. So the four-card engine is a subclass with that one method, verify, restated with the
chip count from tp_shapes (the pair's message is unchanged: 'Two chip-local outputs required'), and proposal_rows, which
QWEN_FAST_BUDGET_CAP widens (below); every other method is inherited. publish is inherited too, with QWEN_FAST_SEQ_STAGE_LOG's begin and end lines around it.
serving_request_factory.device_components builds this class at four cards and the pair's at the pair.
"""

import os
import time
from contextlib import ExitStack

from force_argmax import sample_rows
from gdn_multitoken_conv import release_owned
from serving_fast_request import budget_cap_enabled
import trace_census
import verify_trace_t1
from verifier_engine import VERIFY_WIDTHS, VerifierEngine as PairVerifierEngine
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
        super().__init__(*args, sampler=sampler, **options)

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

    def publish(self, prefix):
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
