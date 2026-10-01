"""verifier_engine.VerifierEngine (the sequential engine) at any served width.

verifier_engine.py is one of serving_bundle.package's eight critical staged-source files (test_quad_draft holds it
untouched: the frozen bundle's inventory sha is checked against the checkout's bytes), and its sequential verify readback
requires exactly two chip-local outputs. So the four-card engine is a subclass with that one method, verify, restated with the
chip count from tp_shapes (the pair's message is unchanged: 'Two chip-local outputs required'), and proposal_rows, which
QWEN_FAST_BUDGET_CAP widens (below); every other method is inherited. publish is inherited too, with QWEN_FAST_SEQ_STAGE_LOG's begin and end lines around it.
serving_request_factory.device_components builds this class at four cards and the pair's at the pair.
"""

import time

from serving_fast_request import budget_cap_enabled
import trace_census
from verifier_engine import VERIFY_WIDTHS, VerifierEngine as PairVerifierEngine
from verifier_inputs import stage_inputs
import tp_shapes


class VerifierEngine(PairVerifierEngine):
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
