"""verifier_engine.VerifierEngine (the sequential engine) at any served width.

verifier_engine.py is one of serving_bundle.package's eight critical staged-source files (test_quad_draft holds it
untouched: the frozen bundle's inventory sha is checked against the checkout's bytes), and its sequential verify readback
requires exactly two chip-local outputs. So the four-card engine is a subclass with that one method, verify, restated with the
chip count from tp_shapes (the pair's message is unchanged: 'Two chip-local outputs required'), and proposal_rows, which
QWEN_FAST_BUDGET_CAP widens for a block round (below); every other method is inherited.
serving_request_factory.device_components builds this class at four cards and the pair's at the pair.
"""

import time

from serving_fast_request import budget_cap_enabled
from verifier_engine import VERIFY_WIDTHS, VerifierEngine as PairVerifierEngine
from verifier_inputs import stage_inputs
import tp_shapes


class VerifierEngine(PairVerifierEngine):
    def proposal_rows(self, packed_rows=None):
        """QWEN_FAST_BUDGET_CAP: a round the packed block serves answers the block's rows from one token
        left (the inherited answer needs the whole block's worth of budget): GreedySession.commit cuts
        the user's emission at its budget. Flag off, or without the block's hint, the inherited answer."""
        remaining = self.session.max_new_tokens - len(self.session.emitted)
        if not budget_cap_enabled() or remaining < 1 or packed_rows is None:
            return super().proposal_rows(packed_rows)
        if (type(packed_rows) is not int or packed_rows not in VERIFY_WIDTHS
                or packed_rows > self.session.verifier_rows):
            raise ValueError('The packed block rows must be a supported verify width within the session verifier rows')
        return packed_rows

    def verify(self, ticket):
        self.session.check_ticket(self.session.request_id, ticket)
        key = self.bucket_key(ticket)
        if self.phase != 'idle' or self.pending is not None or ticket.position != self.position or key not in self.buckets:
            raise ValueError('An idle engine and its next supported request ticket are required')
        self.phase, self.pending = 'verifying', ticket
        self.pending_key = key
        bucket = self.buckets[key]
        try:
            binding_started = time.perf_counter()
            self.validate_bindings()
            carry_started = time.perf_counter()
            restored = self.restore_carry()
            started = time.perf_counter()
            stage_inputs(bucket['fixture'], ticket.tokens, ticket.position)
            staged = time.perf_counter()
            trace_ms = 0.0
            def operation():
                nonlocal trace_ms
                trace_started = time.perf_counter()
                result = self.operations.execute_trace(self.mesh, bucket['trace'], cq_id=0, blocking=True)
                trace_ms += (time.perf_counter() - trace_started) * 1000
                return result
            if bucket['first'] or bucket['fixture'].retained is None:
                operation()
                self.operations.synchronize_device(self.mesh)
            else:
                bucket['fixture'].retained.replay(operation)
            if getattr(self, 'attention_audit', False) and bucket['fixture'].replay_reader is not None:
                bucket['fixture'].replay_reader.audit.check(ticket.position, ticket.tokens)
            replay_finished = time.perf_counter()
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
