"""Single-request captured verifier buckets; prepare only after that request's prefill."""

import time

from attention_batch import capture_operation
from force_argmax import sample_rows
from gdn_commit_dma import prepare
from gdn_multitoken_conv import addresses, release_owned
from model_batch import ModelBatch
from verifier_inputs import stage_inputs


class VerifierEngine:
    def __init__(self, model, session, pages, helpers, *, sampler=None):
        import ttnn

        if session.phase != 'idle' or session.pending is not None or session.finished or len(helpers) != 48:
            raise ValueError('An unfinished prefilled request and all native GDN helpers are required')
        self.model, self.session, self.pages, self.helpers = model, session, pages, helpers
        self.operations, self.mesh, self.sampler = ttnn, model.mesh_device, sampler
        self.position = session.position
        self.phase, self.pending = 'preparing', None
        self.initial, self.buckets = [], {}
        self.native_addresses = [[addresses(ttnn, value) for value in helper.live] for helper in helpers]
        self.widths = (1, 2, 4, 8, 16, 32) if session.verifier_rows == 32 else (1, 2, 4, 8, 16)
        started = time.perf_counter()
        try:
            for helper in helpers:
                self.initial.append(helper.allocate())
            for helper, snapshot in zip(helpers, self.initial, strict=True):
                helper.save(snapshot)
            for rows in self.widths:
                bucket = dict(rows=rows, checkpoints=[], fixture=None, trace=None, output=None, commits={}, first=True)
                self.buckets[rows] = bucket
                for helper in helpers:
                    bucket['checkpoints'].append(helper.allocate())
                bucket['fixture'] = self.fixture(rows, bucket['checkpoints'], retain=rows > 1)
            for rows, bucket in self.buckets.items():
                self.restore_initial()
                warm = self.fixture(rows, bucket['checkpoints'], retain=False)
                result = None
                try:
                    result = self.operation(warm)
                    ttnn.synchronize_device(self.mesh)
                finally:
                    if result is not None:
                        release_owned(ttnn, [value for value in result if value is not None])
                    warm.close()
            for rows, bucket in self.buckets.items():
                self.restore_initial()
                bucket['trace'], bucket['output'] = capture_operation(ttnn, self.mesh,
                    lambda bucket=bucket: self.operation(bucket['fixture']))
                if rows > 1:
                    layers = [[*state.entry, result['states'], *result['packed_conv_states'],
                               state.gdn.rec_state, *state.gdn.conv_states, *checkpoint]
                              for state, result, checkpoint in bucket['fixture'].retained.records]
                    publications = {prefix: prepare(self.mesh, layers, prefix) for prefix in range(rows + 1)}
                    for publication in publications.values():
                        publication()
                    ttnn.synchronize_device(self.mesh)
                    for prefix, publication in publications.items():
                        bucket['commits'][prefix], unused = capture_operation(ttnn, self.mesh, publication)
            self.restore_initial()
            ttnn.synchronize_device(self.mesh)
            self.validate_bindings()
            self.setup_ms = (time.perf_counter() - started) * 1000
            self.phase = 'idle'
        except BaseException:
            self.phase = 'failed'
            self.close()
            raise

    def fixture(self, rows, checkpoints, *, retain):
        return ModelBatch(self.model, [1] * rows, self.position, self.pages, self.helpers, checkpoints,
            0 if rows == 1 else rows, serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True,
            skip_row_clones=True, hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
            batch_conv=True, packed_checkpoints=True, retain_records=retain, ordered_cache=True)

    def operation(self, fixture):
        logits = fixture.run(sharded_logits=self.sampler is not None)
        try:
            ids = sample_rows(self.sampler, logits, fixture.rows, self.operations) if self.sampler is not None else None
            return logits, ids
        except BaseException:
            self.operations.deallocate(logits)
            raise

    def restore_initial(self):
        for helper, snapshot in zip(self.helpers, self.initial, strict=True):
            helper.restore(snapshot)

    def validate_bindings(self):
        if any(helper.gdn.B != 8 or not helper.gdn._stable_state for helper in self.helpers):
            raise ValueError('Native stable B8 state contract changed')
        if [[addresses(self.operations, value) for value in helper.live] for helper in self.helpers] != self.native_addresses:
            raise ValueError('Native GDN buffers changed under captured verifier')

    def verify(self, ticket):
        self.session.check_ticket(self.session.request_id, ticket)
        if self.phase != 'idle' or self.pending is not None or ticket.position != self.position or len(ticket.tokens) not in self.buckets:
            raise ValueError('An idle engine and its next supported request ticket are required')
        self.phase, self.pending = 'verifying', ticket
        bucket = self.buckets[len(ticket.tokens)]
        try:
            self.validate_bindings()
            started = time.perf_counter()
            stage_inputs(bucket['fixture'], ticket.tokens, ticket.position)
            staged = time.perf_counter()
            operation = lambda: self.operations.execute_trace(self.mesh, bucket['trace'], cq_id=0, blocking=True)
            if bucket['first'] or bucket['fixture'].retained is None:
                operation()
                self.operations.synchronize_device(self.mesh)
            else:
                bucket['fixture'].retained.replay(operation)
            logits, ids = bucket['output']
            tensor = logits if ids is None else ids
            parts = self.operations.get_device_tensors(tensor)
            if len(parts) != 2:
                raise AssertionError('Two chip-local outputs required')
            host = self.operations.to_torch(parts[0])
            predictions = (host.reshape(len(ticket.tokens), self.model.args.vocab_size).float().argmax(dim=-1)
                           if ids is None else host.reshape(-1)[:len(ticket.tokens)]).tolist()
            finished = time.perf_counter()
            if len(predictions) != len(ticket.tokens):
                raise AssertionError('Missing target prediction rows')
            bucket['first'] = False
            self.phase = 'verified'
            return predictions, dict(input_ms=(staged - started) * 1000,
                                      verify_readback_ms=(finished - staged) * 1000)
        except BaseException:
            self.phase = 'failed'
            self.session.fail_verification(self.session.request_id, ticket)
            raise

    def publish(self, prefix):
        ticket = self.pending
        if self.phase != 'verified' or ticket is None or self.session.pending is not ticket or self.session.phase != 'committing':
            raise ValueError('Publication requires the live verified ticket during its owner decision')
        if type(prefix) is not int or not 0 <= prefix <= len(ticket.tokens):
            raise ValueError('Selected prefix outside verified block')
        self.phase = 'committing'
        bucket = self.buckets[len(ticket.tokens)]
        try:
            if bucket['fixture'].retained is not None:
                bucket['fixture'].retained.commit(prefix, dma=True, synchronize=True,
                    publication=lambda selected: self.operations.execute_trace(self.mesh, bucket['commits'][selected], cq_id=0, blocking=True))
            else:
                self.validate_bindings()
                if prefix == 0:
                    for helper, snapshot in zip(self.helpers, bucket['checkpoints'], strict=True):
                        helper.restore(snapshot)
                self.operations.synchronize_device(self.mesh)
            self.position += prefix
            self.phase, self.pending = 'idle', None
        except BaseException:
            self.phase = 'failed'
            raise

    def close(self):
        if self.phase == 'closed':
            return
        if self.phase not in ('idle', 'preparing', 'failed'):
            raise ValueError('Finish or abort the pending verifier block before closing')
        self.operations.synchronize_device(self.mesh)
        for bucket in self.buckets.values():
            for trace in bucket['commits'].values():
                self.operations.release_trace(self.mesh, trace)
            if bucket['trace'] is not None:
                self.operations.release_trace(self.mesh, bucket['trace'])
        for bucket in self.buckets.values():
            if bucket['output'] is not None:
                release_owned(self.operations, [value for value in bucket['output'] if value is not None])
            if bucket['fixture'] is not None:
                bucket['fixture'].close()
            release_owned(self.operations, [value for snapshot in bucket['checkpoints'] for value in snapshot])
        release_owned(self.operations, [value for snapshot in self.initial for value in snapshot])
        self.buckets.clear()
        self.initial.clear()
        self.phase, self.pending = 'closed', None
