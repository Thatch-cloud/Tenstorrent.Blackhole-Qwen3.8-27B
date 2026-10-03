"""Observe actual prepared DSpark trace executions without editing qualified proposal code."""

from contextlib import contextmanager
import os
import time

from tensix_mlp_profile import require_profile_mode


class DraftProfile:
    def __init__(self, operations, mesh, *, signpost=None):
        require_profile_mode(os.environ, True)
        if signpost is None:
            from tracy import signpost
        self.operations, self.mesh, self.signpost = operations, mesh, signpost
        self.owner, self.role = None, None
        self.records, self.counts = [], {}
        self.failed, self.installed, self.restored = False, False, False

    @contextmanager
    def observe(self, prepared, role):
        if self.owner is not None or self.failed or role not in ('capture_warmup', 'proposal'):
            raise ValueError('One non-nested prepared proposal attribution scope required')
        self.owner, self.role = prepared, role
        try:
            yield
        except BaseException:
            self.failed = True
            raise
        finally:
            self.owner, self.role = None, None

    def replay(self, original, mesh, trace, *args, **kwargs):
        prepared = self.owner
        if prepared is None or trace != getattr(prepared, 'trace', None):
            return original(mesh, trace, *args, **kwargs)
        if (self.failed or mesh is not self.mesh or prepared.mesh is not self.mesh
                or prepared.operations is not self.operations or prepared.device.max_drafts != 15
                or prepared.closed or args or kwargs != dict(cq_id=0, blocking=True)):
            raise ValueError('Unchanged actual fifteen-query prepared proposal trace required')
        identifier = int(trace)
        ordinal = self.counts.get(identifier, 0)
        record = dict(block=len(self.records), position=prepared.device.position, rows=15,
            trace_id=identifier, trace_ordinal=ordinal, first_replay=ordinal == 0, role=self.role)
        label = f'qwen_draft_{record["block"]}_pos{record["position"]}_trace{identifier}_{self.role}'
        operations = self.operations
        operations.synchronize_device(mesh)
        operations.ReadDeviceProfiler(mesh)
        print('QWEN_DRAFT_PROFILE_BEGIN ' + label, flush=True)
        self.signpost(label + '_begin')
        started = time.perf_counter()
        try:
            result = original(mesh, trace, *args, **kwargs)
            operations.synchronize_device(mesh)
            record.update(label=label, instrumented_host_ms=(time.perf_counter() - started) * 1000)
            self.records.append(record)
            self.counts[identifier] = ordinal + 1
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.signpost(label + '_end')
            operations.ReadDeviceProfiler(mesh)
            print('QWEN_DRAFT_PROFILE_END ' + label, flush=True)

    @contextmanager
    def install(self, prepared_type=None):
        if self.installed or self.restored or self.failed:
            raise ValueError('Fresh single-use proposal profiler required')
        if prepared_type is None:
            from dspark_prepared_proposal import PreparedDSparkProposal
            prepared_type = PreparedDSparkProposal
        original_init, original_propose = prepared_type.__init__, prepared_type.propose
        original_trace = self.operations.execute_trace
        def initialize(prepared, *args, **kwargs):
            with self.observe(prepared, 'capture_warmup'):
                original_init(prepared, *args, **kwargs)
        def propose(prepared, *args, **kwargs):
            with self.observe(prepared, 'proposal'):
                return original_propose(prepared, *args, **kwargs)
        def replay(mesh, trace, *args, **kwargs):
            return self.replay(original_trace, mesh, trace, *args, **kwargs)
        prepared_type.__init__, prepared_type.propose = initialize, propose
        self.operations.execute_trace = replay
        self.installed = True
        try:
            yield self
        finally:
            unchanged = (prepared_type.__init__ is initialize and prepared_type.propose is propose
                and self.operations.execute_trace is replay)
            prepared_type.__init__, prepared_type.propose = original_init, original_propose
            self.operations.execute_trace = original_trace
            self.installed = False
            self.restored = True
            if not unchanged:
                self.failed = True
                raise RuntimeError('Profiler hooks changed during the request')

    def summary(self):
        if (self.failed or not self.restored or self.installed or self.owner is not None
                or len(self.counts) != 1 or len(self.records) < 4
                or self.records[0]['role'] != 'capture_warmup'
                or any(record['role'] != 'proposal' for record in self.records[1:])):
            raise ValueError('One complete restored multi-block proposal trace profile required')
        return dict(records=list(self.records), trace_counts=dict(self.counts), restored=True,
            scope='Instrumented fifteen-query proposal trace only; excludes staging/audits/readback; not TG')
