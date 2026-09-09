"""Observe real request verification calls without replacing their trace or transaction."""

from contextlib import contextmanager
import cProfile
import os
import pstats
import time


PROFILER_FLAGS = ('TTNN_OP_PROFILER', 'TT_METAL_DEVICE_PROFILER', 'TT_METAL_PROFILER_TRACE_TRACKING',
    'TT_METAL_PROFILER_CPP_POST_PROCESS')


class RequestVerifierProfile:
    def __init__(self, operations, mesh, *, signpost=None):
        if not all(os.environ.get(name) == '1' for name in PROFILER_FLAGS):
            raise ValueError('Request attribution requires all four runtime profiler flags')
        if signpost is None:
            from tracy import signpost
        self.operations, self.mesh, self.signpost = operations, mesh, signpost
        self.records, self.trace_counts = [], {}
        self.active = False
        self.host_profile = cProfile.Profile()

    @contextmanager
    def __call__(self, engine, ticket):
        if (self.active or engine.mesh is not self.mesh or engine.operations is not self.operations
                or not engine.commit_only_gdn or not engine.norm_batch or not engine.native_sampling_rows
                or engine.attention_replay or engine.phase != 'idle'):
            raise ValueError('Profile the unchanged idle commit-only native-attention verifier')
        key = engine.bucket_key(ticket)
        bucket = engine.buckets[key]
        trace_id = int(bucket['trace'])
        ordinal = self.trace_counts.get(trace_id, 0)
        record = dict(block=len(self.records), position=ticket.position, rows=len(ticket.tokens),
            trace_id=trace_id, trace_ordinal=ordinal, first_replay=bool(bucket['first']))
        label = f'qwen_request_verify_{record["block"]}_pos{ticket.position}_t{len(ticket.tokens)}_trace{trace_id}'
        operations = self.operations
        operations.synchronize_device(self.mesh)
        operations.ReadDeviceProfiler(self.mesh)
        self.active = True
        print('QWEN_REQUEST_VERIFY_BEGIN ' + label, flush=True)
        self.signpost(label + '_begin')
        self.host_profile.enable()
        started = time.perf_counter()
        try:
            yield
            operations.synchronize_device(self.mesh)
            if engine.phase != 'verified' or engine.pending is not ticket:
                raise AssertionError('Attribution must surround one successful real verification')
            record.update(label=label, instrumented_host_ms=(time.perf_counter() - started) * 1000)
            self.records.append(record)
            self.trace_counts[trace_id] = ordinal + 1
        finally:
            self.host_profile.disable()
            self.signpost(label + '_end')
            operations.ReadDeviceProfiler(self.mesh)
            self.active = False
            print('QWEN_REQUEST_VERIFY_END ' + label, flush=True)

    def summary(self):
        if self.active or len(self.records) < 3 or sum(record['rows'] == 8 for record in self.records) < 3:
            raise ValueError('Complete multi-block request attribution required')
        calls = []
        for (filename, line, name), (primitive, total, own, cumulative, callers) in pstats.Stats(self.host_profile).stats.items():
            calls.append(dict(file=filename, line=line, function=name, primitive_calls=primitive,
                total_calls=total, self_seconds=own, cumulative_seconds=cumulative))
        return dict(records=list(self.records), trace_counts=dict(self.trace_counts),
            host_calls=sorted(calls, key=lambda entry: -entry['cumulative_seconds'])[:40],
            scope='Instrumented actual verify calls including input staging and readback; profiler dumps and correctness digests outside markers; not throughput')
