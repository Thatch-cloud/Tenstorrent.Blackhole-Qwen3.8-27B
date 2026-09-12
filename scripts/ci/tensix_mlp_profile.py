"""Instrument complete native/streamed MLP trace replays without changing their operations."""

import time

from request_verifier_profile import PROFILER_FLAGS


def require_profile_mode(environment, enabled):
    if type(enabled) is not bool:
        raise ValueError('Explicit boolean attribution mode required')
    if enabled:
        if any(environment.get(name) != '1' for name in PROFILER_FLAGS):
            raise ValueError('Attribution requires the complete incremental device profiler configuration')
    elif any(environment.get(name) not in (None, '', '0') for name in PROFILER_FLAGS):
        raise ValueError('Performance comparisons must not run under the device profiler')


class MlpTraceProfile:
    def __init__(self, operations, mesh, traces, *, signpost=None, rows=8):
        if type(rows) is not int or rows not in (8, 16):
            raise ValueError('Qualified T8 or T16 MLP profile required')
        if len(traces) != 2 or len({int(trace) for trace in traces}) != 2:
            raise ValueError('Distinct native and streamed trace IDs required')
        if signpost is None:
            from tracy import signpost
        self.operations, self.mesh, self.traces, self.signpost = operations, mesh, traces, signpost
        self.rows = rows
        self.records, self.counts = [], [0, 0]
        self.failed = False

    def replay(self, arm, pattern, role, sample=-1):
        if (self.failed or type(arm) is not int or arm not in (0, 1)
                or type(pattern) is not int or pattern not in (0, 1, 2)
                or role not in ('audit', 'stale', 'measurement') or type(sample) is not int
                or (role == 'measurement' and sample not in (0, 1, 2, 3))
                or (role != 'measurement' and sample != -1)):
            raise ValueError('One known trace, fixture and replay role required')
        operations = self.operations
        record = dict(block=len(self.records), arm=arm, pattern=pattern, role=role, sample=sample,
            rows=self.rows, trace_id=int(self.traces[arm]), trace_ordinal=self.counts[arm],
            first_replay=self.counts[arm] == 0)
        label = f'qwen_mlp_arm{arm}_pattern{pattern}_{role}_{sample}_ordinal{self.counts[arm]}'
        operations.synchronize_device(self.mesh)
        operations.ReadDeviceProfiler(self.mesh)
        print('QWEN_MLP_PROFILE_BEGIN ' + label, flush=True)
        self.signpost(label + '_begin')
        started = time.perf_counter()
        try:
            operations.execute_trace(self.mesh, self.traces[arm], cq_id=0, blocking=True)
            operations.synchronize_device(self.mesh)
            record.update(label=label, instrumented_host_ms=(time.perf_counter() - started) * 1000)
            self.records.append(record)
            self.counts[arm] += 1
        except BaseException:
            self.failed = True
            raise
        finally:
            self.signpost(label + '_end')
            operations.ReadDeviceProfiler(self.mesh)
            print('QWEN_MLP_PROFILE_END ' + label, flush=True)

    def summary(self):
        expected = 10 if self.rows == 8 else 9
        if self.failed or self.counts != [expected, expected]:
            raise ValueError('All configured complete MLP replays must finish')
        return dict(records=list(self.records), trace_ids=[int(trace) for trace in self.traces],
            trace_counts=list(self.counts), scope='Instrumented complete MLP replays; not throughput or promotion')
