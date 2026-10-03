"""Host-wall publication attribution; no added device fences or GC policy changes."""

from contextlib import contextmanager
import gc
import time


class PublicationDiagnostics:
    def __init__(self):
        self.records = []

    @contextmanager
    def stage(self, name, position, prefix):
        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        pending, pauses = {}, []

        def observe(phase, info):
            generation = info['generation']
            if phase == 'start':
                pending[generation] = time.perf_counter_ns()
            elif phase == 'stop' and generation in pending:
                pauses.append(dict(generation=generation,
                    duration_ms=(time.perf_counter_ns() - pending.pop(generation)) / 1e6))

        gc.callbacks.append(observe)
        passed = False
        try:
            yield
            passed = True
        finally:
            gc.callbacks.remove(observe)
            self.records.append(dict(stage=name, position=position, prefix=prefix, passed=passed,
                host_ms=(time.perf_counter_ns() - started) / 1e6,
                process_cpu_ms=(time.process_time_ns() - cpu_started) / 1e6, gc_pauses=pauses))
