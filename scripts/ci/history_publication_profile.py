"""Opt-in host-wall attribution for history publication; no additional device fences."""

from contextlib import contextmanager
import gc
import time


class HistoryPublicationProfile:
    def __init__(self):
        self.records = []
        self.active = None
        self.used = False
        self.gc_ns = 0
        self.gc_count = 0
        self.gc_started = None

    def observe_gc(self, phase, info):
        if phase == 'start':
            self.gc_started = time.perf_counter_ns()
        elif phase == 'stop' and self.gc_started is not None:
            self.gc_ns += time.perf_counter_ns() - self.gc_started
            self.gc_count += 1
            self.gc_started = None

    @contextmanager
    def stage(self, name):
        started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        gc_started, gc_count = self.gc_ns, self.gc_count
        passed = False
        try:
            yield
            passed = True
        finally:
            self.records.append(dict(stage=name, position=self.active[0], prefix=self.active[1],
                passed=passed, host_ms=(time.perf_counter_ns() - started) / 1e6,
                process_cpu_ms=(time.process_time_ns() - cpu_started) / 1e6,
                gc_ms=(self.gc_ns - gc_started) / 1e6, gc_collections=self.gc_count - gc_count))

    @contextmanager
    def install(self, history, module=None):
        if module is None:
            import dspark_stable_history as module
        if self.used or any(name in vars(history) for name in ('prepare_publication', 'prepare_projected')):
            raise ValueError('Fresh profiler and unmodified history methods required')
        self.used = True
        publish, assemble, project = history.prepare_publication, history.prepare_projected, module.project_chunks

        def publication(features, prefix, *, position):
            if self.active is not None:
                raise ValueError('Nested history publication is not supported')
            self.active = position, prefix
            try:
                with self.stage('history_total'):
                    return publish(features, prefix, position=position)
            finally:
                self.active = None

        def assembly(*args, **kwargs):
            if self.active is None:
                return assemble(*args, **kwargs)
            with self.stage('history_bank_assembly'):
                return assemble(*args, **kwargs)

        def projection(*args, **kwargs):
            if self.active is None:
                return project(*args, **kwargs)
            with self.stage('history_feature_projection'):
                return project(*args, **kwargs)

        history.prepare_publication, history.prepare_projected = publication, assembly
        module.project_chunks = projection
        callback = self.observe_gc
        gc.callbacks.append(callback)
        try:
            yield self
        finally:
            gc.callbacks.remove(callback)
            unchanged = (history.prepare_publication is publication and history.prepare_projected is assembly
                and module.project_chunks is projection)
            del history.prepare_publication
            del history.prepare_projected
            module.project_chunks = project
            if not unchanged:
                raise RuntimeError('History publication profiler hooks changed during execution')
