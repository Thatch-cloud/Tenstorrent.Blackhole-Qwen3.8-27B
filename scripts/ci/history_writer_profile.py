"""Observe existing incremental writer stages without adding synchronization."""

from contextlib import contextmanager
from time import perf_counter, process_time
from unittest.mock import patch


@contextmanager
def profile_writers(module, records):
    original = module.incremental_history

    @contextmanager
    def scoped(history_class, publication_module, updates, *, writer_factory=module.prepare_dma):
        active = []

        def observed(callback, stage, *args, **kwargs):
            if not active:
                raise ValueError('Writer execution outside observed assembly')
            wall, cpu, passed = perf_counter(), process_time(), False
            try:
                result = callback(*args, **kwargs)
                passed = True
                return result
            finally:
                active[-1]['stages'].append(dict(stage=stage, passed=passed,
                    host_ms=(perf_counter() - wall) * 1000,
                    process_cpu_ms=(process_time() - cpu) * 1000))

        def writer(*args, **kwargs):
            operation = observed(writer_factory, 'writer_build', *args, **kwargs)
            return lambda: observed(operation, 'writer_enqueue')

        with original(history_class, publication_module, updates, writer_factory=writer):
            prepare = history_class.prepare_projected

            def assemble(history, added, prefix, *, position):
                record = dict(position=position, prefix=prefix, stages=[], passed=False,
                    added_device_fences=False)
                if active:
                    raise ValueError('Nested history assembly is not supported')
                active.append(record)
                wall, cpu = perf_counter(), process_time()
                synchronize = history.operations.synchronize_device
                try:
                    with patch.object(history.operations, 'synchronize_device',
                            lambda *args, **kwargs: observed(synchronize, 'existing_sync', *args, **kwargs)):
                        result = prepare(history, added, prefix, position=position)
                    stages = [entry['stage'] for entry in record['stages']]
                    if stages != ['writer_build'] * 10 + ['writer_enqueue'] * 10 + ['existing_sync']:
                        raise ValueError('Ten unchanged writers and one existing synchronization required')
                    record['passed'] = True
                    return result
                finally:
                    record.update(host_ms=(perf_counter() - wall) * 1000,
                        process_cpu_ms=(process_time() - cpu) * 1000)
                    records.append(record)
                    active.pop()

            with patch.object(history_class, 'prepare_projected', assemble):
                yield

    with patch.object(module, 'incremental_history', scoped):
        yield
