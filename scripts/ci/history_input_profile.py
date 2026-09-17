"""Bounded host-side publication attribution without changed tensor operations."""

from contextlib import ExitStack, contextmanager
from functools import wraps
from time import perf_counter, process_time
from unittest.mock import patch


@contextmanager
def profile_inputs(arm_class, torch_module, records):
    original = arm_class.publication

    @wraps(original)
    def publication(arm, features, prefix, *, position):
        stages = []

        def observed(callback, stage):
            @wraps(callback)
            def call(*args, **kwargs):
                wall, cpu, passed = perf_counter(), process_time(), False
                try:
                    result = callback(*args, **kwargs)
                    passed = True
                    return result
                finally:
                    stages.append(dict(stage=stage, passed=passed,
                        host_ms=(perf_counter() - wall) * 1000,
                        process_cpu_ms=(process_time() - cpu) * 1000))
            return call

        wall, passed = perf_counter(), False
        try:
            with ExitStack() as stack:
                targets = [(arm.history.operations, name, name) for name in
                    ('pad', 'from_torch', 'deallocate', 'get_device_tensors')]
                targets += [(arm.history.rotary, 'tables', 'rotary_tables'),
                    (torch_module, 'ones', 'host_ones'), (torch_module, 'zeros', 'host_zeros')]
                for owner, name, stage in targets:
                    stack.enter_context(patch.object(owner, name, observed(getattr(owner, name), stage)))
                result = original(arm, features, prefix, position=position)
            passed = True
            return result
        finally:
            records.append(dict(position=position, prefix=prefix, stages=stages, passed=passed,
                host_ms=(perf_counter() - wall) * 1000, added_device_fences=False,
                scope='Nested host-call durations; do not sum overlapping stages'))

    with patch.object(arm_class, 'publication', publication):
        yield
