"""Bounded host-wall attribution inside combined-runtime history publication."""

from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from unittest.mock import patch


@contextmanager
def profile_publications(arm_class, records, emit, *, clock=perf_counter):
    original = arm_class.publication
    owners, counts = [], []

    @wraps(original)
    def publication(arm, features, prefix, *, position):
        if not any(owner is arm for owner in owners):
            if len(owners) == 2:
                return original(arm, features, prefix, position=position)
            owners.append(arm)
            counts.append(0)
        ordinal = next(index for index, owner in enumerate(owners) if owner is arm)
        sample = counts[ordinal]
        if sample >= 3:
            return original(arm, features, prefix, position=position)
        counts[ordinal] += 1
        stages = []

        def observed(callback, name):
            @wraps(callback)
            def call(*args, **kwargs):
                started = clock()
                try:
                    return callback(*args, **kwargs)
                finally:
                    stages.append(dict(stage=name, host_ms=(clock() - started) * 1000))
            return call

        started, passed = clock(), False
        try:
            with patch.object(arm.projection, 'project', observed(arm.projection.project, 'projection')), \
                    patch.object(arm.history, 'prepare_projected',
                        observed(arm.history.prepare_projected, 'bank_assembly')):
                result = original(arm, features, prefix, position=position)
            if [stage['stage'] for stage in stages] != ['projection', 'bank_assembly']:
                raise ValueError('One projection and bank assembly required for sampled publication')
            passed = True
            return result
        finally:
            total = (clock() - started) * 1000
            record = dict(request_ordinal=ordinal, sample=sample, position=position, prefix=prefix,
                stages=stages, total_ms=total,
                other_ms=total - sum(stage['host_ms'] for stage in stages), passed=passed,
                performance_qualified=False, added_device_fences=False)
            records.append(record)
            emit(record)

    with patch.object(arm_class, 'publication', publication):
        yield
