"""Diagnostic host intervals only; preserves audited proposal execution."""

from contextlib import ExitStack, contextmanager
from time import perf_counter, process_time
from unittest.mock import patch


@contextmanager
def observe_audit(module, publication_class, emit, *, clock=perf_counter):
    sequence = 0
    active = None

    def timed(name, operation):
        def invoke(*args, **kwargs):
            nonlocal sequence
            if active is None:
                return operation(*args, **kwargs)
            sequence += 1
            identifier = sequence
            metadata = dict(event='audit_phase', phase=name, identifier=identifier,
                proposal=active, performance_qualified=False,
                scope='Nested host intervals; not device time or committed TG')
            emit(dict(metadata, status='started', elapsed_ms=None))
            started = clock()
            passed = False
            try:
                result = operation(*args, **kwargs)
                passed = True
                return result
            finally:
                emit(dict(metadata, status='completed' if passed else 'failed',
                    elapsed_ms=(clock() - started) * 1000))
        return invoke

    prepared = module.PreparedDSparkProposal
    original_propose = prepared.propose

    def propose(instance, *args, **kwargs):
        nonlocal active
        if not instance.audit:
            return original_propose(instance, *args, **kwargs)
        if active is not None:
            raise ValueError('Nested audited proposals unsupported')
        active = dict(position=instance.device.position, trace=str(instance.trace))
        try:
            with patch.object(instance.operations, 'execute_trace',
                    timed('blocking_replay', instance.operations.execute_trace)):
                return timed('proposal_total', original_propose)(instance, *args, **kwargs)
        finally:
            active = None

    original_publish = publication_class.publish

    def publish(instance, *args, **kwargs):
        nonlocal active
        previous = active
        active = dict(position=instance.position, trace=None)
        try:
            return timed('publication_total', original_publish)(instance, *args, **kwargs)
        finally:
            active = previous

    with ExitStack() as stack:
        for name in ('update', 'snapshot', 'read_tokens'):
            stack.enter_context(patch.object(prepared, name, timed(name, getattr(prepared, name))))
        stack.enter_context(patch.object(module, 'execute', timed('eager_reference', module.execute)))
        stack.enter_context(patch.object(prepared, 'propose', propose))
        stack.enter_context(patch.object(publication_class, 'publish', publish))
        yield


@contextmanager
def observe_boundaries(history_class, target_class, emit, *, clock=perf_counter, cpu_clock=process_time):
    sequence = 0

    def timed(name, operation):
        def invoke(*args, **kwargs):
            nonlocal sequence
            sequence += 1
            identifier = sequence
            metadata = dict(event='audit_boundary', phase=name, identifier=identifier,
                performance_qualified=False, scope='Nested host intervals; CPU time is process-wide')
            emit(dict(metadata, status='started', elapsed_ms=None, cpu_ms=None))
            started, cpu_started = clock(), cpu_clock()
            passed = False
            try:
                result = operation(*args, **kwargs)
                passed = True
                return result
            finally:
                emit(dict(metadata, status='completed' if passed else 'failed',
                    elapsed_ms=(clock() - started) * 1000,
                    cpu_ms=(cpu_clock() - cpu_started) * 1000))
        return invoke

    original_init = target_class.__init__

    def initialize(instance, drafter, snapshot, protected_snapshot=None):
        return original_init(instance, drafter, timed('target_state_snapshot', snapshot),
            protected_snapshot=timed('protected_verifier_snapshot', protected_snapshot)
                if protected_snapshot is not None else None)

    with ExitStack() as stack:
        for name in ('snapshot', 'compare'):
            stack.enter_context(patch.object(history_class, name,
                timed('history_' + name, getattr(history_class, name))))
        stack.enter_context(patch.object(target_class, '__init__', initialize))
        yield
