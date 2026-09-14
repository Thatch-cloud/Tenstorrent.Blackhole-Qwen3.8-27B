"""Opt-in host timing for a prepared proposal; not an acceptance benchmark."""

from contextlib import ExitStack, contextmanager
from time import perf_counter
from unittest.mock import patch


class ProposalProbeComplete(Exception):
    def __init__(self, evidence):
        super().__init__('Bounded proposal diagnostic complete; no full request measured')
        self.evidence = evidence


@contextmanager
def stop_after_prepared_probe(checkpoint):
    from dspark_prepared_proposal import TracedDSparkDevice

    original = TracedDSparkDevice.prepare_trace

    def prepare(device, anchor, *, audit=False):
        original(device, anchor, audit=False)
        evidence = probe_three_replays(device, anchor, checkpoint)
        raise ProposalProbeComplete(evidence)

    with patch.object(TracedDSparkDevice, 'prepare_trace', prepare):
        yield


def probe_three_replays(device, anchor, checkpoint, *, clock=perf_counter):
    if not callable(checkpoint):
        raise ValueError('Immediate evidence checkpoint callback required')
    if device.max_drafts != 15 or device.history.pending is not None:
        raise ValueError('Idle fifteen-query prepared drafter required')
    report = dict(passed=False, performance_qualified=False, committed_tg=None,
        scope='Three fixed-input proposal replays only; no tokens committed or full-request acceptance',
        records=[], completed_replays=0, phase='starting')
    checkpoint(report)
    position = device.position
    expected = tuple(device.prepared.expected_warmup)
    try:
        with profile_proposals(device, report['records'], clock=clock):
            for ordinal in range(3):
                report.update(phase='replay', ordinal=ordinal)
                checkpoint(report)
                tokens = tuple(device.prepared.propose(anchor, 15))
                if tokens != expected or len(tokens) != 15:
                    raise AssertionError('Fixed-input replay differs from eager warmup tokens')
                if device.position != position or device.history.pending is not None:
                    raise AssertionError('Diagnostic replay changed publication frontier')
                report['completed_replays'] += 1
                checkpoint(report)
        report.update(passed=True, phase='complete')
    except BaseException as error:
        report.update(phase='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        checkpoint(report)
    return report


@contextmanager
def profile_proposals(device, records, *, clock=perf_counter):
    prepared = device.prepared
    if prepared is None or prepared.trace is None or prepared.closed or prepared.audit:
        raise ValueError('Live captured proposal with audit disabled required for phase attribution')
    if not isinstance(records, list):
        raise ValueError('Explicit phase record list required')
    operations = device.operations
    active = None

    def timed(name, operation):
        def invoke(*args, **kwargs):
            if active is None:
                return operation(*args, **kwargs)
            started = clock()
            try:
                return operation(*args, **kwargs)
            finally:
                active[name] += (clock() - started) * 1000
        return invoke

    original_propose = prepared.propose

    def propose(anchor, count):
        nonlocal active
        if active is not None:
            raise ValueError('Nested proposal profiling is unsupported')
        active = dict(update_inputs_history_ms=0., trace_replay_ms=0., token_readback_ms=0.)
        started = clock()
        passed = False
        position = device.position
        try:
            result = original_propose(anchor, count)
            passed = True
            return result
        finally:
            elapsed = (clock() - started) * 1000
            record, active = active, None
            record.update(position=position, proposals=count, passed=passed,
                total_ms=elapsed, other_host_ms=elapsed - sum(record.values()),
                scope='Instrumented host timings; replay includes its blocking wait; not device-kernel timing')
            records.append(record)

    with ExitStack() as stack:
        stack.enter_context(patch.object(prepared, 'update', timed('update_inputs_history_ms', prepared.update)))
        stack.enter_context(patch.object(prepared, 'read_tokens', timed('token_readback_ms', prepared.read_tokens)))
        stack.enter_context(patch.object(operations, 'execute_trace', timed('trace_replay_ms', operations.execute_trace)))
        stack.enter_context(patch.object(prepared, 'propose', propose))
        yield records
