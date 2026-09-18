"""Bounded host-wall attribution for an owned captured proposal, without added fences."""

from contextlib import ExitStack, contextmanager
from math import isfinite
from time import perf_counter
from unittest.mock import patch


@contextmanager
def measure_proposals(capture, *, limit=32, clock=perf_counter):
    device = capture.device
    if (type(limit) is not int or not 1 <= limit <= 64 or capture.closed
            or device.progress is not None or getattr(capture, '_host_attribution_active', False)):
        raise ValueError('Live unaudited capture and bounded exclusive diagnostic scope required')
    records = []
    original_propose, original_update = capture.propose, capture.update
    original_select = device.select_proposal
    active = None

    def timed(name, operation, *args, **kwargs):
        if active is None:
            return operation(*args, **kwargs)
        started = clock()
        result = operation(*args, **kwargs)
        active[name].append((clock() - started) * 1000)
        return result

    def update(*args, **kwargs):
        return timed('prepare_ms', original_update, *args, **kwargs)

    def select(*args, **kwargs):
        return timed('readback_merge_select_ms', original_select, *args, **kwargs)

    def propose(*args, **kwargs):
        nonlocal active
        if active is not None:
            raise ValueError('Nested proposal measurement is unsupported')
        if len(records) >= limit:
            return original_propose(*args, **kwargs)
        if device.progress is not None:
            raise ValueError('Audit execution cannot be reported as steady proposal timing')
        active = dict(prepare_ms=[], readback_merge_select_ms=[])
        started = clock()
        try:
            result = original_propose(*args, **kwargs)
            total = (clock() - started) * 1000
            if any(len(samples) != 1 for samples in active.values()):
                raise ValueError('Exactly one preparation and selection per captured proposal required')
            record = {name: samples[0] for name, samples in active.items()}
            remainder = total - sum(record.values())
            intervals = (total, remainder, *record.values())
            if any(not isfinite(value) or value < 0 for value in intervals):
                raise ValueError('Monotonic non-overlapping timing intervals required')
            records.append(dict(record, proposal_ms=total,
                replay_and_bookkeeping_ms=remainder, diagnostic_only=True,
                device_kernel_timing=False, position=device.position))
            return result
        finally:
            active = None

    with ExitStack() as stack:
        stack.enter_context(patch.object(capture, '_host_attribution_active', True, create=True))
        stack.enter_context(patch.object(capture, 'update', update))
        stack.enter_context(patch.object(device, 'select_proposal', select))
        stack.enter_context(patch.object(capture, 'propose', propose))
        yield records
