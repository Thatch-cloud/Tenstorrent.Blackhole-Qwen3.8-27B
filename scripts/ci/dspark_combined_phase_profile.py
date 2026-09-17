"""Bounded draft-phase attribution inside two complete qualified 64K requests."""

from contextlib import contextmanager
import json
import math
from time import perf_counter
from unittest.mock import patch

import dspark_center_fill_timed as timing
from dspark_proposal_phase_profile import profile_proposals


@contextmanager
def profile_request_scope(records, *, clock=perf_counter):
    from dspark_prepared_proposal import TracedDSparkDevice

    original = TracedDSparkDevice.propose
    devices = []
    counts = []

    def propose(device, anchor, count):
        if device.prepared is None or device.prepared.audit:
            return original(device, anchor, count)
        if not any(device is item for item in devices):
            devices.append(device)
            counts.append(0)
        ordinal = next(index for index, item in enumerate(devices) if item is device)
        sample = counts[ordinal]
        if sample >= 3 or ordinal >= 2:
            return original(device, anchor, count)
        counts[ordinal] += 1
        local = []
        try:
            with profile_proposals(device, local, clock=clock, fence_updates=sample == 2):
                return original(device, anchor, count)
        finally:
            for record in local:
                record.update(request_ordinal=ordinal, sample=sample)
                records.append(record)
                print(json.dumps(dict(stage='combined-draft-phase', **record)), flush=True)

    with patch.object(TracedDSparkDevice, 'propose', propose):
        yield


def attach_phases(result, records):
    if (len(records) != 6 or any(record.get('passed') is not True for record in records)
            or [(record['request_ordinal'], record['sample'], record['updates_fenced']) for record in records]
            != [(ordinal, sample, sample == 2) for ordinal in range(2) for sample in range(3)]):
        raise ValueError('Three bounded phase samples from each complete request required')
    for record in records:
        for name in ('update_inputs_history_ms', 'trace_replay_ms', 'token_readback_ms', 'total_ms'):
            value = record.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative phase timings required')
        if record['total_ms'] <= 0:
            raise ValueError('Positive complete proposal timing required')
    for arm in result['arms'].values():
        arm['instrumented_committed_tg'] = arm.pop('committed_tg')
        arm['instrumented_pp'] = arm.pop('pp')
        arm.update(pp=None, committed_tg=None)
    result.update(draft_phase_records=records, instrumented_phase_profile=True,
        performance_qualified=False, qualification_scope=__doc__)
    return result


@contextmanager
def timed_scope(directory):
    records = []
    original = timing.summarize_timed

    def summarize(requests, audited):
        return attach_phases(original(requests, audited), records)

    with profile_request_scope(records), patch.object(timing, 'summarize_timed', summarize), timing.timed_scope(directory):
        yield
