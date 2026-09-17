"""Device attribution for actual draft replays inside two complete 64K requests."""

from contextlib import contextmanager
import json
from unittest.mock import patch

import dspark_center_fill_timed as timing
from dspark_draft_profile import DraftProfile


@contextmanager
def profile_device_scope(records, *, observer_type=DraftProfile):
    from dspark_prepared_proposal import TracedDSparkDevice

    original = TracedDSparkDevice.propose
    devices = []

    def propose(device, anchor, count):
        prepared = device.prepared
        if prepared is None or prepared.audit or any(device is item for item in devices):
            return original(device, anchor, count)
        if len(devices) >= 2:
            raise ValueError('At most two complete profiled requests required')
        ordinal = len(devices)
        devices.append(device)
        observer = observer_type(device.operations, device.mesh)
        execute_trace = device.operations.execute_trace

        def replay(mesh, trace, *args, **kwargs):
            return observer.replay(execute_trace, mesh, trace, *args, **kwargs)

        with observer.observe(prepared, 'proposal'), patch.object(device.operations, 'execute_trace', replay):
            result = original(device, anchor, count)
        if observer.failed or len(observer.records) != 1:
            raise ValueError('Exactly one actual draft replay must be attributed per request')
        record = dict(observer.records[0], request_ordinal=ordinal)
        records.append(record)
        print(json.dumps(dict(stage='combined-draft-device-profile', **record)), flush=True)
        return result

    with patch.object(TracedDSparkDevice, 'propose', propose):
        yield


def attach_device_profile(result, records):
    if (len(records) != 2 or [record.get('request_ordinal') for record in records] != [0, 1]
            or any(record.get('rows') != 15 or record.get('role') != 'proposal' for record in records)
            or len({record.get('label') for record in records}) != 2):
        raise ValueError('Two distinct actual fifteen-query replay profiles required')
    for arm in result['arms'].values():
        arm.update(instrumented_pp=arm['pp'], instrumented_committed_tg=arm['committed_tg'])
        arm.update(pp=None, committed_tg=None)
    result.update(draft_device_records=records, instrumented_device_profile=True,
        performance_qualified=False, qualification_scope=__doc__)
    return result


@contextmanager
def timed_scope(directory):
    records = []
    original = timing.summarize_timed

    def summarize(requests, audited):
        return attach_device_profile(original(requests, audited), records)

    with profile_device_scope(records), patch.object(timing, 'summarize_timed', summarize), timing.timed_scope(directory):
        yield
