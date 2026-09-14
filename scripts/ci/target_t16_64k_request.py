"""Explicit hardware-qualified folded-verifier request experiment; no serving defaults."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from target_t16_64k_geometry import geometry_scope, validate_ticket


RUN_ID = 34828634864
REPORT_SHA256 = 'b7991eb7d6c9ffa31eb0c6922456f891419245c5d98943baf665c248df6ac2f9'


def qualify(directory):
    payload = (Path(directory) / 'target-t16-attention-64k-hardware.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed 64K folded-verifier hardware evidence required')
    report = json.loads(payload)
    if (report.get('passed') is not True or report.get('closed') is not True
            or report.get('backend') != 'hardware'
            or report.get('stale_controls') != 2 or report.get('mask_poison_controls') != 8
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete closed hardware controls required')
    for field, count in (('checks', 8), ('mask_checks', 16), ('source_checks', 4), ('unpoisoned_replay', 2)):
        values = report.get(field, [])
        if len(values) != count or any(value.get('exact') is not True for value in values):
            raise ValueError('Every folded attention comparison must be exact')
    if any(value.get('capacity') != 65792 for field in ('checks', 'mask_checks', 'source_checks')
            for value in report[field]):
        raise ValueError('Full 64K hardware capacity required')
    for name, checksum in report['sources'].items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Hardware-qualified verifier dependency changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256, hardware_qualified=True,
        full_request_qualified=False, performance_qualified=False)


def validate_request_option(enabled, *, rows, position, remaining, replay, norm_batch,
                            native_sampling, group_rows, short_context):
    if enabled is False:
        return
    if (enabled is not True or any(type(value) is not int for value in (rows, position, remaining, group_rows))
            or rows != 16 or position != 65536 or not 1 <= remaining <= 256
            or replay is not True or norm_batch is not True or native_sampling is not True
            or group_rows != 4 or short_context is not False):
        raise ValueError('Explicit 64K T16 request with native sampling and bounded output required')


@contextmanager
def request_scope(directory):
    import attention_request_plan
    import dspark_64k_variants
    import target_t16_attention_gate

    admission = qualify(directory)
    policies = {name: dict(policy) for name, policy in dspark_64k_variants.POLICIES.items()}
    policies['scatter']['target_attention_t16'] = True

    def validate(start, rows, capacity, *, short_context=False):
        return validate_ticket(start, rows, capacity, short_context=short_context, hardware=True)

    with geometry_scope(hardware=True), \
            patch.object(attention_request_plan, 'validate_ticket', validate), \
            patch.object(dspark_64k_variants, 'POLICIES', policies), \
            patch.object(target_t16_attention_gate, 'validate_request_option', validate_request_option), \
            patch.object(target_t16_attention_gate, 'qualify', qualify):
        yield admission
