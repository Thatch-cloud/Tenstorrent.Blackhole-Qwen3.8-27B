"""Separate clean source-bound request audits from fresh ABBA measurements."""

import hashlib
import json
from pathlib import Path

from dspark_request_variants import proposal_signature


SCHEDULE = (('control', True), ('publication', True), ('control', False),
    ('publication', False), ('publication', False), ('control', False))
IDENTITY = ('sources', 'native_sources', 'target_sources', 'target_index_sha256',
    'target_config_sha256', 'parameter_sha256', 'ctx_tokens', 'streams',
    'target_cache_formats')


def validate_audits(requests, route):
    if [(value.get('arm'), value.get('instrumented_timing')) for value in requests] != list(SCHEDULE[:2]):
        raise ValueError('Two complete ordered feature audits required')
    for value in requests:
        if (any(value.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact'))
                or value.get('length') != 32768 or len(value.get('prompt_tokens', [])) != 32768
                or value.get('prompt_tokens') != requests[0].get('prompt_tokens')
                or not value.get('emitted') or value['emitted'] != requests[0]['emitted']
                or value.get('committed_decode_tokens') != len(value['emitted']) - 1):
            raise ValueError('Matched full-context exact output and state audits required')
        draft = value.get('dspark', {})
        if (draft.get('native_attention') is not True or draft.get('proposal_trace') is not True
                or value.get('commit_only_gdn') is not True):
            raise ValueError('Unchanged native traced request policies required')
        route(value, value['arm'])
        blocks = [block for block in value['blocks'] if block['rows'] > 1]
        checks = draft.get('proposal_checks', [])
        if (not blocks or [check.get('position') for check in checks]
                != [value['length'], *(block['position'] for block in blocks)]
                or any(check.get('exact') is not True or check.get('tensors') != 6 for check in checks)):
            raise ValueError('Every eager versus trace proposal check required')
        if value.get('gdn_verify_checks') != [dict(position=block['position'], rows=block['rows'], unchanged=True)
                for block in blocks]:
            raise ValueError('Every deferred-publication state check required')
    if proposal_signature(requests[0]) != proposal_signature(requests[1]):
        raise ValueError('Assembly must preserve every proposal and acceptance')


def qualify(retained, current, route):
    if (any(retained.get(name) is not True for name in ('passed', 'closed_cleanly', 'checkpoint_closed'))
            or retained.get('stage') != 'complete' or retained.get('request_phase') != 'audit'
            or retained.get('correctness_only') is not True or retained.get('pp') is not None
            or retained.get('committed_tg') is not None):
        raise ValueError('Clean completed correctness-only qualification required')
    for name in IDENTITY:
        if retained.get(name) in (None, {}, []) or retained[name] != current.get(name):
            raise ValueError('Qualification identity changed: ' + name)
    for name in ('sources', 'native_sources'):
        if retained[name] != retained.get(name + '_after'):
            raise ValueError('Final source immutability required: ' + name)
    validate_audits(retained.get('request_checks', []), route)
    return retained['request_checks']


def prepare(directory, report, schedule):
    path = Path(directory) / 'frozen-request-phase.json'
    if not path.exists():
        return dict(mode='combined', schedule=schedule)
    if tuple(schedule) != SCHEDULE:
        raise ValueError('Only the admitted matched 32K request schedule may be split')
    configuration = json.loads(path.read_text())
    mode = configuration.get('mode')
    if mode == 'audit' and set(configuration) == {'mode'}:
        result = dict(mode=mode, schedule=SCHEDULE[:2])
    elif mode == 'timed' and set(configuration) == {'mode', 'qualification_sha256'}:
        payload = (Path(directory) / 'frozen-request-qualification.json').read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != configuration['qualification_sha256']:
            raise ValueError('Pinned qualification digest required')
        from gdn_shared_qk_variants import validate_route
        retained = json.loads(payload)
        result = dict(mode=mode, schedule=SCHEDULE[2:],
            audits=qualify(retained, report, validate_route), report_sha256=checksum,
            workflow_run=retained.get('workflow_run'))
    else:
        raise ValueError('Explicit audit or source-pinned timed phase required')
    report['request_phase'] = mode
    return result


def finish(phase, report):
    mode = phase['mode']
    if mode == 'combined':
        return False
    from gdn_shared_qk_variants import validate_route
    if mode == 'audit':
        validate_audits(report['request_checks'], validate_route)
        report.update(correctness_only=True, instrumented_timing=True, pp=None, committed_tg=None,
            scope='Two fresh full-request feature audits; no throughput measurement')
        return True
    if mode != 'timed' or [(value.get('arm'), value.get('instrumented_timing'))
            for value in report['request_checks']] != list(SCHEDULE[2:]):
        raise ValueError('Four fresh uninstrumented A/B/B/A requests required')
    report['retained_audit'] = dict(report_sha256=phase['report_sha256'],
        workflow_run=phase['workflow_run'], requests=2, fresh_timed_requests=4)
    report['request_checks'] = phase['audits'] + report['request_checks']
    report['scope'] = 'Four fresh ABBA requests with two source-bound retained feature audits'
    return False
