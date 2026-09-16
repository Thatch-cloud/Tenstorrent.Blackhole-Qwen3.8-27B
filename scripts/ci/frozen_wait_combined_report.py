"""Bind complete MLP wait scopes to validated full-request verifier replays."""

import hashlib
import json
from pathlib import Path
import sys

from frozen_wait_zone_report import read_raw_trace_events, summarize


def expected_executions(events, attribution):
    if attribution.get('passed') is not True or attribution.get('context') != 32768 or attribution.get('streams') != 1:
        raise ValueError('Validated single-stream 32K verifier attribution required')
    full = [entry for entry in attribution['devices'] if entry['steady_replays'] >= 2
        and all(replay['rows'] == 16 for replay in entry['replays'])]
    traces = {entry['trace_id'] for entry in full}
    if not traces or len(full) != 2 * len(traces):
        raise ValueError('Complete full-row verifier traces on both chips required')
    expected = set()
    for trace in traces:
        entries = [entry for entry in full if entry['trace_id'] == trace]
        if {int(entry['device']) for entry in entries} != {0, 1}:
            raise ValueError('Both chips required')
        if entries[0]['replays'] != entries[1]['replays']:
            first = [replay['replay_session'] for replay in entries[0]['replays']]
            second = [replay['replay_session'] for replay in entries[1]['replays']]
            if first != second:
                raise ValueError('Matching actual replay sessions required')
        for entry in entries:
            chip = int(entry['device'])
            programs = None
            sessions = {replay['replay_session'] for replay in entry['replays']}
            selected = [event for event in events if event['trace_id'] == trace and event['chip'] == chip]
            if {event['replay'] for event in selected} != sessions:
                raise ValueError('Raw scopes must cover exactly the validated request sessions')
            for replay in sessions:
                current = {event['host_id'] for event in selected if event['replay'] == replay}
                if len(current) != 64 or (programs is not None and current != programs):
                    raise ValueError('All 64 MLP programs must appear in every replay')
                programs = current
                expected.update((chip, program, trace, replay) for program in programs)
    return expected


def main(root):
    root = Path(root)
    result = dict(passed=False, diagnostic_only=True, committed_tg=None)
    try:
        attribution = json.loads((root / 'attribution.json').read_text())
        request = root / 'request.json'
        if hashlib.sha256(request.read_bytes()).hexdigest() != attribution.get('request_sha256'):
            raise ValueError('Attribution must bind the actual audited request')
        raw = root / 'metadata/profile_log_device.csv'
        events = list(read_raw_trace_events(raw))
        expected = expected_executions(events, attribution)
        result = summarize(events, expected, (root / 'console.log').read_text(errors='replace'))
        result.update(context=32768, streams=1, executions=len(expected),
            request_sha256=attribution['request_sha256'], raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            performance_qualified=False, serving_qualified=False)
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (root / 'wait-attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, executions=len(expected), samples=len(result['samples']), committed_tg=None)))


if __name__ == '__main__':
    main(sys.argv[1])
