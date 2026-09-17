"""Simulator-bound hardware marker smoke test; no full model or throughput claim."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once
from frozen_wait_zone_report import read_raw_trace_events, summarize


REPORT_SHA256 = 'd246e6e3b23fb6a5615cafe261c90dc5e15db195cb0e78f4f6a47641584088fd'
PROJECTION_SHA256 = '4c215d945e723ef3d8c2be1757e7c170bf21eebb01e204359243d766c6c7cf93'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_numerics(report, backend):
    if (report.get('passed') is not True or report.get('backend') != backend
            or report.get('math_approx_mode') is not True or report.get('timings') != []
            or report.get('checks') != [dict(rows=16, chip=chip, exact=True) for chip in (0, 1)]):
        raise ValueError('Exact T16 diagnostic matrix without timing required')
    traces = report.get('trace_replays', [])
    if len(traces) != 1 or traces[0].get('passed') is not True or traces[0].get('rows') != 16:
        raise ValueError('One complete T16 trace matrix required')
    expected = [dict(arm=arm, repetition=index, pattern=pattern, chip=chip, exact=True)
        for index, pattern in enumerate((0, 1, 0)) for arm in ('control', 'fused') for chip in (0, 1)]
    negative = [dict(arm=arm, chip=chip, stale_input_detected=True)
        for arm in ('control', 'fused') for chip in (0, 1)]
    if traces[0].get('checks') != expected or traces[0].get('negative_controls') != negative:
        raise ValueError('Changed-input and stale-input controls required')
    weights = report.get('weight_checks', [])
    if (len(weights) != 4 or {(item.get('projection'), item.get('chip')) for item in weights}
            != {(projection, chip) for projection in ('gate', 'up') for chip in (0, 1)}
            or any(item.get('pages') != 43520 or item.get('mismatched_words') != 0
                or item.get('exact') is not True or item.get('source_exact') is not True for item in weights)):
        raise ValueError('All packed weights must be exact')


def qualify(directory, evidence):
    directory, evidence = Path(directory), Path(evidence)
    path = evidence / 'fused-batch.json'
    if digest(path) != REPORT_SHA256 or (evidence / 'fused-batch.exit-status').read_text().strip() != '0':
        raise ValueError('Exact retained successful simulator run required')
    report = json.loads(path.read_text())
    validate_numerics(report, 'simulator')
    expected = dict(report['kernels'][0]['reader_sha256'])
    expected.update({'fused_1d.py': PROJECTION_SHA256, 'fusion_trace.py': report['trace_source_sha256'],
        'fused-batch-probe.py': '403f07af2901d4867b6560ce43c0d6aac6eb00277b2ed052c0816c9eda957447'})
    expected.update(report['weight_check_sources'])
    for name, checksum in expected.items():
        if digest(directory / name) != checksum:
            raise ValueError('Source differs from simulation: ' + name)
    return report


def adapt_hardware_probe(source):
    source = replace_once(source, 'if options.hardware or options.timing or not (',
        'if not options.hardware or options.timing or not (')
    source = replace_once(source, 'T16 simulator-only buffering qualification requires all correctness gates',
        'T16 allocated hardware marker diagnostic requires all correctness gates')
    compile(source, 'fused-batch-probe.py', 'exec')
    return source


def stage(directory, evidence):
    directory = Path(directory)
    qualify(directory, evidence)
    manifest = directory / 'hardware-marker-sources.json'
    if manifest.exists():
        raise ValueError('Fresh hardware marker staging required')
    probe = adapt_hardware_probe((directory / 'fused-batch-probe.py').read_text())
    trace = replace_once((directory / 'fusion_trace.py').read_text(),
        "    report['passed'] = (len(report['checks']) == 12",
        "    report['profiler_fused_trace_id'] = int(traces['fused'])\n"
        "    report['passed'] = (len(report['checks']) == 12")
    compile(trace, 'fusion_trace.py', 'exec')
    (directory / 'fused-batch-probe.py').write_bytes(probe.encode())
    (directory / 'fusion_trace.py').write_bytes(trace.encode())
    manifest.write_text(json.dumps({path.name: digest(path) for path in directory.iterdir()
        if path.is_file() and path.suffix in ('.py', '.cpp', '.hpp', '.sh')}, indent=2) + '\n')


def verify_sources(directory):
    directory = Path(directory)
    expected = json.loads((directory / 'hardware-marker-sources.json').read_text())
    for name, checksum in expected.items():
        if digest(directory / name) != checksum:
            raise ValueError('Staged hardware source changed: ' + name)


def validate(directory, evidence, output):
    directory, evidence, output = Path(directory), Path(evidence), Path(output)
    verify_sources(directory)
    simulator = json.loads((evidence / 'fused-batch.json').read_text())
    if digest(evidence / 'fused-batch.json') != REPORT_SHA256:
        raise ValueError('Simulator report changed')
    report = json.loads((output / 'fused-batch.json').read_text())
    validate_numerics(report, 'hardware')
    if report['kernels'] != simulator['kernels'] or report['trace_source_sha256'] != digest(directory / 'fusion_trace.py'):
        raise ValueError('Simulator-admitted kernels and recorded replay helper required')
    trace = report['trace_replays'][0].get('profiler_fused_trace_id')
    if type(trace) is not int or trace < 0:
        raise ValueError('Host-recorded fused trace identity required')
    raw = output / 'raw-profiler/profile_log_device.csv'
    events = [event for event in read_raw_trace_events(raw) if event['trace_id'] == trace]
    programs = {chip: {event['host_id'] for event in events if event['chip'] == chip} for chip in (0, 1)}
    if any(len(values) != 1 for values in programs.values()):
        raise ValueError('One fused program on each chip required')
    expected = {(chip, next(iter(programs[chip])), trace, replay) for chip in (0, 1) for replay in range(1, 5)}
    actual = {(event['chip'], event['host_id'], event['trace_id'], event['replay']) for event in events}
    if actual != expected:
        raise ValueError('Exactly four fused replays on both chips required')
    result = summarize(events, expected, (output / 'probe.log').read_text())
    result.update(simulator_report_sha256=REPORT_SHA256, hardware_report_sha256=digest(output / 'fused-batch.json'),
        raw_sha256=digest(raw), numerics_passed=True, serving_qualified=False, performance_qualified=False)
    (output / 'marker-qualification.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'verify-sources', 'validate'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if options.action == 'stage':
        stage(options.directory, options.evidence)
    elif options.action == 'verify-sources':
        verify_sources(options.directory)
    else:
        validate(options.directory, options.evidence, options.output)


if __name__ == '__main__':
    main()
