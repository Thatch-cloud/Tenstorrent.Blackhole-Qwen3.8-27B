"""Validate both-chip T8 replay attribution; summed durations are not critical-path latency."""

import csv
import importlib.util
import json
from pathlib import Path
import sys


def analyze(generation, rows, console):
    if generation.get('passed') is not True or generation.get('instrumented_timing') is not True:
        raise AssertionError('Passed instrumented verifier correctness report required')
    timings = generation.get('timings', [])
    if sorted(timing['length'] for timing in timings) != [4095, 16383]:
        raise AssertionError('Both matched coding contexts required')
    spec = importlib.util.spec_from_file_location('model_profile_check', Path(__file__).with_name('check-model-profile.py'))
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    results = []
    seen = set()
    for timing in timings:
        if timing['rows'] != 8 or timing.get('exact') is not True or timing.get('blocks'):
            raise AssertionError('Exact attribution-only T8 record required')
        records = timing['device_profile']['records']
        if [(record['repeat'], record['arm']) for record in records] != [
                (repeat, arm) for repeat in range(3) for arm in ('serial', 'control', 'batch')]:
            raise AssertionError('Three complete native/control/candidate replay groups required')
        for record in records:
            label = f"qwen_verifier_t8_ctx{timing['length']}_{record['arm']}_{record['repeat']}"
            if record.get('exact') is not True or record['label'] != label:
                raise AssertionError('Missing exact labeled replay')
            begin, end = 'QWEN_VERIFIER_PROFILE_BEGIN ' + label, 'QWEN_VERIFIER_PROFILE_END ' + label
            if console.count(begin) != 1 or console.count(end) != 1 or console.index(end) < console.index(begin):
                raise AssertionError('Missing unambiguous replay boundaries')
            if 'markers were dropped' in console.split(begin, 1)[1].split(end, 1)[0]:
                raise AssertionError('Dropped markers during verifier attribution')
        for arm in ('serial', 'control', 'batch'):
            ids = {record['trace_id'] for record in records if record['arm'] == arm}
            if len(ids) != 1 or seen.intersection(ids):
                raise AssertionError('Distinct stable trace identifiers required')
            seen.update(ids)
            trace_id = next(iter(ids))
            devices = checker.analyze(rows, trace_id, 3)
            if {device['device'] for device in devices} != {'0', '1'}:
                raise AssertionError('Physical chips zero and one required')
            results.append(dict(context=timing['length'], arm=arm, trace_id=trace_id, devices=devices))
    return dict(passed=True, scope=__doc__, traces=results)


def main(root, generation_path):
    generation = json.loads(generation_path.read_text())
    selected = {str(record['trace_id']) for timing in generation['timings']
                for record in timing['device_profile']['records']}
    reports = list(root.rglob('*ops_perf_results*.csv'))
    direct_device_report = False
    if not reports:
        reports = list(root.rglob('cpp_device_perf_report.csv'))
        direct_device_report = True
    if len(reports) != 1:
        raise AssertionError('Exactly one operation report required')
    with reports[0].open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row.get('METAL TRACE ID') in selected]
    if direct_device_report:
        for row in rows:
            row['OP CODE'] = row['OP NAME']
    result = analyze(generation, rows, (root / 'console.log').read_text(encoding='utf-8', errors='replace'))
    result['source_format'] = 'runtime C++ device operation report' if direct_device_report else 'merged operation report'
    (root / 'attribution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
