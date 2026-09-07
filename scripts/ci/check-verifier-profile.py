"""Validate both-chip T8 replay attribution; summed durations are not critical-path latency."""

import csv
import importlib.util
import json
from pathlib import Path
import sys


def validate_correctness(report):
    if (report.get('passed') is not True or report.get('correctness_only') is not True
            or report.get('instrumented_timing') or report.get('timings')):
        raise AssertionError('Separate passed uninstrumented correctness matrix required')
    for key, dimension, values in (('batch_checks', 'rows', (1, 2, 4, 8, 16, 32)),
                                   ('checks', 'prefix', (0, 1, 16, 32))):
        checks = report.get(key, [])
        expected = {(length, trace, value) for length in (4095, 16383) for trace in (False, True) for value in values}
        actual = {(check['length'], check['trace'], check[dimension]) for check in checks}
        if actual != expected or len(checks) != len(expected) or not all(all(check.get(field) is True
                for field in ('logits_exact', 'all_gdn_states_exact', 'valid_kv_exact')) for check in checks):
            raise AssertionError('Incomplete exact broad correctness matrix')
    negatives = report.get('negative_controls', [])
    if (len(negatives) != 4 or {(item['length'], item['trace']) for item in negatives} != {
            (length, trace) for length in (4095, 16383) for trace in (False, True)}
            or not all(item.get('stale_gdn_detected') is True and item.get('wrong_page_detected') is True for item in negatives)):
        raise AssertionError('Both negative controls required in every context/mode')


def analyze(generation, rows, console, *, contexts=(4095, 16383)):
    if generation.get('passed') is not True or generation.get('instrumented_timing') is not True:
        raise AssertionError('Passed instrumented verifier correctness report required')
    timings = generation.get('timings', [])
    if tuple(contexts) not in ((4095, 16383), (4095,), (16383,)):
        raise AssertionError('Audited profiling contexts required')
    if sorted(timing['length'] for timing in timings) != list(contexts):
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


def read_profile(root, generation_path, contexts):
    generation = json.loads(generation_path.read_text())
    selected = {str(record['trace_id']) for timing in generation['timings']
                for record in timing['device_profile']['records']}
    reports = list(root.rglob('*ops_perf_results*.csv'))
    direct_device_report = False
    if not reports:
        reports = [directory / 'cpp_device_perf_report.csv' for directory in (root / '.logs', root / 'metadata', root)
                   if (directory / 'cpp_device_perf_report.csv').is_file()][:1]
        direct_device_report = True
    if len(reports) != 1:
        raise AssertionError('Exactly one operation report required')
    with reports[0].open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row.get('METAL TRACE ID') in selected]
    if direct_device_report:
        for row in rows:
            row['OP CODE'] = row['OP NAME']
    result = analyze(generation, rows, (root / 'console.log').read_text(encoding='utf-8', errors='replace'), contexts=contexts)
    result['source_format'] = 'runtime C++ device operation report' if direct_device_report else 'merged operation report'
    return result


def main(root, generation_path=None):
    validate_correctness(json.loads((root / 'correctness.json').read_text()))
    if generation_path is not None:
        result = read_profile(root, generation_path, (4095, 16383))
    else:
        result = dict(passed=True, scope=__doc__, processes=[])
        for context in (4095, 16383):
            directory = root / f'context-{context}'
            profile = read_profile(directory, directory / 'generation.json', (context,))
            result['processes'].append(dict(context=context, attribution=profile))
    (root / 'attribution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
