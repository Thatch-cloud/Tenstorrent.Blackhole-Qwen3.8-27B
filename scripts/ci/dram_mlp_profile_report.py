"""Validate T16 DRAM MLP device attribution, never throughput or promotion."""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from dram_mlp_gate import qualify, variant_sources
from tensix_mlp_profile_report import attribute
from tensix_mlp_weight_views import qualify_views
from tensix_stream_gate import matrix


def profile_sources(root):
    names = ('dram_mlp_profile_report.py', 'dram-mlp-profile.sh', 'tensix_mlp_profile.py',
        'tensix_mlp_profile_report.py', 'request_verifier_profile.py', 'request_verifier_profile_report.py',
        'tensix_mlp_weight_views.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def validate_profile(report, console):
    if (any(report.get(name) is not True for name in
                ('passed', 'closed_cleanly', 'instrumented_timing', 'correctness_only', 'sharded_product'))
            or report.get('stage') != 'complete' or report.get('backend') != 'hardware'
            or report.get('error') or report.get('eligible_for_full_model_gate') is not False
            or report.get('blocks') != []
            or any(name in report for name in ('control_ms', 'candidate_ms', 'tg_tokens_per_second'))
            or any(type(report.get(name)) is not int or report[name] != value for name, value in
                (('rows', 16), ('streams', 1), ('layer', 0), ('collective_links', 4), ('repeats_per_sample', 1)))
            or report.get('seeds') != [1659, 2670, 3781]
            or report.get('sources') != report.get('sources_after')
            or report.get('native_sources') != report.get('native_sources_after')):
        raise ValueError('Clean T16 attribution only, with unchanged sources and no promotion, required')
    qualify_views(report.get('weight_views'))
    if report.get('native_weights_after') != {name: check['native'] for name, check in report['weight_views'].items()}:
        raise ValueError('Native weight metadata and buffers must remain unchanged')
    matrix(report.get('eager_checks'), ('pattern', 'chip'),
        {(pattern, chip) for pattern in range(3) for chip in range(2)}, ('exact',))
    matrix(report.get('trace_checks'), ('pattern', 'arm', 'chip'),
        {(pattern, arm, chip) for pattern in range(3) for arm in range(2) for chip in range(2)}, ('exact',))
    matrix(report.get('profile_checks'), ('pattern', 'sample', 'chip'),
        {(pattern, sample, chip) for pattern in range(3) for sample in range(4) for chip in range(2)}, ('exact',))
    matrix(report.get('input_checks'), ('pattern',), {(pattern,) for pattern in range(3)},
        ('unchanged', 'bindings_stable'))
    profile = report.get('profile', {})
    traces, records = profile.get('trace_ids'), profile.get('records')
    if (not isinstance(traces, list) or len(traces) != 2
            or any(type(trace) is not int or trace < 0 for trace in traces) or len(set(traces)) != 2
            or profile.get('trace_counts') != [9, 9] or not isinstance(records, list) or len(records) != 18):
        raise ValueError('All eighteen replays of two distinct traces required')
    expected = []
    for pattern in range(3):
        expected.extend((arm, pattern, 'audit', -1) for arm in range(2))
        expected.extend((arm, pattern, 'measurement', sample) for sample, arm in enumerate((0, 1, 1, 0)))
    counts, previous_end = [0, 0], -1
    for index, (record, (arm, pattern, role, sample)) in enumerate(zip(records, expected, strict=True)):
        integers = dict(block=index, arm=arm, pattern=pattern, sample=sample, rows=16,
            trace_id=traces[arm], trace_ordinal=counts[arm])
        duration = record.get('instrumented_host_ms')
        if (any(type(record.get(name)) is not int or record[name] != value for name, value in integers.items())
                or record.get('role') != role or record.get('first_replay') is not (counts[arm] == 0)
                or type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0):
            raise ValueError('Ordered complete replay metadata required')
        label = f'qwen_mlp_arm{arm}_pattern{pattern}_{role}_{sample}_ordinal{counts[arm]}'
        begin, end = 'QWEN_MLP_PROFILE_BEGIN ' + label, 'QWEN_MLP_PROFILE_END ' + label
        if (record.get('label') != label or console.count(begin) != 1 or console.count(end) != 1
                or not previous_end < console.index(begin) < console.index(end)
                or 'markers were dropped' in console.split(begin, 1)[1].split(end, 1)[0]):
            raise ValueError('Ordered markers without dropped device events required')
        previous_end = console.index(end)
        counts[arm] += 1
    return records


def validate_sources(report, root):
    simulator_path = root / 'dram-mlp-sharded-simulator.json'
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in variant_sources(True)}
    if (report.get('sources') != sources
            or report.get('hardware_script_sha256') != hashlib.sha256((root / 'dram-mlp-hardware.py').read_bytes()).hexdigest()
            or report.get('simulator_report_sha256') != hashlib.sha256(simulator_path.read_bytes()).hexdigest()
            or report.get('profile_sources') != profile_sources(root)
            or report.get('profile_sources_after') != profile_sources(root)):
        raise ValueError('Matching simulator, harness and profiler fingerprints required')
    qualify(json.loads(simulator_path.read_text()), sources, report['native_sources'],
        (root / 'dram-mlp-sharded-simulator.exit-status').read_text().strip(), hardware=True, sharded=True)


def main(root):
    root = Path(root)
    result = dict(passed=False, scope=__doc__)
    try:
        report_path, device_path = root / 'mlp.json', root / 'metadata/cpp_device_perf_report.csv'
        report = json.loads(report_path.read_text())
        validate_sources(report, Path(__file__).parent)
        records = validate_profile(report, (root / 'console.log').read_text(errors='replace'))
        with device_path.open(newline='') as stream:
            devices = attribute(records, list(csv.DictReader(stream)), full_rows=16)
        for device in devices:
            if device['arm'] == 'streamed':
                device['arm'] = 'dram_sharded'
        result.update(passed=True, devices=devices,
            mlp_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            device_report_sha256=hashlib.sha256(device_path.read_bytes()).hexdigest(),
            caveats=['Instrumented durations are not TG or a performance qualification.',
                'Operation and RISC times include waits and overlap; sums are not a critical path.'])
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (root / 'attribution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, chips_and_arms=len(devices), scope=__doc__)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
