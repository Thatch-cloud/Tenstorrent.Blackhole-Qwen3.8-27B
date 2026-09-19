"""Source-bound hardware admission for the simulator-qualified diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once
from mlp_clock_report import validate


SIMULATOR_SHA256 = '6eb70a4126095ce96b14c78ff23453e470e3a78199f2dacdc4d2867038f9753e'


def retained(evidence):
    evidence = Path(evidence)
    raw = (evidence / 'fused-batch.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SIMULATOR_SHA256:
        raise ValueError('Exact reviewed simulator report required')
    if (evidence / 'fused-batch.exit-status').read_text().strip() != '0':
        raise ValueError('Clean simulator exit required')
    cleanup = json.loads((evidence / 'container-cleanup.json').read_text())
    if cleanup != dict(stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0):
        raise ValueError('Clean simulator container teardown required')
    if (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9':
        raise ValueError('Pinned simulator runtime required')
    report = json.loads(raw)
    validate(report, backend='simulator')
    return report


def verify_sources(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'clock-hardware-sources.json').read_text())
    for name, expected in manifest.items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Hardware diagnostic source changed: ' + name)


def stage(directory, evidence):
    directory, evidence = Path(directory), Path(evidence)
    retained(evidence)
    manifest = json.loads((evidence / 'clock-candidate.json').read_text())
    for name, expected in manifest['sources'].items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Candidate differs from simulator source: ' + name)
    probe = directory / 'fused-batch-probe.py'
    candidate = replace_once(probe.read_text(),
        'if options.hardware or options.timing or not (options.target_math and options.trace_t16',
        'if not options.hardware or options.timing or not (options.target_math and options.trace_t16')
    candidate = replace_once(candidate,
        'T16 simulator-only buffering qualification requires all correctness gates',
        'T16 hardware diagnostic requires all correctness gates')
    compile(candidate, str(probe), 'exec')
    probe.write_bytes(candidate.encode())
    runner = directory / 'run-wait-zone-hardware.sh'
    runner_source = replace_once(runner.read_text(), '/experiment-scripts/ci/wait-zone-hardware-suite.sh',
        '/experiment-scripts/ci/mlp-clock-hardware-suite.sh')
    (directory / 'run-mlp-clock-hardware.sh').write_bytes(runner_source.encode())
    names = {*manifest['sources'], 'mlp_clock_hardware.py', 'mlp_clock_report.py',
        'run-mlp-clock-hardware.sh', 'mlp-clock-hardware-suite.sh'}
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sorted(names)}
    (directory / 'clock-hardware-sources.json').write_text(json.dumps(hashes, indent=2) + '\n')
    verify_sources(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('stage', 'verify', 'validate'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if options.command == 'stage':
        stage(options.directory, options.evidence)
    elif options.command == 'verify':
        verify_sources(options.directory)
    else:
        verify_sources(options.directory)
        simulator = retained(options.evidence)
        raw = (options.output / 'fused-batch.json').read_bytes()
        if (options.output / 'fused-batch.exit-status').read_text().strip() != '0':
            raise ValueError('Clean hardware process exit required')
        report = json.loads(raw)
        summary = validate(report, backend='hardware')
        if report.get('kernels') != simulator.get('kernels'):
            raise ValueError('Hardware kernel manifest differs from simulation')
        summary.update(simulator_report_sha256=SIMULATOR_SHA256, report_sha256=hashlib.sha256(raw).hexdigest())
        (options.output / 'clock-hardware.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
