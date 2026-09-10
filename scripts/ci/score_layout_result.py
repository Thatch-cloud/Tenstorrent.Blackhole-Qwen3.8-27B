"""Independently validate revision-bound score-layout CI artifacts and optional coding cases."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

EXTRA_SOURCES = frozenset(('ccl-links-build.sh', 'dspark-backbone-cpu-reference.json',
    'dspark-backbone-upstream-reference.json', 'dspark-hardware-suite.sh', 'run-dspark-hardware.sh'))

def validate_sources(report, snapshot):
    expected = {}
    with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.parent == PurePosixPath('scripts/ci') and (
                    path.suffix in ('.py', '.cpp', '.h', '.hpp') or path.name in EXTRA_SOURCES):
                key = path.name
            elif path.parent == PurePosixPath('speculative-decoding/harness') and path.suffix == '.py':
                key = '../../' + str(path)
            else:
                continue
            if not member.isfile() or key in expected:
                raise ValueError('Regular unique source files required in revision snapshot')
            expected[key] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if (not expected or report.get('sources') != expected or report.get('sources_after') != expected
            or not report.get('native_sources') or report.get('native_sources') != report.get('native_sources_after')):
        raise ValueError('Complete revision-bound source set and unchanged native sources required')
    return len(expected)


def request_diagnostics(requests):
    from request_host_health import summarize

    result = []
    for ordinal, request in enumerate(requests):
        health = request.get('host_health')
        if health is not None and health != summarize(health['before'], health['after']):
            raise ValueError('Recorded host diagnostics must match independent counter recomputation')
        result.append(dict(ordinal=ordinal, arm=request['arm'], audit=request['instrumented_timing'],
            committed_tokens=request['committed_decode_tokens'], decode_ms=request['decode_ms'],
            prefill_ms=request['prefill_ms'], host_health=health))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--git-dir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--functional', action='store_true')
    options = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', options.revision) or not options.run.isdigit():
        raise ValueError('Immutable full commit SHA and numeric CI run required')
    report = json.loads(options.report.read_text())
    if (report.get('source_revision') != options.revision or str(report.get('workflow_run')) != options.run
            or report.get('passed') is not True or report.get('closed_cleanly') is not True
            or options.report.with_suffix('.exit-status').read_text().strip() != '0'):
        raise ValueError('Passed closed artifact and exit status from the declared revision/run required')
    snapshot = subprocess.check_output(['git', '--git-dir=' + str(options.git_dir), 'archive',
        options.revision, 'scripts/ci', 'speculative-decoding/harness'])
    count = validate_sources(report, snapshot)
    from dspark_score_layout_hardware_gate import validate_hardware
    from dspark_score_layout_variants import summarize_variants
    audit = validate_hardware(report['score_layout_hardware_audit'], Path(__file__).parent)
    comparison = summarize_variants(report['request_checks'])
    if comparison != report.get('request_comparison'):
        raise ValueError('Independent complete request recomputation must match the reported comparison')
    result = dict(passed=True, run=options.run, revision=options.revision, source_files=count,
        report_sha256=hashlib.sha256(options.report.read_bytes()).hexdigest(),
        hardware_audit_sha256=audit, comparison=comparison, held_out_quality_certified=False,
        request_diagnostics=request_diagnostics(report['request_checks']))
    if options.functional:
        from coding_functional_eval import evaluate
        from coding_holdout_tasks import EXPECTED
        coding = report.get('coding_output', {})
        task = coding.get('task')
        if (task not in EXPECTED or coding.get('task_sha256') != EXPECTED[task]
                or coding.get('terminated') is not True or report['coding_context'].get('task_sha256') != EXPECTED[task]):
            raise ValueError('Frozen task identity and EOS-terminated generated output required')
        result['functional'] = evaluate(task, coding['text'])
        result['passed'] = result['functional']['passed']
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
