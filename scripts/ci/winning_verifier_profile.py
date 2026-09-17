"""Profile one audited request after staging the unchanged winning context recipe."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


SOURCE_FILES = ('dspark_request_experiment.py', 'full_dspark_request.py',
    'request_verifier_profile_report.py', 'dspark-combined-profile.sh',
    'run-dspark-hardware.sh', 'dspark-hardware-suite.sh', 'verifier_engine.py', 'frozen_draft_tail_scope.py')


def adapt(sources):
    from frozen_verifier_profile import adapt_sources
    from frozen_wait_combined_stage import drain_setup

    if set(sources) != set(SOURCE_FILES):
        raise ValueError('Exact winning-recipe host instrumentation source set required')
    result = drain_setup(adapt_sources(sources))
    name = 'dspark_request_experiment.py'
    result[name] = replace_once(result[name],
        "    if profile_verifier or profile_drafter or combined_profile:\n"
        "        raise ValueError('Unprofiled combined ladder requests required')\n"
        "    schedule = (('publication', True), ('publication', False), ('publication', False))",
        "    if profile_verifier or profile_drafter or not combined_profile:\n"
        "        raise ValueError('Explicit combined winning-recipe profile required')\n"
        "    schedule = (('publication', True),)")
    result[name] = replace_once(result[name],
        '    from frozen_ladder_requests import finish\n    finish(report, summarize)',
        '    from winning_verifier_profile import finish_profile\n    finish_profile(report)')
    name = 'frozen_draft_tail_scope.py'
    result[name] = replace_once(result[name],
        "if os.environ.get('QWEN_COMBINED_TRACE_PROFILE', '0') != '0':",
        "if os.environ.get('QWEN_COMBINED_TRACE_PROFILE', '0') != '1':")
    result[name] = replace_once(result[name], 'Unprofiled matched combined requests required',
        'Explicit winning-recipe profiling required')
    name = 'dspark-combined-profile.sh'
    result[name] = replace_once(result[name],
        'python3 /experiment-scripts/ci/request_verifier_profile_report.py "$output" dspark',
        'test -s "$output/metadata/tracy_ops_data.csv"\n'
        'test -s "$output/metadata/cpp_device_perf_report.csv"')
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def finish_profile(report):
    from frozen_ladder_requests import validate_audit

    requests = report.get('request_checks', [])
    if len(requests) != 1:
        raise ValueError('Exactly one complete audited profiling request required')
    request = requests[0]
    validate_audit(request)
    if (request.get('arm') != 'publication' or not request.get('verifier_profile')
            or request.get('length') != 4096
            or any(request.get(name, {}).get('enabled') is not True
                for name in ('incremental_history', 'gdn_norm_prefetch', 'draft_tail'))
            or request.get('fused_t16_mlp', {}).get('restored') is not True):
        raise ValueError('Executed winning recipe and verifier attribution required')
    report.update(pp=None, committed_tg=None, performance_qualified=False,
        eligible_for_serving=False, combined_runtime_profile=True, fresh_context_audit=True,
        scope='One complete winning-recipe request under profiling; not throughput evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh profiling manifest required')
    directory = options.checkout / 'scripts' / 'ci'
    sources = {name: (directory / name).read_text() for name in SOURCE_FILES}
    result = adapt(sources)
    for name, source in result.items():
        (directory / name).write_text(source)
    helpers = ('winning_verifier_profile.py',)
    for name in helpers:
        (directory / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
        after={name: hashlib.sha256(value.encode()).hexdigest() for name, value in result.items()},
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
