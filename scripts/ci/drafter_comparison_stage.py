"""Overlay the matched drafter driver onto the fully staged five-component recipe."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once

FILES = ('drafter_comparison_experiment.py', 'drafter_comparison_report.py',
         'drafter_request_environment.py', 'drafter_request_metadata.py', 'dflash_combined_request.py', 'full_dflash_request.py',
         'dflash_device.py', 'dflash_request_runtime.py', 'draft_attention_branch.py', 'dflash_proposal_inputs.py',
         'dflash_attention_mask.py', 'draft_shared_head.py', 'cumulative_t16_scope.py', 'dflash-fixtures.sh',
         'dflash_proposal_trace.py', 'dflash_t16_native_scope.py', 'dflash_t16_native_attention.py',
         'dflash_t16_native_attention_gate.py', 'dflash-t16-native-attention-probe.py',
         'dflash_native_comparison_experiment.py', 'dflash_native_comparison_report.py',
         'dflash_combined_sim_runtime.py')


def stage(checkout, manifest, *, native_evidence=None):
    directory, scripts = Path(__file__).parent, Path(checkout) / 'scripts/ci'
    manifest = Path(manifest)
    if manifest.exists():
        raise ValueError('Fresh comparison staging required')
    for name in ('compact-score-evidence', 'mlp-down-grid-evidence', 'register-epilogue-evidence'):
        if not (scripts / name).is_dir():
            raise ValueError('Complete promoted component evidence required: ' + name)
    from dspark_hardware_gate import simulator_preflight
    simulator_preflight(scripts)
    native_payloads = {}
    if native_evidence is not None:
        from dflash_t16_native_scope import REPORTS
        from dflash_t16_native_attention_gate import SOURCES, hashes, qualify

        for context, expected in REPORTS.items():
            name = f'dflash-t16-{context}.json'
            data = (Path(native_evidence) / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Pinned native T16 simulator report required')
            if (Path(native_evidence) / f'dflash-t16-{context}.exit-status').read_text().strip() != '0':
                raise ValueError('Successful outer native T16 simulator exit required')
            report = json.loads(data)
            qualify(report, context, hashes(directory, SOURCES), report['native_sources'])
            native_payloads[name] = data
    originals = {name: (scripts / name).read_text() for name in ('dspark-target-hardware.py', 'run-dspark-hardware.sh')}
    payloads = {name: (directory / name).read_text() for name in FILES}
    payloads['dspark-target-hardware.py'] = replace_once(originals['dspark-target-hardware.py'],
        'from cumulative_t16_experiment import run_loaded_requests',
        'from ' + ('dflash_native_comparison_experiment' if native_evidence is not None
                   else 'drafter_comparison_experiment') + ' import run_loaded_requests')
    shell = replace_once(originals['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_DRAFTER_COMPARISON=1" \\\n    -e "QWEN_DSPARK_MODE=$mode"')
    if native_evidence is not None:
        shell = shell.replace('QWEN_DRAFTER_COMPARISON=1', 'QWEN_DFLASH_NATIVE_COMPARISON=1')
        suite = (scripts / 'dspark-hardware-suite.sh').read_text()
        anchor = '    > /experiment/results/dspark-build-time.json\nset +e'
        payloads['dspark-hardware-suite.sh'] = replace_once(suite, anchor,
            '    > /experiment/results/dspark-build-time.json\n'
            'timeout -k 5 30 python3 -B /experiment-scripts/ci/dflash_t16_native_scope.py '
            '> /experiment/results/dflash-native-preload-admission.json\nset +e')
    fixture_copy = '''source scripts/ci/dflash-fixtures.sh
dflash_cache=/home/thatch/.cache/qwen-experiments
dflash_revision=dedf8df68adfb1afeaf7b7480c0a0243108177b4
dflash_layout="$output/dflash-fixture-layout"
mkdir -p "$dflash_layout"/{attention,convolution,mlp,projection,selector,layer-1,layer-2,layer-3,layer-4}
for component in attention convolution mlp projection selector; do
    test -f "$dflash_cache/dflash2-$component-$dflash_revision/manifest.json"
done
for layer in 1 2 3 4; do
    test -f "$dflash_cache/dflash2-stack-$dflash_revision/layer-$layer/manifest.json"
done
copy_dflash_fixtures
'''
    payloads['run-dspark-hardware.sh'] = replace_once(shell,
        'docker start -a "$test_id" | tee "$output/dspark-console.log"',
        fixture_copy + 'docker start -a "$test_id" | tee "$output/dspark-console.log"')
    for name, source in payloads.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    if native_payloads:
        destination = scripts / 'dflash-t16-native-evidence'
        destination.mkdir()
        for name, data in native_payloads.items():
            (destination / name).write_bytes(data)
        for context in REPORTS:
            report = json.loads(native_payloads[f'dflash-t16-{context}.json'])
            qualify(report, context, hashes(scripts, SOURCES), report['native_sources'])
    simulator_preflight(scripts)
    record = dict(before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        target_context=4096, target_rows=16, streams=1, hardware_qualified=False,
        drafter_revision='dedf8df68adfb1afeaf7b7480c0a0243108177b4',
        cache_only=True, serving_defaults_changed=False)
    if native_payloads:
        record['native_t16_simulator_reports'] = {name: hashlib.sha256(data).hexdigest()
            for name, data in native_payloads.items()}
        record['comparison'] = 'dflash-composed-versus-native'
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--native-evidence', type=Path)
    arguments = parser.parse_args()
    stage(arguments.checkout, arguments.manifest, native_evidence=arguments.native_evidence)
