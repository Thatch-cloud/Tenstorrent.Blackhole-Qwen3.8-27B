"""Overlay the matched drafter driver onto the fully staged five-component recipe."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once

FILES = ('drafter_comparison_experiment.py', 'drafter_comparison_report.py',
         'drafter_request_environment.py', 'dflash_combined_request.py', 'full_dflash_request.py',
         'dflash_device.py', 'dflash_request_runtime.py', 'draft_attention_branch.py', 'dflash_proposal_inputs.py',
         'dflash_attention_mask.py', 'draft_shared_head.py', 'cumulative_t16_scope.py', 'dflash-fixtures.sh')


def stage(checkout, manifest):
    directory, scripts = Path(__file__).parent, Path(checkout) / 'scripts/ci'
    manifest = Path(manifest)
    if manifest.exists():
        raise ValueError('Fresh comparison staging required')
    for name in ('compact-score-evidence', 'mlp-down-grid-evidence', 'register-epilogue-evidence'):
        if not (scripts / name).is_dir():
            raise ValueError('Complete promoted component evidence required: ' + name)
    from dspark_hardware_gate import simulator_preflight
    simulator_preflight(scripts)
    originals = {name: (scripts / name).read_text() for name in ('dspark-target-hardware.py', 'run-dspark-hardware.sh')}
    payloads = {name: (directory / name).read_text() for name in FILES}
    payloads['dspark-target-hardware.py'] = replace_once(originals['dspark-target-hardware.py'],
        'from cumulative_t16_experiment import run_loaded_requests',
        'from drafter_comparison_experiment import run_loaded_requests')
    shell = replace_once(originals['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_DRAFTER_COMPARISON=1" \\\n    -e "QWEN_DSPARK_MODE=$mode"')
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
    simulator_preflight(scripts)
    record = dict(before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        target_context=4096, target_rows=16, streams=1, hardware_qualified=False,
        drafter_revision='dedf8df68adfb1afeaf7b7480c0a0243108177b4',
        cache_only=True, serving_defaults_changed=False)
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    arguments = parser.parse_args()
    stage(arguments.checkout, arguments.manifest)
