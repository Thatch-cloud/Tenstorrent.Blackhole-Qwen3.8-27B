"""Stage matched k-block requests without changing the winning baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from mlp_k_block_gate import qualify, stage_candidate


def adapt(sources):
    result = dict(sources)
    result['dspark_request_experiment.py'] = replace_once(result['dspark_request_experiment.py'],
        "    schedule = (('publication', True), ('publication', False), ('publication', False))",
        "    schedule = (('publication', True), ('publication', True), ('publication', False),\n"
        "        ('publication', False), ('publication', False), ('publication', False))")
    result['dspark-target-hardware.py'] = replace_once(result['dspark-target-hardware.py'],
        '            from dspark_request_experiment import run_loaded_requests',
        '            from mlp_k_block_experiment import run_loaded_requests')
    result['run-dspark-hardware.sh'] = replace_once(result['run-dspark-hardware.sh'],
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_MLP_K_BLOCK=${QWEN_MLP_K_BLOCK:-0}" \\\n'
        '    -e "QWEN_DSPARK_MODE=$mode"')
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    if options.manifest.exists() or (scripts / 'k-block-evidence').exists():
        raise ValueError('Fresh k-block comparison staging required')
    names = ('dspark_request_experiment.py', 'dspark-target-hardware.py', 'run-dspark-hardware.sh')
    original = {name: (scripts / name).read_bytes() for name in names}
    result = adapt({name: payload.decode() for name, payload in original.items()})
    for name in ('mlp_k_block_experiment.py', 'mlp_k_block_comparison.py', 'mlp_k_block_gate.py',
            'mlp_k_block.py', 'frozen_recipe_context.py', 'frozen_mlp_buffer_trial.py',
            'mlp_weight_pipeline_gate.py', 'mlp_weight_pipeline.py'):
        result[name] = Path(__file__).with_name(name).read_text()
    for name, source in result.items():
        (scripts / name).write_bytes(source.encode())
    stage_candidate(scripts)
    shutil.copytree(options.evidence, scripts / 'k-block-evidence')
    evidence = qualify(scripts, scripts / 'k-block-evidence')
    options.manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(payload).hexdigest() for name, payload in original.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        simulator=evidence, hardware_qualified=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
