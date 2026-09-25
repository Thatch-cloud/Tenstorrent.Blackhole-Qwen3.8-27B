"""Overlay transport-only ABBA on the staged native-DFlash combined recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from mlp_block_stream_gate import qualify
from mlp_register_epilogue_gate import qualify as qualify_register


FILES = ('mlp_block_stream.py', 'mlp_block_stream.cpp', 'mlp_block_stream_projection.py',
    'mlp_block_stream_probe.py', 'mlp_block_stream_gate.py', 'mlp_block_stream_pool.py',
    'mlp_block_stream_runtime.py', 'mlp_block_stream_request.py', 'mlp_block_stream_experiment.py',
    'mlp_block_stream_report.py', 'mlp_block_stream_preload.py', 'mlp_weight_pipeline_report.py')


def stage(checkout, evidence, native_root, manifest):
    directory, scripts = Path(__file__).parent, checkout / 'scripts/ci'
    if manifest.exists() or (scripts / 'block-stream-evidence').exists():
        raise ValueError('Fresh block-stream hardware staging required')
    originals = {name: (scripts / name).read_text() for name in
        ('dspark-target-hardware.py', 'run-dspark-hardware.sh', 'dspark-hardware-suite.sh')}
    payloads = {name: (directory / name).read_text() for name in FILES}
    payloads['dspark-target-hardware.py'] = replace_once(originals['dspark-target-hardware.py'],
        'from dflash_native_comparison_experiment import run_loaded_requests',
        'from mlp_block_stream_experiment import run_loaded_requests')
    payloads['run-dspark-hardware.sh'] = replace_once(originals['run-dspark-hardware.sh'],
        '    -e "QWEN_DFLASH_NATIVE_COMPARISON=1"',
        '    -e "QWEN_DFLASH_NATIVE_COMPARISON=1" \\\n'
        '    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"')
    payloads['dspark-hardware-suite.sh'] = replace_once(originals['dspark-hardware-suite.sh'],
        '> /experiment/results/dflash-native-preload-admission.json\nset +e',
        '> /experiment/results/dflash-native-preload-admission.json\n'
        'timeout -k 5 30 python3 -B /experiment-scripts/ci/mlp_block_stream_preload.py '
        '> /experiment/results/block-stream-preload-admission.json\nset +e')
    for name, source in payloads.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(evidence, scripts / 'block-stream-evidence')
    register = qualify_register(scripts, scripts / 'register-epilogue-evidence', runtime_root=native_root)
    admission = qualify(scripts, scripts / 'block-stream-evidence', register)
    manifest.write_text(json.dumps(dict(admission=admission,
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        hardware_qualified=False, performance_qualified=False, serving_defaults_changed=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    arguments = parser.parse_args()
    stage(arguments.checkout, arguments.evidence, arguments.native_root, arguments.manifest)
