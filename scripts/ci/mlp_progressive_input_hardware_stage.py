"""Stage one-load combined activation comparison with fixed direct KV publication."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from mlp_progressive_input_preload import preload


FILES = ('mlp_progressive_input.py', 'mlp_progressive_input_report.py', 'mlp_progressive_input_gate.py',
    'mlp_progressive_input_preload.py', 'mlp_block_stream_runtime.py', 'mlp_block_stream_request.py',
    'mlp_block_stream_experiment.py', 'mlp_block_stream_report.py',
    'dflash_combined_request.py', 'dflash_native_comparison_report.py')


def stage(checkout, evidence, native_root, manifest):
    directory, scripts = Path(__file__).parent, Path(checkout) / 'scripts/ci'
    manifest = Path(manifest)
    destination = scripts / 'progressive-input-evidence'
    if manifest.exists() or destination.exists():
        raise ValueError('Fresh progressive hardware staging required')
    if not (scripts / 'draft-kv-slide-evidence').is_dir():
        raise ValueError('Stage the direct KV publication recipe first')
    payloads = {name: (directory / name).read_text() for name in FILES}
    payloads['run-dspark-hardware.sh'] = replace_once((scripts / 'run-dspark-hardware.sh').read_text(),
        '    -e "QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1"',
        '    -e "QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1" \\\n    -e "QWEN_PROGRESSIVE_INPUT_EXPERIMENT=1"')
    payloads['dspark-hardware-suite.sh'] = replace_once((scripts / 'dspark-hardware-suite.sh').read_text(),
        '> /experiment/results/draft-kv-slide-preload-admission.json\nset +e',
        '> /experiment/results/draft-kv-slide-preload-admission.json\n'
        'timeout -k 5 30 python3 -B /experiment-scripts/ci/mlp_progressive_input_preload.py '
        '> /experiment/results/progressive-input-preload-admission.json\nset +e')
    for name, source in payloads.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(evidence, destination)
    admission = preload(scripts, native_root)
    manifest.write_text(json.dumps(dict(admission=admission, hardware_qualified=False,
        performance_qualified=False, serving_defaults_changed=False,
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.evidence, options.native_root, options.manifest)
