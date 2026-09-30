"""Stage the admitted publication comparison on the complete serial-weight recipe."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from draft_kv_slide_gate import qualify
from frozen_recipe_context import replace_once


FILES = ('draft_kv_slide.py', 'draft_kv_slide.cpp', 'draft-kv-slide-probe.py',
    'draft_kv_slide_report.py', 'draft_kv_slide_gate.py', 'draft_kv_slide_adapter.py',
    'draft_kv_slide_scope.py', 'dflash_combined_request.py', 'dflash_native_comparison_report.py',
    'mlp_block_stream_experiment.py', 'mlp_block_stream_report.py')


def stage(checkout, evidence, manifest, *, direct_dma=False):
    scripts, directory = Path(checkout) / 'scripts/ci', Path(__file__).parent
    destination, manifest = scripts / 'draft-kv-slide-evidence', Path(manifest)
    if destination.exists() or manifest.exists():
        raise ValueError('Fresh publication staging required')
    payloads = {name: (directory / name).read_text() for name in FILES}
    if direct_dma:
        payloads['draft_kv_slide.cpp'] = (directory / 'draft_kv_slide_direct.cpp').read_text()
    payloads['run-dspark-hardware.sh'] = replace_once((scripts / 'run-dspark-hardware.sh').read_text(),
        '    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"',
        '    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1" \\\n    -e "QWEN_DRAFT_KV_SLIDE_EXPERIMENT=1"')
    payloads['dspark-hardware-suite.sh'] = replace_once((scripts / 'dspark-hardware-suite.sh').read_text(),
        '> /experiment/results/block-stream-preload-admission.json\nset +e',
        '> /experiment/results/block-stream-preload-admission.json\n'
        'timeout -k 5 30 python3 -B /experiment-scripts/ci/draft_kv_slide_gate.py '
        '> /experiment/results/draft-kv-slide-preload-admission.json\nset +e')
    for name, source in payloads.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(evidence, destination)
    admission = qualify(scripts, destination)
    manifest.write_text(json.dumps(dict(admission=admission, direct_dma=direct_dma, hardware_qualified=False,
        performance_qualified=False, serving_defaults_changed=False,
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--direct-dma', action='store_true')
    options = parser.parse_args()
    stage(options.checkout, options.evidence, options.manifest, direct_dma=options.direct_dma)
