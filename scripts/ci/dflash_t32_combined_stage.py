"""Overlay the T32 components and stop at runtime admission, before weight loading."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from frozen_recipe_context import replace_once
from dflash_t32_cached_adapter import adapt_sources
from dflash_t32_native_stage import payloads as attention_payloads
from gdn_shared_qk_t32_adapter import payloads as gdn_payloads
from gdn_direct_window_t32_hardware_sources import payloads as window_payloads


FILES = ('full_dflash_request.py', 'mlp_block_stream_runtime.py', 'mlp_block_stream_gate.py',
    'mlp_weight_pipeline_report.py', 'mlp_down_grid_gate.py', 'frozen_recipe_context.py',
    'frozen_mlp_buffer_trial.py', 'mlp_register_epilogue_stage.py', 'dflash_native_comparison_report.py')


def stage(checkout, evidence, manifest, *, preflight_only=True):
    if type(preflight_only) is not bool:
        raise ValueError('Explicit preflight or full comparison selection required')
    checkout, evidence, manifest = Path(checkout), Path(evidence), Path(manifest)
    directory, scripts = Path(__file__).parent, checkout / 'scripts/ci'
    destination = scripts / 't32-evidence'
    if manifest.exists() or destination.exists():
        raise ValueError('Fresh T32 combined staging required')
    for name in ('attention', 'cache', 'gdn', 'windows', 'down', 'stream'):
        if not (evidence / name).is_dir():
            raise ValueError('Missing T32 component evidence: ' + name)
    sources = {name: (directory / name).read_text() for name in FILES}
    sources.update({path.name: path.read_text() for path in directory.glob('*t32*.py')
        if not path.name.startswith('test_')})
    for name, source in sources.items():
        compile(source, name, 'exec')
        (scripts / name).write_bytes(source.encode())
    generated = adapt_sources({name: (scripts / name).read_text() for name in
        ('dflash_device.py', 'draft_attention_branch.py', 'dflash_proposal_trace.py')})
    generated.update({name: source for name, source in attention_payloads(scripts).items() if name.endswith('.py')})
    generated.update(gdn_payloads({name: (scripts / name).read_text() for name in
        ('gdn_shared_qk_program.py', 'gdn_shared_qk_pipeline.py', 'shared_qk_norm_scatter.py')}))
    generated.update(window_payloads(scripts))
    launcher = scripts / 'run-dspark-hardware.sh'
    suite = scripts / 'dspark-hardware-suite.sh'
    generated[launcher.name] = replace_once(launcher.read_text(),
        '    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1"',
        '    -e "QWEN_MLP_BLOCK_STREAM_EXPERIMENT=1" \\\n    -e "QWEN_T32_COMBINED_EXPERIMENT=1"')
    generated[suite.name] = replace_once(suite.read_text(),
        '> /experiment/results/block-stream-preload-admission.json\nset +e',
        '> /experiment/results/block-stream-preload-admission.json\n'
        'timeout -k 5 45 python3 -B /experiment-scripts/ci/dflash_t32_preload.py '
        '> /experiment/results/t32-preload-admission.json\n'
        + ('exit 0\n' if preflight_only else '') + 'set +e')
    if not preflight_only:
        generated['dspark-target-hardware.py'] = replace_once(
            (scripts / 'dspark-target-hardware.py').read_text(),
            'from mlp_block_stream_experiment import run_loaded_requests',
            'from dflash_t32_comparison_experiment import run_loaded_requests')
    for name, source in generated.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(evidence, destination)
    manifest.write_text(json.dumps(dict(preflight_only=preflight_only, hardware_qualified=False,
        performance_qualified=False, serving_defaults_changed=False,
        sources={name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in {**sources, **generated}.items()}), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-comparison', action='store_true')
    options = parser.parse_args()
    stage(options.checkout, options.evidence, options.manifest, preflight_only=not options.run_comparison)
