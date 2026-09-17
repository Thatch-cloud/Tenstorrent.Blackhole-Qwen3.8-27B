"""Add compact selection to an explicitly staged direct-window experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from compact_score_gate import qualify
from compact_score_hardware_sources import payloads as hardware_payloads
from cumulative_t16_experiment import COMPACT_FILES, DOWN_FILES
from frozen_recipe_context import replace_once
from mlp_down_grid_gate import qualify as qualify_down


def stage(checkout, evidence, manifest, *, down_evidence=None, native_root=None):
    directory, scripts = Path(__file__).parent, Path(checkout) / 'scripts/ci'
    manifest = Path(manifest)
    if manifest.exists() or (scripts / 'compact-score-evidence').exists():
        raise ValueError('Fresh cumulative staging required')
    qualify(directory, evidence)
    if (down_evidence is None) != (native_root is None):
        raise ValueError('Down-grid evidence and pinned runtime sources must be supplied together')
    down = qualify_down(down_evidence, directory, native_root) if down_evidence is not None else None
    for name in ('dspark_markov_device.py', 'dspark_markov_score_layout.py', 'dspark_score_layout.py',
                 'dspark_score_layout_io.cpp', 'dspark_score_layout_compute.cpp',
                 'attention_batch.py', 'gdn_multitoken_conv.py'):
        if (scripts / name).read_bytes() != (directory / name).read_bytes():
            raise ValueError('Frozen control differs from qualified compact source: ' + name)
    payloads = hardware_payloads(directory)
    for name in COMPACT_FILES + DOWN_FILES + ('cumulative_t16_experiment.py', 'cumulative_t16_scope.py'):
        if name not in payloads:
            payloads[name] = (directory / name).read_text()
    originals = {name: (scripts / name).read_bytes() for name in
                 ('dspark-target-hardware.py', 'run-dspark-hardware.sh')}
    payloads['dspark-target-hardware.py'] = replace_once(originals['dspark-target-hardware.py'].decode(),
        '            from gdn_direct_window_experiment import run_loaded_requests',
        '            from cumulative_t16_experiment import run_loaded_requests')
    payloads['run-dspark-hardware.sh'] = replace_once(originals['run-dspark-hardware.sh'].decode(),
        '    -e "QWEN_DSPARK_MODE=$mode"',
        '    -e "QWEN_CUMULATIVE_T16=${QWEN_CUMULATIVE_T16:-0}" \\\n'
        '    -e "QWEN_CUMULATIVE_MLP_DOWN=${QWEN_CUMULATIVE_MLP_DOWN:-0}" \\\n'
        '    -e "QWEN_COMPACT_SCORE_HARDWARE=${QWEN_COMPACT_SCORE_HARDWARE:-0}" \\\n'
        '    -e "QWEN_DSPARK_MODE=$mode"')
    for name, source in payloads.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    shutil.copytree(evidence, scripts / 'compact-score-evidence')
    admission = qualify(scripts, scripts / 'compact-score-evidence')
    if down is not None:
        shutil.copytree(down_evidence, scripts / 'mlp-down-grid-evidence')
        if qualify_down(scripts / 'mlp-down-grid-evidence', scripts, native_root) != down:
            raise ValueError('Staged down-grid admission differs')
    result = dict(compact_admission=admission,
        before={name: hashlib.sha256(source).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        components=['direct_windows', 'compact_scores'] + (['wider_mlp_down'] if down is not None else []),
        down_admission=down, hardware_qualified=False, performance_qualified=False)
    manifest.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--down-evidence', type=Path)
    parser.add_argument('--native-root', type=Path)
    options = parser.parse_args()
    stage(options.checkout, options.evidence, options.manifest,
          down_evidence=options.down_evidence, native_root=options.native_root)
