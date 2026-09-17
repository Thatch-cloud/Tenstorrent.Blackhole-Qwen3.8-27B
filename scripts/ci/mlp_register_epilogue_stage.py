"""Stage a bounded numerical test; no hardware or serving route."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION, replace_once
from frozen_mlp_buffer_trial import adapt_probe
from mlp_register_epilogue import adapt_projection


def stage(checkout, manifest):
    if manifest.exists():
        raise ValueError('Fresh candidate manifest required')
    scripts = checkout / 'scripts/ci'
    originals = {}
    for name in ('fused_1d.py', 'fused-batch-probe.py'):
        source = subprocess.check_output(['git', '-C', str(checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text(encoding='utf-8') != source:
            raise ValueError('Exact frozen source required: ' + name)
        originals[name] = source
    payloads = {'fused_1d.py': adapt_projection(originals['fused_1d.py']),
        'fused-batch-probe.py': replace_once(adapt_probe(originals['fused-batch-probe.py']),
            'T16 buffering only; other row widths and performance unqualified',
            'T16 register-resident rounded epilogue; no performance qualification'),
        'mlp_register_epilogue.py': Path(__file__).with_name('mlp_register_epilogue.py').read_text(encoding='utf-8')}
    payloads['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(encoding='utf-8'),
        'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
        'timeout -k 15 510 python3 -u /experiment-scripts/ci/fused-batch-probe.py')
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    manifest.write_text(json.dumps(dict(
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        token_rows=16, pairs_per_worker=3, readers_changed=False, buffers_changed=False,
        accumulation_changed=False, activation_processor='MATH instead of PACK',
        bf16_rounding='explicit SFPU before unchanged BF16 product',
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False), indent=2) + '\n',
        encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.manifest)


if __name__ == '__main__':
    main()
