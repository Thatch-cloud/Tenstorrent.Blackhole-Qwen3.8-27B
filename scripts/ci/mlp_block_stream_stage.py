"""Stage transport-admitted T16 numerical qualification, never serving code."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION, replace_once
from mlp_block_stream_admission import admit_transport
from mlp_block_stream_probe import adapt_probe
from mlp_block_stream_projection import adapt_projection
from mlp_register_epilogue import adapt_projection as register_projection


def stage(checkout, report, manifest):
    if manifest.exists():
        raise ValueError('Fresh staging manifest required')
    root = Path(__file__).parent
    admission = admit_transport(json.loads(report.read_text()), root)
    scripts = checkout / 'scripts/ci'
    originals = {}
    for name in ('fused_1d.py', 'fused-batch-probe.py'):
        original = subprocess.check_output(['git', '-C', str(checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != original:
            raise ValueError('Exact frozen source required: ' + name)
        originals[name] = original
    payloads = {'fused_1d.py': adapt_projection(register_projection(originals['fused_1d.py'], nearest_away=True)),
        'fused-batch-probe.py': adapt_probe(originals['fused-batch-probe.py']),
        'simulator-suite.sh': replace_once((scripts / 'simulator-suite.sh').read_text(),
            'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py',
            'timeout -k 15 420 python3 -u /experiment-scripts/ci/fused-batch-probe.py')}
    for name in ('mlp_block_stream.py', 'mlp_block_stream.cpp', 'mlp_block_stream_projection.py',
            'mlp_block_stream_probe.py', 'mlp_register_epilogue.py', 'mlp_rounding_policy.py',
            'frozen_mlp_buffer_trial.py', 'mlp_register_epilogue_stage.py',
            'frozen_recipe_context.py', 'frozen_context_geometry.py'):
        payloads[name] = (root / name).read_text()
    for name, source in payloads.items():
        (scripts / name).write_bytes(source.encode())
    manifest.write_text(json.dumps(dict(transport_admission=admission,
        transport_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        before={name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()},
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in payloads.items()},
        simulator_qualified=False, hardware_qualified=False, performance_qualified=False,
        scope='T16 transport plus existing rounded register arithmetic; no combined qualification'), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--transport-report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.transport_report, options.manifest)


if __name__ == '__main__':
    main()
