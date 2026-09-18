"""Stage T32 replay of the same block-stream register arithmetic; no hardware admission."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


def payloads(originals):
    projection = originals['fused_1d.py']
    projection = replace_once(projection, 'token_rows != 16', 'token_rows != 32')
    projection = replace_once(projection, 'T16 target-math simulator candidate', 'T32 target-math simulator candidate')
    projection = replace_once(projection, 'from mlp_block_stream_projection import validate_binding',
        'from mlp_block_stream_t32_projection import validate_binding')
    projection = replace_once(projection, '        self.rounding_runtime = rounding_runtime(source_root)',
        '        from mlp_block_stream_t32_stage import require_simulator\n'
        '        require_simulator()\n'
        '        self.rounding_runtime = rounding_runtime(source_root)')
    helper = replace_once(originals['mlp_block_stream_projection.py'],
        'projection.token_rows != 16', 'projection.token_rows != 32')
    helper = replace_once(helper, 'Exact T16 three-pair fused target configuration required',
        'Exact T32 three-pair fused target configuration required')
    probe = originals['fused-batch-probe.py']
    for before, after in (
        ('    trace_rows = (16,)', '    trace_rows = (32,)'),
        ('for rows in (16,):', 'for rows in (32,):'),
        ('from mlp_block_stream_projection import bind_stream', 'from mlp_block_stream_t32_projection import bind_stream'),
        ('T16 simulator-only buffering qualification', 'T32 simulator-only block-stream qualification'),
        ('T16 and both chips required', 'T32 and both chips required'),
        ('T16 block-stream MLP numerical and replay checks', 'T32 block-stream MLP numerical and replay checks')):
        probe = replace_once(probe, before, after)
    result = {'fused_1d.py': projection, 'fused-batch-probe.py': probe,
        'mlp_block_stream_t32_projection.py': helper}
    for name, source in result.items():
        compile(source, name, 'exec')
    return result


def require_simulator():
    import os

    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') or os.environ.get('QWEN_CARDS_ALLOCATED')):
        raise ValueError('T32 register and block-stream arithmetic requires separate simulator admission')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    if options.manifest.exists():
        raise ValueError('Fresh T32 staging manifest required')
    scripts = options.checkout / 'scripts/ci'
    baseline = json.loads((options.checkout / 'experiment-results/block-stream-candidate.json').read_text())
    originals = {name: (scripts / name).read_text() for name in
        ('fused_1d.py', 'fused-batch-probe.py', 'mlp_block_stream_projection.py')}
    for name in originals:
        if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != baseline['after'][name]:
            raise ValueError('Exact T16 staging prerequisite changed: ' + name)
    sources = payloads(originals)
    sources[Path(__file__).name] = Path(__file__).read_text()
    for name, source in sources.items():
        (scripts / name).write_bytes(source.encode())
    options.manifest.write_text(json.dumps(dict(token_rows=32, arithmetic_changed=False,
        weight_transport_changed=False, simulator_qualified=False, hardware_qualified=False,
        performance_qualified=False, sources={name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in sources.items()}), indent=2) + '\n')


if __name__ == '__main__':
    main()
