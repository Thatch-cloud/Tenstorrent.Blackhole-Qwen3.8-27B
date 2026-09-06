"""Simulator-only fused GDN prefix publication with inactive-slot canaries."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from gdn_commit_dma import publish


def main():
    spec = importlib.util.spec_from_file_location('sim_guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.require_simulator(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, choices=(2, 4, 8, 16), default=2)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--layers', type=int, choices=(2, 48), default=2)
    args = parser.parse_args()
    path = Path(os.environ['QWEN_SIM_REPORT'])
    root = Path(__file__).resolve().parents[2] / 'scripts/ci'
    report = dict(passed=False, backend='ttsim', rows=args.rows, seed=args.seed, layers=args.layers, checks=[],
        hashes={suffix: hashlib.sha256((root / f'gdn_commit_dma.{suffix}').read_bytes()).hexdigest() for suffix in ('py', 'cpp')},
        scope='Synthetic layers, all prefix selections, exact native slot zero and external checkpoint; inactive slots and sources unchanged. No hardware timing claim.')

    def stage(name):
        report['last_stage'] = name
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=name)), flush=True)

    mesh = None
    try:
        import torch
        import ttnn
        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)

        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def equal(value, expected):
            shards = ttnn.get_device_tensors(value)
            return len(shards) == 2 and all(torch.equal(ttnn.to_torch(part), expected) for part in shards)

        torch.manual_seed(args.seed)
        host_layers, layers = [], []
        compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
        shapes = compact + [(args.rows, 24, 128, 128)] + [(1, args.rows, 5120)] * 4
        shapes += [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4 + compact
        stage('fixture-upload')
        for layer in range(args.layers):
            values = [torch.randn(shape).bfloat16() for shape in shapes]
            host_layers.append(values)
            layers.append([upload(value) for value in values])
        for prefix in range(args.rows + 1):
            stage(f'commit-{prefix}')
            publish(mesh, layers, prefix)
            for index, (values, native) in enumerate(zip(host_layers, layers, strict=True)):
                selected = values[:5] if prefix == 0 else [values[5][prefix - 1:prefix]] + [value[:, prefix - 1:prefix] for value in values[6:10]]
                for slot in range(5):
                    expected = values[10 + slot].clone()
                    if slot == 0:
                        expected[0:1] = selected[slot]
                    else:
                        expected[:, 0:1] = selected[slot]
                    if not equal(native[10 + slot], expected) or not equal(native[15 + slot], selected[slot]):
                        raise AssertionError(f'Commit mismatch or inactive-slot corruption: {prefix=} {index=} {slot=}')
            report['checks'].append(dict(prefix=prefix, native_exact=True, inactive_exact=True, checkpoints_exact=True))
        stage('immutable-sources')
        for values, native in zip(host_layers, layers, strict=True):
            if any(not equal(value, reference) for value, reference in zip(native[:10], values[:10], strict=True)):
                raise AssertionError('Entry or packed history was modified')
        stage('mesh-close')
        ttnn.close_mesh_device(mesh)
        mesh = None
        report['passed'] = True
        stage('complete')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        path.write_text(json.dumps(report, indent=2))
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    main()
