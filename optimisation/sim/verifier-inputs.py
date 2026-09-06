"""Simulator-only in-place input staging versus native RoPE construction."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from verifier_inputs import stage_inputs


def main():
    spec = importlib.util.spec_from_file_location('sim_guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.require_simulator(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, choices=(1, 2, 4, 8, 16), default=2)
    args = parser.parse_args()
    path = Path(os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, backend='ttsim', rows=args.rows, checks=[],
        scope='In-place token/position/RoPE host staging only; no trace replay, model or throughput certification')

    def stage(name):
        report['last_stage'] = name
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=name)), flush=True)

    mesh = None
    try:
        import torch
        import ttnn
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        report['native_rope_sha256'] = hashlib.sha256(Path(inspect.getfile(rot_mats_decode)).read_bytes()).hexdigest()
        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)

        def upload(value, dtype):
            return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def equal(value, expected):
            shards = ttnn.get_device_tensors(value)
            return len(shards) == 2 and all(torch.equal(ttnn.to_torch(part), expected) for part in shards)

        positions = torch.arange(args.rows, dtype=torch.int32)
        pages_host = torch.arange(1024, dtype=torch.int32).reshape(1, 1024).repeat(args.rows, 1)
        cos, sin = rot_mats_decode(mesh, 64, 65536, 1000000, positions)
        fixture = SimpleNamespace(rows=args.rows, operations=ttnn,
            model=SimpleNamespace(mesh_device=mesh, args=SimpleNamespace(vocab_size=248320, rope_head_dim=64, rope_theta=1000000)),
            tokens=upload(positions.reshape(args.rows, 1), ttnn.uint32), positions=upload(positions, ttnn.int32),
            cos=cos, sin=sin, pages=upload(pages_host, ttnn.int32),
            singleton_positions=[upload(position.reshape(1), ttnn.int32) for position in positions])
        for start in (31, 4095, 16383):
            stage(f'input-update-{start}')
            tokens = [(start + index * 137) % 248320 for index in range(args.rows)]
            stage_inputs(fixture, tokens, start)
            expected_positions = torch.arange(start, start + args.rows, dtype=torch.int32)
            expected_rope = rot_mats_decode(mesh, 64, 65536, 1000000, expected_positions)
            if not equal(fixture.tokens, torch.tensor(tokens, dtype=torch.uint32).reshape(args.rows, 1)) or not equal(fixture.positions, expected_positions):
                raise AssertionError('Packed token or position update differs')
            for index, position in enumerate(fixture.singleton_positions):
                if not equal(position, expected_positions[index:index + 1]):
                    raise AssertionError('A B1 attention position was left stale')
            for value, reference in zip((fixture.cos, fixture.sin), expected_rope, strict=True):
                if not all(torch.equal(ttnn.to_torch(actual), ttnn.to_torch(expected))
                           for actual, expected in zip(ttnn.get_device_tensors(value), ttnn.get_device_tensors(reference), strict=True)):
                    raise AssertionError('Staged RoPE differs from native construction')
                ttnn.deallocate(reference)
            if not equal(fixture.pages, pages_host):
                raise AssertionError('Input staging modified page ownership')
            report['checks'].append(dict(start=start, tokens_exact=True, all_positions_exact=True, native_rope_exact=True, pages_unchanged=True))
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
