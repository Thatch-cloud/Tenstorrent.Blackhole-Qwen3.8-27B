"""Simulator-only shared-page cache prerequisite against native serial writes."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from ordered_cache import HASHES, load_kernels, update


def main():
    spec = importlib.util.spec_from_file_location('sim_guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.require_simulator(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, choices=(1, 2, 4, 8, 16, 32), default=2)
    parser.add_argument('--start', type=int, choices=(15, 31, 63, 65, 4095, 16383), default=63)
    parser.add_argument('--page-columns', type=int, choices=(4, 1024), default=4)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.start + args.rows > args.page_columns * 64:
        parser.error('Positions exceed logical page-table coverage')
    kernels = load_kernels(os.environ['TT_METAL_HOME'])
    path = Path(os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, backend='ttsim', rows=args.rows, start=args.start, seed=args.seed, page_columns=args.page_columns,
        native_hashes=HASHES, generated_hashes={role: hashlib.sha256(source.encode()).hexdigest() for role, source in kernels.items()},
        scope='Complete physical BF8 cache equality against native ordered B1 writes; not performance or full-model certification')

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

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(value, device=mesh, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(value):
            parts = ttnn.get_device_tensors(value)
            if len(parts) != 2:
                raise AssertionError('Two simulated chips required')
            return [ttnn.to_torch(part).clone() for part in parts]

        torch.manual_seed(args.seed)
        initial = upload(torch.randn(max(8, args.page_columns), 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
        serial = ttnn.clone(initial, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        candidate = ttnn.clone(initial, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        packed_host = torch.randn(1, args.rows, 32, 256).bfloat16()
        packed = upload(packed_host)
        positions = torch.arange(args.start, args.start + args.rows, dtype=torch.int32)
        pages_host = torch.tensor([[3, 1, 6, 2]], dtype=torch.int32).repeat(1, args.page_columns // 4)
        packed_positions, packed_pages = upload(positions, ttnn.int32), upload(pages_host.repeat(args.rows, 1), ttnn.int32)
        single_pages = upload(pages_host, ttnn.int32)
        shard_config = ttnn.create_sharded_memory_config([32, 256], ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        for index in range(args.rows):
            stage(f'native-serial-{index}')
            row = upload(packed_host[:, index:index + 1].contiguous())
            sharded = ttnn.to_memory_config(row, shard_config)
            position = upload(positions[index:index + 1], ttnn.int32)
            ttnn.experimental.paged_update_cache(serial, sharded, update_idxs_tensor=position, page_table=single_pages)
            for value in (row, sharded, position):
                ttnn.deallocate(value)
        expected = host(serial)
        stage('ordered-candidate')
        update(mesh, candidate, packed, packed_positions, packed_pages, kernels)
        actual = host(candidate)
        report['unequal'] = [int((left != right).sum()) for left, right in zip(actual, expected, strict=True)]
        if any(report['unequal']):
            raise AssertionError(f"Complete physical cache differs: {report['unequal']}")
        if not all(torch.equal(value, packed_host) for value in host(packed)):
            raise AssertionError('Packed input modified')
        if all(torch.equal(value, expected[chip]) for chip, value in enumerate(host(initial))):
            raise AssertionError('Omitted-write negative control not detected')
        stage('repeat-after-restore')
        ttnn.copy(initial, candidate)
        update(mesh, candidate, packed, packed_positions, packed_pages, kernels)
        if not all(torch.equal(value, expected[chip]) for chip, value in enumerate(host(candidate))):
            raise AssertionError('Repeated ordered cache write differs')
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
