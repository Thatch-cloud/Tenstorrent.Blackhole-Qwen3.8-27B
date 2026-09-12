"""Bounded two-chip transfer health check; no model, trace or custom kernels."""

import faulthandler
import json
import os
from pathlib import Path
import time


def require_hardware(environment):
    if environment.get('QWEN_HARDWARE_TESTS') != '1' or environment.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    if environment.get('TT_METAL_SIMULATOR') or environment.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch silicon required')


def check_shards(shards):
    if len(shards) != 2:
        raise AssertionError('Exactly two device shards required')


def main():
    require_hardware(os.environ)
    import torch
    import ttnn

    path = Path('/experiment/results/device-readback.json')
    report = dict(passed=False, checks=[], scope='Replicated BF16 TILE DRAM upload/readback on 1x2 fabric mesh; no model, trace, custom kernel or reset')
    started = time.monotonic()

    def stage(name, **details):
        report['last_stage'] = dict(stage=name, elapsed_seconds=time.monotonic() - started, **details)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['last_stage']), flush=True)

    mesh = None
    with path.with_suffix('.stacks.log').open('w') as stacks:
        faulthandler.dump_traceback_later(30, repeat=True, file=stacks)
        try:
            stage('mesh-open')
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
            stage('mesh-ready')
            for shape in ((1, 1, 32, 32), (1, 24, 128, 128)):
                count = 1
                for dimension in shape:
                    count *= dimension
                for pattern in range(3):
                    expected = ((torch.arange(count) + pattern * 7) % 31 - 15).reshape(shape).bfloat16()
                    stage('upload', shape=shape, pattern=pattern)
                    tensor = ttnn.from_torch(expected, device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    stage('upload-submitted', shape=shape, pattern=pattern)
                    shards = ttnn.get_device_tensors(tensor)
                    check_shards(shards)
                    for chip, shard in enumerate(shards):
                        stage('readback', shape=shape, pattern=pattern, chip=chip)
                        actual = ttnn.to_torch(shard)
                        if not torch.equal(actual, expected):
                            raise AssertionError(f'Upload/readback differs: {shape=} {pattern=} {chip=}')
                        report['checks'].append(dict(shape=shape, pattern=pattern, chip=chip, exact=True))
                        stage('readback-exact', shape=shape, pattern=pattern, chip=chip)
                    ttnn.deallocate(tensor)
            stage('synchronize')
            ttnn.synchronize_device(mesh)
            stage('mesh-close')
            ttnn.close_mesh_device(mesh)
            mesh = None
            if len(report['checks']) != 12:
                raise AssertionError('Incomplete health check')
            report['passed'] = True
            stage('complete')
        except BaseException as error:
            report['error'] = f'{type(error).__name__}: {error}'
            path.write_text(json.dumps(report, indent=2))
            raise
        finally:
            if mesh is not None:
                stage('cleanup-mesh-close')
                ttnn.close_mesh_device(mesh)
            faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
