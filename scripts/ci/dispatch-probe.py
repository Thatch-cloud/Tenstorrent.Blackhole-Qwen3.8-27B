"""Bounded dispatch placement gate; no model, custom kernels or upload routing changes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from gdn_multitoken_conv import addresses, release_owned


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def backend(environment):
    if environment.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Dispatch placement must be tested in fast-dispatch mode')
    if environment.get('TT_METAL_SIMULATOR'):
        return 'ttsim-fast-dispatch'
    if environment.get('QWEN_HARDWARE_TESTS') != '1' or environment.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit two-card hardware allocation required')
    return 'hardware'


def dispatch_config(operations, selection):
    if selection not in ('worker', 'ethernet'):
        raise ValueError('Explicit worker or ethernet dispatch required')
    selected = operations.DispatchCoreType.ETH if selection == 'ethernet' else operations.DispatchCoreType.WORKER
    return operations.DispatchCoreConfig(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', choices=('worker', 'ethernet'), required=True)
    parser.add_argument('--fabric', choices=('1d', 'none'), default='1d')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    report = dict(passed=False, backend=backend(os.environ), dispatch=options.dispatch, fabric=options.fabric, checks=[],
        scope='Explicit fabric configuration, reported grid, replicated transfers and exact changed-input add trace; no collective or model performance claim')
    report['probe_sha256'] = file_hash(__file__)
    root = Path(os.environ['TT_METAL_HOME'])
    report['source_hashes'] = {name: file_hash(root / name) for name in (
        'ttnn/core/device.cpp', 'tt_metal/llrt/core_descriptor.cpp',
        'tt_metal/impl/context/metal_env_impl.hpp',
        'tt_metal/core_descriptors/blackhole_140_arch.yaml',
        'tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml')}
    report['runtime_hashes'] = {name: file_hash(root / 'build_Release/lib' / name)
        for name in ('libtt_metal.so', '_ttnncpp.so')}
    if os.environ.get('TT_METAL_MOCK_CLUSTER_DESC_PATH'):
        report['mock_cluster'] = dict(path=os.environ['TT_METAL_MOCK_CLUSTER_DESC_PATH'],
            sha256=file_hash(os.environ['TT_METAL_MOCK_CLUSTER_DESC_PATH']))
    mesh, trace = None, None
    owned = []

    def stage(name):
        report['last_stage'] = name
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=name, dispatch=options.dispatch)), flush=True)

    import torch
    import ttnn

    try:
        stage('config')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.fabric == '1d' else ttnn.FabricConfig.DISABLED)
        config = dispatch_config(ttnn, options.dispatch)
        report['resolved_config'] = dict(type=str(config.type), axis=str(config.axis))
        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
            trace_region_size=134217728, dispatch_core_config=config)
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        report['mesh_compute_grid'] = [grid.x, grid.y]
        expected = [((torch.arange(1024) + pattern * 7) % 31 - 15).reshape(1, 1, 32, 32).bfloat16()
                    for pattern in range(3)]
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for value in expected]
        stage('upload')
        tensor = ttnn.from_torch(expected[0], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        owned.append(tensor)
        before = addresses(ttnn, tensor)
        if len(before) != 2:
            raise AssertionError('Exactly two device shards required')
        stage('warmup')
        warm = ttnn.add(tensor, tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        stage('capture')
        trace, output = capture_operation(ttnn, mesh,
            lambda: ttnn.add(tensor, tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        owned.append(output)
        for pattern, source in enumerate(host_inputs):
            stage(f'replay-{pattern}')
            ttnn.copy_host_to_device_tensor(source, tensor)
            ttnn.synchronize_device(mesh)
            if addresses(ttnn, tensor) != before:
                raise AssertionError('Captured input addresses changed')
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            shards = ttnn.get_device_tensors(output)
            if len(shards) != 2:
                raise AssertionError('Exactly two output shards required')
            for chip, shard in enumerate(shards):
                if not torch.equal(ttnn.to_torch(shard), expected[pattern] * 2):
                    raise AssertionError(f'Dispatch trace output differs at pattern{pattern}/chip{chip}')
                report['checks'].append(dict(pattern=pattern, chip=chip, exact=True))
        if len(report['checks']) != 6:
            raise AssertionError('All six changed-input checks required')
    except BaseException as error:
        report['failed_stage'] = report.get('last_stage')
        report['error'] = f'{type(error).__name__}: {error}'
        stage('failed')
        raise
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
    report['passed'] = True
    stage('complete')


if __name__ == '__main__':
    main()
