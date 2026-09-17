"""Weight-free wide-page ordered-cache eager/replay checks against native serial writes."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from frozen_context_geometry import geometry
from frozen_ladder_ordered_cache import page_geometry
from ladder_cache_reference import snapshot as reference_snapshot
from ordered_cache import HASHES, load_kernels, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=int, choices=(65536, 131072, 261888), required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or options.output.exists()):
        raise ValueError('Fresh device-free simulator required')
    import torch
    import ttnn

    names = ('ladder-cache-probe.py', 'frozen_ladder_ordered_cache.py', 'frozen_context_geometry.py',
        'ordered_cache.py', 'attention_batch.py', 'ladder_cache_reference.py')
    fingerprints = lambda: {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in names}
    kernels = load_kernels(os.environ['TT_METAL_HOME'])
    report = dict(passed=False, closed_cleanly=False, backend='simulator', checks=[], context=options.context,
        sources=fingerprints(), native_hashes=HASHES,
        generated_hashes={role: hashlib.sha256(source.encode()).hexdigest() for role, source in kernels.items()},
        performance_qualified=False, model_integrated=False, scope=__doc__)
    mesh, trace = None, None
    owned = []

    def stage(name, **details):
        report['stage'] = name
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=name, **details)), flush=True)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()

        def upload(value, dtype):
            result = ttnn.from_torch(value, device=mesh, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            owned.append(result)
            return result

        def replace(value, destination, dtype):
            host = ttnn.from_torch(value, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(host, destination)

        def snapshot(value):
            parts = ttnn.get_device_tensors(value)
            if len(parts) != 2:
                raise AssertionError('Both simulated chips required')
            return [ttnn.to_torch(part).clone() for part in parts]

        def compare(left, expected_parts, context, seed, name):
            left_parts = ttnn.get_device_tensors(left)
            if len(left_parts) != 2 or len(expected_parts) != 2:
                raise AssertionError('Both simulated chips required')
            for chip, (actual, expected) in enumerate(zip(left_parts, expected_parts, strict=True)):
                exact = torch.equal(ttnn.to_torch(actual), expected)
                report['checks'].append(dict(context=context, seed=seed, chip=chip, name=name, exact=exact))
                if not exact:
                    raise AssertionError('Wide page-table cache mismatch: ' + name)

        for context in (options.context,):
            pages_count = geometry(context)['target_page_count']
            stage('allocate', context=context, page_columns=pages_count)
            initial = torch.zeros(pages_count + 8, 2, 64, 256, dtype=torch.bfloat16)
            serial, candidate = [upload(initial, ttnn.bfloat8_b) for unused in range(2)]
            packed = upload(torch.zeros(1, 16, 32, 256, dtype=torch.bfloat16), ttnn.bfloat16)
            positions = upload(torch.arange(context - 1, context + 15, dtype=torch.int32), ttnn.int32)
            page_values = torch.arange(pages_count, dtype=torch.int32).flip(0).reshape(1, -1)
            pages = upload(page_values.repeat(16, 1), ttnn.int32)
            singleton_pages = upload(page_values, ttnn.int32)
            singleton_position = upload(torch.tensor([context - 1], dtype=torch.int32), ttnn.int32)
            singleton_input = upload(torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16), ttnn.bfloat16)
            sharding = ttnn.create_sharded_memory_config([32, 256], ttnn.CoreGrid(y=1, x=1),
                ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
            references = []
            written_pages = set()
            for seed in (0, 1):
                stage('native-reference', context=context, seed=seed)
                torch.manual_seed(seed)
                inputs = torch.randn(1, 16, 32, 256).bfloat16()
                indexes = torch.arange(context - 1 + seed * 2, context + 15 + seed * 2, dtype=torch.int32)
                for index in range(16):
                    replace(inputs[:, index:index + 1].contiguous(), singleton_input, ttnn.bfloat16)
                    replace(indexes[index:index + 1], singleton_position, ttnn.int32)
                    sharded = ttnn.to_memory_config(singleton_input, sharding)
                    ttnn.experimental.paged_update_cache(serial, sharded,
                        update_idxs_tensor=singleton_position, page_table=singleton_pages)
                    ttnn.deallocate(sharded)
                written_pages.update(int(page_values[0, int(position) // 64]) for position in indexes)
                references.append((inputs, indexes, reference_snapshot(ttnn, serial, initial, written_pages)))
                report.setdefault('reference_pages', []).append(dict(seed=seed, pages=sorted(written_pages),
                    candidate_comparison='complete allocated cache on both chips'))
            with page_geometry(context) as evidence:
                for seed, (inputs, indexes, expected) in enumerate(references):
                    stage('candidate', context=context, seed=seed)
                    replace(inputs, packed, ttnn.bfloat16)
                    replace(indexes, positions, ttnn.int32)
                    operation = lambda: update(mesh, candidate, packed, positions, pages, kernels)
                    if seed == 0:
                        operation()
                        compare(candidate, expected, context, seed, 'eager')
                        trace, unused = capture_operation(ttnn, mesh, operation)
                    stage('replay', context=context, seed=seed)
                    ttnn.execute_trace(mesh, trace, blocking=True)
                    compare(candidate, expected, context, seed, 'replay')
                    for chip, shard in enumerate(ttnn.get_device_tensors(packed)):
                        exact = torch.equal(ttnn.to_torch(shard), inputs)
                        report['checks'].append(dict(context=context, seed=seed, chip=chip, name='input_unchanged', exact=exact))
                        if not exact:
                            raise AssertionError('Packed input changed')
                ttnn.release_trace(mesh, trace)
                trace = None
            if not evidence['restored']:
                raise AssertionError('Default cache validation not restored')
            ttnn.synchronize_device(mesh)
            for value in reversed(owned):
                ttnn.deallocate(value)
            owned.clear()
        if len(report['checks']) != 10 or not all(check['exact'] for check in report['checks']):
            raise AssertionError('Complete eager/replay and input matrix required')
        report['passed'] = True
        stage('complete')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                for value in reversed(owned):
                    ttnn.deallocate(value)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            report['sources_after'] = fingerprints()
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
