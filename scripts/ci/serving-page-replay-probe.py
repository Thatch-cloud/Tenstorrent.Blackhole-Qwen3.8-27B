"""Weight-free page binding and captured native cache-write qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from serving_page_binding import VerifierPageBinding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or output.exists()):
        raise ValueError('Fresh device-free simulator required')
    import torch
    import ttnn

    report = dict(passed=False, closed_cleanly=False, checks=[], backend='simulator',
        model_integrated=False, attention_reader_qualified=False, performance_qualified=False,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('serving-page-replay-probe.py', 'serving_page_binding.py')})
    mesh, trace, owned = None, None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()

        def upload(value, dtype, device=True):
            options = dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}
            result = ttnn.from_torch(value, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), **options)
            if device:
                owned.append(result)
            return result

        pages_host = torch.full((1, 68), 3, dtype=torch.int32)
        primary = upload(pages_host.repeat(16, 1), ttnn.int32)
        singleton = upload(pages_host, ttnn.int32)
        reader_pages = upload(pages_host.repeat(2, 1), ttnn.int32)
        reader = SimpleNamespace(metadata=[(None, reader_pages, None, None)], audit=None)
        fixture = SimpleNamespace(pages=primary, singleton_pages=singleton, replay_reader=reader,
            grouped_readers=[reader], readers=[reader], writers=[SimpleNamespace(pages=[singleton])])
        engine = SimpleNamespace(phase='idle', pages=pages_host.clone(), mesh=mesh, operations=ttnn,
            buckets={16: dict(fixture=fixture)})
        binding = VerifierPageBinding(engine, (3,), physical_pages=8)
        expected = torch.zeros((8, 2, 64, 256), dtype=torch.bfloat16)
        cache = upload(expected, ttnn.bfloat8_b)
        position = upload(torch.tensor([0], dtype=torch.int32), ttnn.int32)
        source = upload(torch.ones((1, 1, 32, 256), dtype=torch.bfloat16), ttnn.bfloat16)
        layout = ttnn.create_sharded_memory_config([32, 256], ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        sharded = ttnn.to_memory_config(source, layout)
        owned.append(sharded)

        def operation():
            ttnn.experimental.paged_update_cache(cache, sharded,
                update_idxs_tensor=position, page_table=singleton)

        def check(ordinal):
            shards = ttnn.get_device_tensors(cache)
            if len(shards) != 2:
                raise AssertionError('Both simulated chips required')
            for chip, shard in enumerate(shards):
                exact = torch.equal(ttnn.to_torch(shard), expected)
                report['checks'].append(dict(ordinal=ordinal, chip=chip, name='complete_cache', exact=exact))
                if not exact:
                    raise AssertionError('Captured write used stale page mapping')
            for name, tensor in (('primary', primary), ('singleton', singleton), ('reader', reader_pages)):
                reference = engine.pages.repeat(tensor.shape[0], 1)
                for chip, shard in enumerate(ttnn.get_device_tensors(tensor)):
                    exact = torch.equal(ttnn.to_torch(shard), reference)
                    report['checks'].append(dict(ordinal=ordinal, chip=chip, name=name, exact=exact))
                    if not exact:
                        raise AssertionError('Captured metadata upload mismatch')
            print(json.dumps(dict(stage='checked', ordinal=ordinal, checks=len(report['checks']))), flush=True)
            output.write_text(json.dumps(report, indent=2) + '\n')

        operation()
        ttnn.synchronize_device(mesh)
        expected[3, :, 0, :] = 1
        check(-1)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            operation()
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for ordinal, (blocks, index) in enumerate((((3,), 0), ((3, 6), 64), ((3, 6, 2), 128), ((3, 6, 2), 129))):
            binding.refresh(blocks, position=index, rows=1)
            ttnn.copy_host_to_device_tensor(upload(torch.tensor([index], dtype=torch.int32), ttnn.int32, False), position)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            expected[blocks[index // 64], :, index % 64, :] = 1
            check(ordinal)
        report['passed'] = True
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            for tensor in reversed(owned):
                ttnn.deallocate(tensor)
            ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
