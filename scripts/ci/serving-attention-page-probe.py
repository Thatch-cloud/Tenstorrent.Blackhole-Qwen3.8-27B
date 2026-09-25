"""T16 captured attention follows appended scheduler pages at the 4K frontier."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from attention_batch import capture_operation
from attention_replay import ReplayAttentionReader
from gdn_multitoken_conv import release_owned
from serving_page_binding import VerifierPageBinding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output_path = parser.parse_args().output
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or output_path.exists()):
        raise ValueError('Fresh device-free simulator required')
    import torch
    import ttnn

    names = ('serving-attention-page-probe.py', 'serving_page_binding.py', 'attention_replay.py',
        'attention_parallel.py', 'attention_mask_replay.py', 'attention_mask_replay.cpp',
        'attention_fold_dma.py', 'attention_fold_dma.cpp')
    report = dict(passed=False, closed_cleanly=False, checks=[], stale_controls=0,
        backend='simulator', context=4096, rows=16, model_integrated=False, performance_qualified=False,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names})
    mesh, reader, trace, owned = None, None, None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(value, device=mesh, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def stage(value, destination, dtype):
            host = ttnn.from_torch(value, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(host, destination)

        def host(tensor):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            return [ttnn.to_torch(shard).clone() for shard in shards]

        blocks = tuple(range(67, 2, -1))
        states = ((blocks, 4096), ((*blocks, 2), 4160))
        pages_host = torch.full((1, 68), blocks[0], dtype=torch.int32)
        pages_host[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        torch.manual_seed(7)
        keys = upload(torch.randn(68, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
        values_host = torch.randn(68, 2, 64, 256).bfloat16() * 0.1
        values_host[2] = 16
        values = upload(values_host, ttnn.bfloat8_b)
        query_host = torch.randn(1, 16, 12, 256).bfloat16()
        query = upload(query_host)
        primary, singleton = upload(pages_host.repeat(16, 1), ttnn.int32), upload(pages_host, ttnn.int32)
        owned.extend((keys, values, query, primary, singleton))
        original_cache = (host(keys), host(values))
        grid = mesh.compute_with_storage_grid_size()
        config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)
        references = []
        for allocation, start in states:
            table = torch.full((1, 68), allocation[0], dtype=torch.int32)
            table[0, :len(allocation)] = torch.tensor(allocation, dtype=torch.int32)
            stage(table, singleton, ttnn.int32)
            outputs = []
            for row in range(16):
                row_query = upload(query_host[:, row:row + 1].contiguous())
                position = upload(torch.tensor([start + row], dtype=torch.int32), ttnn.int32)
                result = None
                try:
                    result = ttnn.transformer.paged_scaled_dot_product_attention_decode(row_query, keys, values,
                        page_table_tensor=singleton, cur_pos_tensor=position, scale=0.0625,
                        program_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
                    outputs.append(host(result))
                finally:
                    release_owned(ttnn, [row_query, position] + ([result] if result is not None else []))
            references.append([torch.cat([entry[chip] for entry in outputs], dim=1) for chip in range(2)])
            print(json.dumps(dict(stage='native-reference', start=start)), flush=True)
        stage(pages_host, singleton, ttnn.int32)
        reader = ReplayAttentionReader(ttnn, mesh, 16, 4352, pages_host, upload)
        fixture = SimpleNamespace(pages=primary, singleton_pages=singleton, replay_reader=reader,
            grouped_readers=[reader], readers=[reader], writers=[])
        engine = SimpleNamespace(phase='idle', pages=pages_host.clone(), mesh=mesh, operations=ttnn,
            buckets={16: dict(fixture=fixture)})
        binding = VerifierPageBinding(engine, blocks, physical_pages=68)
        operation = lambda: reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)
        warm = operation()
        try:
            if any(not torch.equal(actual, expected) for actual, expected in zip(host(warm), references[0])):
                raise AssertionError('Eager T16 differs from native serial reference')
        finally:
            ttnn.deallocate(warm)
        trace, result = capture_operation(ttnn, mesh, operation)
        owned.append(result)
        for ordinal, ((allocation, start), reference) in enumerate(zip(states, references)):
            binding.refresh(allocation, position=start, rows=16)
            reader.stage(start)
            stage(torch.full(tuple(result.shape), float('nan'), dtype=torch.bfloat16), result, ttnn.bfloat16)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for chip, (actual, expected) in enumerate(zip(host(result), reference)):
                exact = torch.equal(actual, expected)
                report['checks'].append(dict(ordinal=ordinal, chip=chip, exact=exact))
                if not exact:
                    raise AssertionError('Page-updated captured T16 differs from serial attention')
                if ordinal == 1:
                    if torch.equal(actual, references[0][chip]):
                        raise AssertionError('Changed page produced stale output')
                    report['stale_controls'] += 1
            print(json.dumps(dict(stage='replay', start=start, exact=True)), flush=True)
        for tensor, snapshots in zip((keys, values), original_cache):
            if any(not torch.equal(actual, expected) for actual, expected in zip(host(tensor), snapshots)):
                raise AssertionError('Attention changed its KV cache')
        report['passed'] = True
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            if reader is not None:
                reader.close()
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        output_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
