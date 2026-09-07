"""Hardware trace composition gate for same-address, fixed-family attention replay."""

import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from attention_replay import ReplayAttentionReader
from gdn_multitoken_conv import addresses, release_owned


def main():
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch hardware required')
    import torch
    import ttnn
    from sdpa_tree_scratch import audit

    output_path = Path('/experiment/results/attention-replay.json')
    report = dict(passed=False, fixtures=[], scope='Read-only KV attention replay only; no full-model or committed-rate claim',
        native_sources=audit('/opt/tt-metal'), simulator_prerequisite='20260907T001643Z-681',
        mask_hardware_prerequisite=34068177963,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('attention_replay.py', 'attention_mask_replay.py', 'attention_mask_replay.cpp',
                              'attention_parallel.py', 'attention_fold_dma.py', 'attention_fold_dma.cpp')})
    mesh = None

    def stage(name, **details):
        report['last_stage'] = dict(name=name, **details)
        output_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['last_stage']), flush=True)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(value, dtype=dtype, device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(value):
            return [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(value)]

        for capacity in (4352, 16640):
            for rows in (8, 16, 32):
                for seed in (0, 1):
                    stage('fixture', capacity=capacity, rows=rows, seed=seed)
                    torch.manual_seed(seed)
                    blocks = capacity // 64
                    pages_host = torch.arange(blocks, dtype=torch.int32).flip(0).reshape(1, blocks)
                    query_host = torch.randn(1, rows, 12, 256).bfloat16()
                    queries = [query_host, -query_host, query_host.roll(1, dims=1), query_host]
                    starts = [capacity - 256, capacity - rows, capacity - 249, capacity - 256]
                    inputs, expected = [], []
                    reader, captured = None, None
                    results = []
                    try:
                        keys = upload(torch.randn(blocks, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
                        inputs.append(keys)
                        values = upload(torch.randn(blocks, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
                        inputs.append(values)
                        pages = upload(pages_host, ttnn.int32)
                        inputs.append(pages)
                        query = upload(query_host)
                        inputs.append(query)
                        original_cache = [host(value) for value in (keys, values)]
                        for start, ticket_query in zip(starts, queries, strict=True):
                            outputs = []
                            for token in range(rows):
                                token_query = upload(ticket_query[:, token:token + 1].contiguous())
                                position = upload(torch.tensor([start + token], dtype=torch.int32), ttnn.int32)
                                result = None
                                try:
                                    result = ttnn.transformer.paged_scaled_dot_product_attention_decode(token_query, keys, values,
                                        page_table_tensor=pages, cur_pos_tensor=position, scale=0.0625,
                                        program_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
                                    outputs.append(host(result))
                                finally:
                                    release_owned(ttnn, [token_query, position] + ([result] if result is not None else []))
                            expected.append([torch.cat([value[chip] for value in outputs], dim=1) for chip in range(2)])
                        reader = ReplayAttentionReader(ttnn, mesh, rows, capacity, pages_host, upload)
                        warm = reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)
                        ttnn.synchronize_device(mesh)
                        if any(not torch.equal(actual, target) for actual, target in zip(host(warm), expected[0], strict=True)):
                            raise AssertionError('Warm reader differs from native B1')
                        ttnn.deallocate(warm)
                        captured, result = capture_operation(ttnn, mesh,
                            lambda: reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG))
                        results.append(result)
                        before = [addresses(ttnn, value) for value in inputs + reader.owned]
                        checks = []
                        for ticket, (start, ticket_query, target) in enumerate(zip(starts, queries, expected, strict=True)):
                            reader.stage(start)
                            staged = ttnn.from_torch(ticket_query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                            ttnn.copy_host_to_device_tensor(staged, query)
                            ttnn.execute_trace(mesh, captured, cq_id=0, blocking=True)
                            for chip, (actual, reference) in enumerate(zip(host(result), target, strict=True)):
                                if not torch.equal(actual, reference):
                                    raise AssertionError('Changed query/position trace differs from native B1')
                                checks.append(dict(ticket=ticket, start=start, chip=chip, exact=True))
                            if before != [addresses(ttnn, value) for value in inputs + reader.owned]:
                                raise AssertionError('Replay replaced a captured input or mask address')
                        if any(not torch.equal(before_chip, after_chip) for tensor, before_tensor in zip((keys, values), original_cache, strict=True)
                               for before_chip, after_chip in zip(before_tensor, host(tensor), strict=True)):
                            raise AssertionError('Read-only attention modified K/V cache')
                        report['fixtures'].append(dict(capacity=capacity, rows=rows, seed=seed, checks=checks,
                            cache_unchanged=True, trace_replays=4, exact=True))
                    finally:
                        if captured is not None:
                            ttnn.release_trace(mesh, captured)
                        release_owned(ttnn, results)
                        if reader is not None:
                            reader.close()
                        release_owned(ttnn, inputs)
        if len(report['fixtures']) != 12 or sum(len(item['checks']) for item in report['fixtures']) != 96:
            raise AssertionError('Incomplete attention replay matrix')
        report['passed'] = True
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        output_path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
