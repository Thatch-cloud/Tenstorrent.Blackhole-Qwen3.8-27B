"""Static TP2 attention groups versus serial SDPA; includes device layout costs."""

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from attention_batch import SerialAttentionReader, capture_operation
from attention_head_fold import causal_mask, chunk_groups, device_layout
from gdn_multitoken_conv import addresses, release_owned


def validate_matrix(fixtures):
    expected = {(seed, rows, start) for seed in (0, 1, 2) for rows in (1, 2, 4, 8, 16, 32)
                for start in (4095, 16383)}
    if len(fixtures) != len(expected) or {(value['seed'], value['rows'], value['start']) for value in fixtures} != expected:
        raise AssertionError('Incomplete grouped attention matrix')
    if any(value['exact'] is not True or value['timed_replays'] != 120 or value['refreshed_checks'] != 2 for value in fixtures):
        raise AssertionError('Missing correctness, timing or changed-input checks')
    for value in fixtures:
        samples = value['samples']
        if [sample['arm'] for sample in samples] != ['control', 'candidate', 'candidate', 'control'] * 3:
            raise AssertionError('Incomplete ABBA samples')
        if any(len(sample['replay_ms']) != 10 or any(type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0
               for cost in sample['replay_ms']) for sample in samples):
            raise AssertionError('Finite positive raw replay samples required')


def main():
    if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
        raise RuntimeError('Explicit hardware allocation required')
    if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch hardware required')
    import torch
    import ttnn

    path = Path('/experiment/results/attention-group-timing.json')
    report = dict(passed=False, fixtures=[], simulator_prerequisites=['20260906T211544Z-314', '20260906T211739Z-300'],
        scope='Synthetic static attention with device packing/unpacking; masks and page views prepared outside timing; no model throughput',
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('attention-group-timing.py', 'attention_head_fold.py', 'attention_batch.py')})
    mesh = None

    def stage(name, **details):
        report['last_stage'] = dict(name=name, **details)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['last_stage']), flush=True)

    try:
        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        grid = mesh.compute_with_storage_grid_size()
        config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False,
                                      q_chunk_size=0, k_chunk_size=0)
        for start in (4095, 16383):
            for rows in (1, 2, 4, 8, 16, 32):
                for seed in (0, 1, 2):
                    stage('fixture', start=start, rows=rows, seed=seed)
                    inputs, traces, outputs, temporaries = [], {}, {}, {}
                    try:
                        def upload(value, dtype=ttnn.bfloat16):
                            result = ttnn.from_torch(value, device=mesh, dtype=dtype,
                                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                            inputs.append(result)
                            return result

                        def host(value):
                            parts = ttnn.get_device_tensors(value)
                            if len(parts) != 2:
                                raise AssertionError('Two-chip output required')
                            return [ttnn.to_torch(part).clone() for part in parts]

                        def release_scratch(owned):
                            protected = {addresses(ttnn, tensor) for tensor in inputs}
                            release_owned(ttnn, [tensor for tensor in owned if addresses(ttnn, tensor) not in protected])

                        torch.manual_seed(seed)
                        groups = chunk_groups(start, rows)
                        capacity = groups[-1]['signature'][1]
                        blocks = capacity // 64
                        query_host = torch.randn(1, rows, 12, 256).bfloat16()
                        changed_host = torch.randn_like(query_host)
                        query, changed = upload(query_host), upload(changed_host)
                        keys = upload(torch.randn(blocks, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
                        values = upload(torch.randn(blocks, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
                        immutable = (host(keys), host(values))
                        page_ids = torch.arange(blocks, dtype=torch.int32).flip(0).reshape(1, blocks)
                        pages = upload(page_ids, ttnn.int32)
                        positions = [upload(torch.tensor([start + offset], dtype=torch.int32), ttnn.int32) for offset in range(rows)]
                        metadata = [(group, upload(page_ids[:, :group['signature'][1] // 64].contiguous(), ttnn.int32),
                                     upload(causal_mask(group['rows'], start + group['offset'], group['signature'][1])))
                                    for group in groups]

                        def gold(source):
                            tokens = []
                            for offset in range(rows):
                                row = upload(source[:, offset:offset + 1].contiguous())
                                result = ttnn.transformer.paged_scaled_dot_product_attention_decode(row, keys, values,
                                    page_table_tensor=pages, cur_pos_tensor=positions[offset], scale=0.0625,
                                    program_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
                                try:
                                    tokens.append(host(result))
                                finally:
                                    ttnn.deallocate(result)
                            return [torch.cat([token[chip] for token in tokens], dim=1) for chip in range(2)]

                        expected, refreshed = gold(query_host), gold(changed_host)
                        if any(torch.equal(first, second) for first, second in zip(expected, refreshed, strict=True)):
                            raise AssertionError('Changed-query fixture must change native output')
                        reader = SerialAttentionReader(ttnn, positions, [pages] * rows)

                        def operation(arm, owned):
                            if arm == 'control':
                                if rows == 1:
                                    result = ttnn.transformer.paged_scaled_dot_product_attention_decode(query, keys, values,
                                        page_table_tensor=pages, cur_pos_tensor=positions[0], scale=0.0625,
                                        program_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
                                else:
                                    result = reader(query, keys, values, page_table_tensor=pages, cur_pos_tensor=positions[0],
                                        scale=0.0625, program_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
                                owned.append(result)
                                return result
                            chunks = []
                            for group, group_pages, mask in metadata:
                                count = group['rows']
                                packed = device_layout(ttnn, query, count, owned, offset=group['offset'])
                                result = ttnn.transformer.paged_scaled_dot_product_attention_decode(packed, keys, values,
                                    page_table_tensor=group_pages, is_causal=False, attn_mask=mask, scale=0.0625,
                                    program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                                        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=group['signature'][0]),
                                    memory_config=ttnn.L1_MEMORY_CONFIG)
                                owned.append(result)
                                chunks.append(device_layout(ttnn, result, count, owned, inverse=True))
                            result = ttnn.concat(chunks, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
                            owned.append(result)
                            return result

                        def check(result, reference):
                            if any(not torch.equal(actual, target) for actual, target in zip(host(result), reference, strict=True)):
                                raise AssertionError('Grouped attention differs from independent native B1')

                        for arm in ('control', 'candidate'):
                            warm = []
                            try:
                                check(operation(arm, warm), expected)
                            finally:
                                ttnn.synchronize_device(mesh)
                                release_scratch(warm)
                        for arm in ('control', 'candidate'):
                            temporaries[arm] = []
                            traces[arm], outputs[arm] = capture_operation(ttnn, mesh,
                                lambda arm=arm: operation(arm, temporaries[arm]))
                            ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=True)
                            check(outputs[arm], expected)
                        samples = []
                        for block in range(3):
                            for arm in ('control', 'candidate', 'candidate', 'control'):
                                elapsed = []
                                for repeat in range(10):
                                    ttnn.synchronize_device(mesh)
                                    begun = time.perf_counter()
                                    ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=True)
                                    elapsed.append((time.perf_counter() - begun) * 1000)
                                check(outputs[arm], expected)
                                samples.append(dict(block=block, arm=arm, replay_ms=elapsed, mean_ms=statistics.mean(elapsed)))
                        ttnn.copy(changed, query)
                        ttnn.synchronize_device(mesh)
                        for arm in ('control', 'candidate'):
                            ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=True)
                            check(outputs[arm], refreshed)
                        if any(not torch.equal(before, after) for tensor, saved in zip((keys, values), immutable, strict=True)
                               for before, after in zip(saved, host(tensor), strict=True)):
                            raise AssertionError('Read-only attention changed KV data')
                        report['fixtures'].append(dict(start=start, rows=rows, seed=seed, groups=groups, exact=True,
                            timed_replays=120, refreshed_checks=2, samples=samples))
                        stage('fixture-passed', start=start, rows=rows, seed=seed)
                    finally:
                        for trace in traces.values():
                            ttnn.release_trace(mesh, trace)
                        release_scratch([tensor for owned in temporaries.values() for tensor in owned])
                        release_owned(ttnn, inputs)
        validate_matrix(report['fixtures'])
        report['passed'] = True
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
