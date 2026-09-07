"""Static TP2 attention groups versus serial SDPA; includes device layout costs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from attention_batch import SerialAttentionReader, capture_operation
from attention_head_fold import causal_mask, chunk_groups, device_layout, parallel_groups
from gdn_multitoken_conv import addresses, release_owned


def validate_matrix(fixtures, *, tree_scratch=False, parallel=False, tree_parallel=False):
    expected = {(seed, rows, start) for seed in (0, 1, 2) for rows in (1, 2, 4, 8, 16, 32)
                for start in (4095, 16383)}
    if len(fixtures) != len(expected) or {(value['seed'], value['rows'], value['start']) for value in fixtures} != expected:
        raise AssertionError('Incomplete grouped attention matrix')
    if any(value['exact'] is not True or value['timed_replays'] != 120 or value['refreshed_checks'] != 2 for value in fixtures):
        raise AssertionError('Missing correctness, timing or changed-input checks')
    for value in fixtures:
        if parallel and json.dumps(value.get('parallel_plan'), sort_keys=True) != json.dumps(
                parallel_groups(value['start'], value['rows']), sort_keys=True):
            raise AssertionError('Complete order-preserving parallel groups required')
        if tree_parallel:
            for key, limit in (('control_parallel_plan', 4), ('parallel_plan', 8)):
                if json.dumps(value.get(key), sort_keys=True) != json.dumps(
                        parallel_groups(value['start'], value['rows'], max_group_rows=limit), sort_keys=True):
                    raise AssertionError('Matched parallel T4 and T8 plans required')
        if tree_scratch or tree_parallel:
            for key, limit in (('groups', 4), ('candidate_groups', 8)):
                actual = [(group['offset'], group['rows'], tuple(group['signature'])) for group in value.get(key, [])]
                expected_plan = [(group['offset'], group['rows'], group['signature'])
                    for group in chunk_groups(value['start'], value['rows'], max_group_rows=limit)]
                if actual != expected_plan:
                    raise AssertionError('Tree experiment must compare complete T4 and T8 group plans')
        samples = value['samples']
        if [sample['arm'] for sample in samples] != ['control', 'candidate', 'candidate', 'control'] * 3:
            raise AssertionError('Incomplete ABBA samples')
        if any(len(sample['replay_ms']) != 10 or any(type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0
               for cost in sample['replay_ms']) for sample in samples):
            raise AssertionError('Finite positive raw replay samples required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dma-layout', action='store_true')
    parser.add_argument('--tree-scratch', action='store_true')
    parser.add_argument('--parallel-groups', action='store_true')
    parser.add_argument('--tree-parallel', action='store_true')
    options = parser.parse_args()
    if sum((options.tree_scratch, options.dma_layout, options.parallel_groups, options.tree_parallel)) > 1:
        parser.error('Tree, layout DMA and parallel grouping are separate experiments')
    if (options.tree_scratch or options.tree_parallel) and os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':
        raise RuntimeError('Tree experiment requires the process-fixed native scratch override')
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
    report['dma_layout'] = options.dma_layout
    report['tree_scratch'] = options.tree_scratch
    report['parallel_groups'] = options.parallel_groups
    report['tree_parallel'] = options.tree_parallel
    if options.tree_parallel:
        from sdpa_tree_scratch import audit
        report['native_sources'] = audit('/opt/tt-metal', patched=True)
        report['scope'] = 'Parallel T4 versus T8 groups, both with compact native scratch and DMA layout; no model throughput'
        report['simulator_prerequisites'] = ['20260907T011925Z-426', '20260907T012041Z-303']
        report['worker_cap_per_head'] = 16
        report['maximum_sdpa_workers'] = 96
        report['sources'].update({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('attention_parallel.py', 'attention_fold_dma.py', 'attention_fold_dma.cpp')})
    if options.parallel_groups:
        from sdpa_tree_scratch import audit
        report['native_sources'] = audit('/opt/tt-metal')
        report['scope'] = 'Serial versus up to three parallel four-query groups, both with device DMA layout; one stream verification, no model throughput'
        report['simulator_prerequisites'] = ['20260906T224631Z-303', '20260906T225115Z-307', '20260906T225312Z-461']
        report['worker_cap_per_head'] = 16
        report['maximum_sdpa_workers'] = 96
        report['sources'].update({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('attention_parallel.py', 'attention_fold_dma.py', 'attention_fold_dma.cpp')})
    if options.tree_scratch:
        from sdpa_tree_scratch import audit
        report['native_sources'] = audit('/opt/tt-metal', patched=True)
        report['scope'] = 'Matched grouped attention T4 versus T8 groups; both use compact tree scratch and stock device layout; no model throughput'
        report['simulator_prerequisites'] = ['20260906T221933Z-382', '20260906T222729Z-298',
                                            '20260906T223545Z-304', '20260906T223905Z-308']
        report['worker_cap_per_head'] = 16
    if options.dma_layout:
        report['simulator_prerequisites'] = ['20260906T215257Z-312', '20260906T215411Z-605', '20260906T220602Z-302']
        report['scope'] = 'Matched grouped attention: stock layout chain versus direct tile DMA; synthetic static inputs, no model throughput'
        report['sources'].update({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('attention_fold_dma.py', 'attention_fold_dma.cpp')})
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
                        candidate_groups = chunk_groups(start, rows, max_group_rows=8) if (options.tree_scratch or options.tree_parallel) else groups
                        candidate_metadata = [(group, upload(page_ids[:, :group['signature'][1] // 64].contiguous(), ttnn.int32),
                            upload(causal_mask(group['rows'], start + group['offset'], group['signature'][1])))
                            for group in candidate_groups] if options.tree_scratch else metadata
                        parallel_metadata, plans = {}, {}
                        if options.parallel_groups or options.tree_parallel:
                            for arm, limit in (('control', 1), ('candidate', 3)):
                                plans[arm] = parallel_groups(start, rows,
                                    max_batches=3 if options.tree_parallel else limit,
                                    max_group_rows=8 if options.tree_parallel and arm == 'candidate' else 4)
                                parallel_metadata[arm] = []
                                for bundle in plans[arm]:
                                    chunk_size, extent = bundle[0]['signature']
                                    group_pages = page_ids[:, :extent // 64].repeat(len(bundle), 1).contiguous()
                                    mask = torch.cat([causal_mask(group['rows'], start + group['offset'], extent)
                                                      for group in bundle], dim=0)
                                    config_group = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                                        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=chunk_size)
                                    parallel_metadata[arm].append((bundle, upload(group_pages, ttnn.int32), upload(mask), config_group))

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
                            if options.parallel_groups or options.tree_parallel:
                                from attention_parallel import execute
                                return execute(mesh, ttnn, query, keys, values, parallel_metadata[arm], owned,
                                    scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)
                            if arm == 'control' and not (options.dma_layout or options.tree_scratch):
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
                            def layout(tensor, count, *, inverse=False, offset=0):
                                if options.dma_layout and arm == 'candidate':
                                    from attention_fold_dma import device_layout_dma
                                    return device_layout_dma(mesh, tensor, count, owned, inverse=inverse, offset=offset)
                                return device_layout(ttnn, tensor, count, owned, inverse=inverse, offset=offset)

                            selected_metadata = candidate_metadata if arm == 'candidate' else metadata
                            for group, group_pages, mask in selected_metadata:
                                count = group['rows']
                                packed = layout(query, count, offset=group['offset'])
                                result = ttnn.transformer.paged_scaled_dot_product_attention_decode(packed, keys, values,
                                    page_table_tensor=group_pages, is_causal=False, attn_mask=mask, scale=0.0625,
                                    program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                                        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=group['signature'][0]),
                                    memory_config=ttnn.L1_MEMORY_CONFIG)
                                owned.append(result)
                                chunks.append(layout(result, count, inverse=True))
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
                        report['fixtures'].append(dict(start=start, rows=rows, seed=seed, groups=groups,
                            control_parallel_plan=plans.get('control'),
                            parallel_plan=plans.get('candidate'),
                            candidate_groups=candidate_groups, exact=True,
                            timed_replays=120, refreshed_checks=2, samples=samples))
                        stage('fixture-passed', start=start, rows=rows, seed=seed)
                    finally:
                        for trace in traces.values():
                            ttnn.release_trace(mesh, trace)
                        release_scratch([tensor for owned in temporaries.values() for tensor in owned])
                        release_owned(ttnn, inputs)
        validate_matrix(report['fixtures'], tree_scratch=options.tree_scratch, parallel=options.parallel_groups,
            tree_parallel=options.tree_parallel)
        report['passed'] = True
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
