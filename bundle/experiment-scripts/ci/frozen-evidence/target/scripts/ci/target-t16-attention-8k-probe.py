"""Experimental T16 long-context folded attention versus exact native B1 SDPA."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from frozen_context_geometry import selected_geometry

from attention_batch import capture_operation
from attention_head_fold import causal_mask
from attention_replay import ReplayAttentionReader
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    from attention_mask_replay import validate_ticket
    capacity = selected_geometry()['capacity']
    for start in (capacity - 256, capacity - 239, capacity - 16):
        validate_ticket(start, 16, capacity, short_context=False)
    import torch
    import ttnn

    report = dict(passed=False, backend='hardware' if options.hardware else 'simulator',
        scope='Selected-context BF16 KV target replay; not full-model correctness or TG',
        context=selected_geometry()['context'], kv_dtype='bfloat16',
        checks=[], mask_checks=[], source_checks=[], stale_controls=0, mask_poison_controls=0,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('attention_replay.py', 'attention_mask_replay.py', 'attention_mask_replay.cpp',
                              'attention_parallel.py', 'attention_fold_dma.py', 'attention_fold_dma.cpp', 'target-t16-attention-8k-probe.py', 'frozen_context_geometry.py')})
    scratch = os.environ.get('QWEN_FROZEN_TARGET_SCRATCH', '0')
    if scratch not in ('0', '1'):
        raise ValueError('Explicit scratch variant required')
    report['compact_tree_scratch'] = scratch == '1'
    if scratch == '1':
        from sdpa_tree_scratch import audit
        from dspark_fp32_build import validate_manifest
        if os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':
            raise ValueError('Compiled scratch candidate must be explicitly enabled')
        report['native_sources'] = audit('/opt/tt-metal', patched=True)
        report['factory_build'] = validate_manifest('/opt/tt-metal', '/experiment/results/dspark-fp32-build.json')
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        grid = mesh.compute_with_storage_grid_size()
        config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(value, device=mesh, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(tensor):
            parts = ttnn.get_device_tensors(tensor)
            if len(parts) != 2:
                raise AssertionError('Two independent chip results required')
            return [ttnn.to_torch(part).clone() for part in parts]

        for capacity in (selected_geometry()['capacity'],):
            torch.manual_seed(capacity)
            first, rows = capacity - 256, 16
            starts = [first, first + 17, capacity - rows, first]
            initial = torch.randn(1, rows, 12, 256).bfloat16()
            queries = [initial, -initial, initial.roll(1, dims=1), initial]
            pages_host = torch.arange(capacity // 64, dtype=torch.int32).flip(0).reshape(1, -1)
            owned, gold = [], []
            reader, trace = None, None
            try:
                keys = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat16)
                owned.append(keys)
                values = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat16)
                owned.append(values)
                pages, query = upload(pages_host, ttnn.int32), upload(initial)
                owned.extend((pages, query))
                original_cache = [host(tensor) for tensor in (keys, values)]
                reader = ReplayAttentionReader(ttnn, mesh, rows, capacity, pages_host, upload, short_context=False)
                allocation_check = reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)
                allocation_host = host(allocation_check)
                ttnn.deallocate(allocation_check)
                print(json.dumps(dict(stage='target-allocation-ready', capacity=capacity)), flush=True)
                for start, ticket_query in zip(starts, queries, strict=True):
                    if gold and start == starts[0] and torch.equal(ticket_query, queries[0]):
                        gold.append([value.clone() for value in gold[0]])
                        print(json.dumps(dict(stage='native-reference-reused', capacity=capacity, start=start)), flush=True)
                        continue
                    outputs = []
                    for row in range(rows):
                        row_query = upload(ticket_query[:, row:row + 1].contiguous())
                        position = upload(torch.tensor([start + row], dtype=torch.int32), ttnn.int32)
                        output = None
                        try:
                            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(row_query, keys, values,
                                page_table_tensor=pages, cur_pos_tensor=position, scale=0.0625,
                                program_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
                            outputs.append(host(output))
                        finally:
                            release_owned(ttnn, [row_query, position] + ([output] if output is not None else []))
                    gold.append([torch.cat([value[chip] for value in outputs], dim=1) for chip in range(2)])
                    print(json.dumps(dict(stage='native-reference', capacity=capacity, start=start)), flush=True)
                if any(not torch.equal(actual, expected) for actual, expected in zip(allocation_host, gold[0], strict=True)):
                    raise AssertionError('T16 long-context warm output differs from native B1')
                trace, output = capture_operation(ttnn, mesh,
                    lambda: reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG))
                owned.append(output)
                original_ids = [addresses(ttnn, tensor) for tensor in owned + reader.owned]
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                report['unpoisoned_replay'] = [dict(chip=chip, exact=torch.equal(actual, expected),
                    nonfinite=int((~torch.isfinite(actual)).sum()), mismatches=int((actual != expected).sum()))
                    for chip, (actual, expected) in enumerate(zip(host(output), gold[0], strict=True))]
                options.output.write_text(json.dumps(report, indent=2))
                for ticket, (start, ticket_query, expected) in enumerate(zip(starts, queries, gold, strict=True)):
                    reader.stage(start)
                    staged = ttnn.from_torch(ticket_query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    ttnn.copy_host_to_device_tensor(staged, query)
                    for _, _, mask, _ in reader.metadata:
                        poison = torch.zeros(tuple(mask.shape), dtype=torch.bfloat16)
                        poison[..., capacity - 256:] = float('nan')
                        staged_mask = ttnn.from_torch(poison, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                        ttnn.copy_host_to_device_tensor(staged_mask, mask)
                        report['mask_poison_controls'] += 1
                    poisoned_output = ttnn.from_torch(torch.full(tuple(output.shape), float('nan')),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    ttnn.copy_host_to_device_tensor(poisoned_output, output)
                    if not all(bool(torch.isnan(value).all()) for value in host(output)):
                        raise AssertionError('Output poison not installed')
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for chip, (actual, reference) in enumerate(zip(host(output), expected, strict=True)):
                        if not torch.equal(actual, reference):
                            report['failure'] = dict(chip=chip, start=start, ticket=ticket,
                                mismatches=int((actual != reference).sum()), nonfinite=int((~torch.isfinite(actual)).sum()),
                                max_abs=float((actual.float() - reference.float()).abs().max()),
                                query_exact=[torch.equal(value, ticket_query) for value in host(query)],
                                mask_exact=[[torch.equal(value, torch.cat([causal_mask(group['rows'],
                                    start + group['offset'], capacity) for group in bundle], dim=0)) for value in host(mask)]
                                    for bundle, _, mask, _ in reader.metadata])
                            raise AssertionError(f'T16 replay differs from native B1: capacity={capacity}, start={start}, chip={chip}')
                        if ticket == 1:
                            if torch.equal(actual, gold[0][chip]):
                                raise AssertionError('Stale query/position output was not rejected')
                            report['stale_controls'] += 1
                        report['checks'].append(dict(capacity=capacity, start=start, ticket=ticket, chip=chip, exact=True))
                    for bundle, _, mask, _ in reader.metadata:
                        expected_mask = torch.cat([causal_mask(group['rows'], start + group['offset'], capacity)
                                                   for group in bundle], dim=0)
                        for chip, actual in enumerate(host(mask)):
                            if not torch.equal(actual, expected_mask):
                                raise AssertionError('Captured T16 long-context causal mask differs from host oracle')
                            report['mask_checks'].append(dict(capacity=capacity, start=start, ticket=ticket, chip=chip, exact=True))
                    if original_ids != [addresses(ttnn, tensor) for tensor in owned + reader.owned]:
                        raise AssertionError('Captured T16 long-context buffers moved')
                    options.output.write_text(json.dumps(report, indent=2))
                    print(json.dumps(dict(stage='replay', capacity=capacity, start=start, exact=True)), flush=True)
                for tensor, original in zip((keys, values), original_cache, strict=True):
                    for chip, (actual, expected) in enumerate(zip(host(tensor), original, strict=True)):
                        if not torch.equal(actual, expected):
                            raise AssertionError('Read-only short attention modified KV')
                        report['source_checks'].append(dict(capacity=capacity, chip=chip, exact=True))
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                if reader is not None:
                    reader.close()
                release_owned(ttnn, owned)
        report['passed'] = (len(report['checks']) == 8 and len(report['mask_checks']) == 16
                            and len(report['source_checks']) == 4 and report['stale_controls'] == 2
                            and report['mask_poison_controls'] == 8)
        if not report['passed']:
            raise AssertionError('Incomplete T16 long-context attention gate')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        if scratch == '1':
            report['native_sources_after'] = audit('/opt/tt-metal', patched=True)
        report['closed'] = mesh is not None
        report['sources_after'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in report['sources']}
        report['passed'] = report['passed'] and report['closed'] and report['sources'] == report['sources_after']
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
