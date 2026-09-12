"""Experimental T8 short-context folded attention versus exact native B1 SDPA."""

import argparse
import hashlib
import json
import os
from pathlib import Path

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
    import torch
    import ttnn

    report = dict(passed=False, backend='hardware' if options.hardware else 'simulator',
        scope='Short-context T8 attention component, including CTX170; not full-model correctness or TG',
        checks=[], mask_checks=[], source_checks=[], stale_controls=0, mask_poison_controls=0,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('attention_replay.py', 'attention_mask_replay.py', 'attention_mask_replay.cpp',
                              'attention_parallel.py', 'attention_fold_dma.py', 'attention_fold_dma.cpp')})
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

        for capacity in (256, 512, 768):
            torch.manual_seed(capacity)
            first, rows = max(128, capacity - 256), 8
            starts = [first, 170 if capacity == 256 else first + 17, capacity - rows, first]
            initial = torch.randn(1, rows, 12, 256).bfloat16()
            queries = [initial, -initial, initial.roll(1, dims=1), initial]
            pages_host = torch.arange(capacity // 64, dtype=torch.int32).flip(0).reshape(1, -1)
            owned, gold = [], []
            reader, trace = None, None
            try:
                keys = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
                owned.append(keys)
                values = upload(torch.randn(capacity // 64, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
                owned.append(values)
                pages, query = upload(pages_host, ttnn.int32), upload(initial)
                owned.extend((pages, query))
                original_cache = [host(tensor) for tensor in (keys, values)]
                for start, ticket_query in zip(starts, queries, strict=True):
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
                reader = ReplayAttentionReader(ttnn, mesh, rows, capacity, pages_host, upload, short_context=True)
                warm = reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)
                try:
                    if any(not torch.equal(actual, expected) for actual, expected in zip(host(warm), gold[0], strict=True)):
                        raise AssertionError('Short-context warm output differs from native B1')
                finally:
                    ttnn.deallocate(warm)
                trace, output = capture_operation(ttnn, mesh,
                    lambda: reader(query, keys, values, scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG))
                owned.append(output)
                original_ids = [addresses(ttnn, tensor) for tensor in owned + reader.owned]
                for ticket, (start, ticket_query, expected) in enumerate(zip(starts, queries, gold, strict=True)):
                    reader.stage(start)
                    staged = ttnn.from_torch(ticket_query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    ttnn.copy_host_to_device_tensor(staged, query)
                    for _, _, mask, _ in reader.metadata:
                        poison = torch.full(tuple(mask.shape), float('nan'), dtype=torch.bfloat16)
                        staged_mask = ttnn.from_torch(poison, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                        ttnn.copy_host_to_device_tensor(staged_mask, mask)
                        report['mask_poison_controls'] += 1
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for chip, (actual, reference) in enumerate(zip(host(output), expected, strict=True)):
                        if not torch.equal(actual, reference):
                            raise AssertionError(f'Short replay differs from native B1: capacity={capacity}, start={start}, chip={chip}')
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
                                raise AssertionError('Captured short-context causal mask differs from host oracle')
                            report['mask_checks'].append(dict(capacity=capacity, start=start, ticket=ticket, chip=chip, exact=True))
                    if original_ids != [addresses(ttnn, tensor) for tensor in owned + reader.owned]:
                        raise AssertionError('Captured short-context buffers moved')
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
        report['passed'] = (len(report['checks']) == 24 and len(report['mask_checks']) == 24
                            and len(report['source_checks']) == 12 and report['stale_controls'] == 6
                            and report['mask_poison_controls'] == 12)
        if not report['passed']:
            raise AssertionError('Incomplete short-context attention gate')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
