"""Replay an exported real-query attention failure; no model or throughput claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import Overlay, capture_operation
from attention_replay import ReplayAttentionReader
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def load_fixture(path, expected_sha256):
    import torch

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Real-query fixture checksum changed')
    fixture = torch.load(path, weights_only=True, map_location='cpu')
    if set(fixture) - {'mask'} != {'query', 'keys', 'values', 'actual', 'expected', 'pages', 'position', 'scale', 'chip'}:
        raise ValueError('Complete frozen real-query fixture required')
    capacity = fixture['pages'].numel() * 64
    if capacity not in (256, 512, 768) or tuple(fixture['pages'].shape) != (1, capacity // 64):
        raise ValueError('Bounded short-context page table required')
    if fixture['pages'].dtype != torch.int32 or not torch.equal(
            fixture['pages'].sort().values, torch.arange(capacity // 64, dtype=torch.int32).reshape(1, -1)):
        raise ValueError('A complete permutation of the bounded physical pages required')
    for name in ('query', 'actual', 'expected'):
        if tuple(fixture[name].shape) != (1, 8, 12, 256) or fixture[name].dtype != torch.bfloat16:
            raise ValueError('Native BF16 T8 query/output geometry required')
    for name in ('keys', 'values'):
        if tuple(fixture[name].shape) != (capacity // 64, 2, 64, 256) or fixture[name].dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('Dequantized native BF8 cache geometry required')
    from attention_mask_replay import validate_ticket
    validate_ticket(fixture['position'], 8, capacity, short_context=True)
    if fixture['scale'] != 0.0625 or type(fixture['chip']) is not int or fixture['chip'] not in (0, 1):
        raise ValueError('Native Qwen scale and original physical chip required')
    if 'mask' in fixture and (tuple(fixture['mask'].shape) != (2, 1, 48, capacity)
                             or fixture['mask'].dtype != torch.bfloat16):
        raise ValueError('The exact native T8 mask geometry is required')
    return fixture, capacity


def comparison(actual, expected):
    import torch

    return dict(exact=torch.equal(actual, expected), differing_elements=int((actual != expected).sum()),
                max_absolute_error=float((actual.float() - expected.float()).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    fixture, capacity = load_fixture(options.fixture, options.sha256)
    import torch
    import ttnn

    report = dict(passed=False, diagnostic_only=True, throughput_claim=False,
                  fixture_sha256=options.sha256, capacity=capacity, position=fixture['position'],
                  backend='hardware' if options.hardware else 'simulator', checks=[], hardware_reference=[],
                  original_difference=comparison(fixture['actual'], fixture['expected']),
                  sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ('real-attention-probe.py', 'attention_replay.py', 'attention_parallel.py',
                                   'attention_mask_replay.py', 'attention_mask_replay.cpp',
                                   'attention_fold_dma.py', 'attention_fold_dma.cpp')})
    mesh = None
    owned = []
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
                raise AssertionError('Both chip results required')
            return [ttnn.to_torch(part).clone() for part in parts]

        keys = upload(fixture['keys'], ttnn.bfloat8_b)
        owned.append(keys)
        values = upload(fixture['values'], ttnn.bfloat8_b)
        owned.append(values)
        for name, tensor in (('keys', keys), ('values', values)):
            if any(not torch.equal(part, fixture[name]) for part in host(tensor)):
                raise AssertionError('BF8 reconstruction changed the saved hardware cache')
        pages = upload(fixture['pages'], ttnn.int32)
        query = upload(fixture['query'])
        owned.extend((pages, query))
        queries = [fixture['query'], -fixture['query'], fixture['query']]
        gold = []
        for ticket, ticket_query in enumerate(queries[:2]):
            outputs = []
            for row in range(8):
                row_query = upload(ticket_query[:, row:row + 1].contiguous())
                position = upload(torch.tensor([fixture['position'] + row], dtype=torch.int32), ttnn.int32)
                output = None
                try:
                    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(row_query, keys, values,
                        page_table_tensor=pages, cur_pos_tensor=position, scale=fixture['scale'],
                        program_config=config, memory_config=ttnn.L1_MEMORY_CONFIG)
                    outputs.append(host(output))
                finally:
                    release_owned(ttnn, [row_query, position] + ([output] if output is not None else []))
            gold.append([torch.cat([entry[chip] for entry in outputs], dim=1) for chip in range(2)])
            print(json.dumps(dict(stage='native-real-query', ticket=ticket)), flush=True)
        gold.append(gold[0])
        if any(torch.equal(first, changed) for first, changed in zip(gold[0], gold[1], strict=True)):
            raise AssertionError('Changed-query negative control did not change native attention')
        report['hardware_reference'] = [comparison(part, fixture['expected']) for part in gold[0]]
        arms = ((False, False), (True, False), (True, True)) if 'mask' in fixture else ((False, True),)
        for captured_mask, repair_mask in arms:
            reader, trace, output = None, None, None
            try:
                operations = ttnn if repair_mask else Overlay(ttnn, full_like=lambda tensor, *args, **kwargs: tensor)
                reader = ReplayAttentionReader(operations, mesh, 8, capacity, fixture['pages'], upload, short_context=True)
                reader.stage(fixture['position'])
                if captured_mask:
                    staged_mask = ttnn.from_torch(fixture['mask'].contiguous(), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    ttnn.copy_host_to_device_tensor(staged_mask, reader.metadata[0][2])
                warm = reader(query, keys, values, scale=fixture['scale'], memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(warm)
                trace, output = capture_operation(ttnn, mesh,
                    lambda: reader(query, keys, values, scale=fixture['scale'], memory_config=ttnn.L1_MEMORY_CONFIG))
                first_outputs = None
                for ticket, ticket_query in enumerate(queries):
                    staged = ttnn.from_torch(ticket_query.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                             mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                    ttnn.copy_host_to_device_tensor(staged, query)
                    if captured_mask:
                        ttnn.copy_host_to_device_tensor(staged_mask, reader.metadata[0][2])
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual_outputs = host(output)
                    for chip, actual in enumerate(actual_outputs):
                        check = dict(captured_mask=captured_mask, repair_mask=repair_mask, ticket=ticket, chip=chip,
                                     **comparison(actual, gold[ticket][chip]))
                        if ticket == 0:
                            check['matches_original_hardware_candidate'] = torch.equal(actual, fixture['actual'])
                        if ticket == 1:
                            check['rejects_stale_query'] = not torch.equal(actual, first_outputs[chip])
                        report['checks'].append(check)
                        print(json.dumps(check), flush=True)
                    if ticket == 0:
                        first_outputs = actual_outputs
                    options.output.write_text(json.dumps(report, indent=2))
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                if output is not None:
                    ttnn.deallocate(output)
                if reader is not None:
                    reader.close()
        report['diagnostic_complete'] = len(report['checks']) == (18 if 'mask' in fixture else 6)
        report['fresh_mask_exact'] = all(check['exact'] for check in report['checks'] if not check['captured_mask'])
        reproductions = [check['matches_original_hardware_candidate'] for check in report['checks']
                         if check['captured_mask'] and not check['repair_mask'] and check['ticket'] == 0]
        report['hardware_failure_reproduced'] = bool(reproductions) and all(reproductions)
        report['repaired_mask_exact'] = all(check['exact'] for check in report['checks'] if check['repair_mask'])
        report['passed'] = report['diagnostic_complete'] and report['fresh_mask_exact'] and all(
            check['exact'] for check in report['hardware_reference']) and report['hardware_failure_reproduced'] and report['repaired_mask_exact']
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        release_owned(ttnn, owned)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
