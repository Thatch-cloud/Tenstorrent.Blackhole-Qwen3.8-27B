"""Simulator-only attention folding probe against native serial causal SDPA."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from attention_head_fold import causal_mask, chunk_groups, device_layout, fold_query, unfold_output
from gdn_multitoken_conv import release_owned


def main():
    spec = importlib.util.spec_from_file_location('sim_guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.require_simulator(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, choices=(1, 2, 4, 8, 16, 32), default=2)
    parser.add_argument('--start', type=int, choices=(15, 31, 63, 127, 4095, 4096, 16383, 16384), default=63)
    parser.add_argument('--finite-mask', action='store_true')
    parser.add_argument('--grouped', action='store_true')
    parser.add_argument('--max-group-rows', type=int, choices=(4, 8, 16, 32), default=4)
    parser.add_argument('--device-layout', action='store_true')
    parser.add_argument('--dma-layout', action='store_true')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--capacity', type=int, choices=(64, 128, 256, 4352, 16640), default=256)
    parser.add_argument('--chunk-size', type=int, choices=(32, 64, 128, 256), default=32)
    args = parser.parse_args()
    if args.dma_layout and not args.device_layout:
        parser.error('DMA layout requires the device-layout oracle checks')
    if args.device_layout and args.max_group_rows > (4 if args.dma_layout else 8):
        parser.error('Group exceeds the selected device layout implementation')
    if args.start + args.rows > args.capacity or args.capacity % args.chunk_size:
        parser.error('Candidate cache view must cover positions and contain complete chunks')
    if args.device_layout and not args.grouped and args.rows > (4 if args.dma_layout else 8):
        parser.error('Device layout exceeds its supported group width')
    if args.grouped and any(group['signature'][1] % 64 for group in chunk_groups(args.start, args.rows, max_group_rows=args.max_group_rows)):
        parser.error('Grouped probe requires complete native cache-page views')
    path = Path(os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, backend='ttsim', rows=args.rows, start=args.start, finite_mask=args.finite_mask,
        capacity=args.capacity, chunk_size=args.chunk_size,
        grouped=args.grouped, seed=args.seed,
        max_group_rows=args.max_group_rows, tree_scratch_rounds=os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') == '1',
        device_layout=args.device_layout, dma_layout=args.dma_layout,
        scope='Query grouping and explicit masks versus native causal B1; optional stock or DMA device layout correctness, no speed certification')
    report['runtime_library_sha256'] = hashlib.sha256((Path(os.environ['TT_METAL_HOME']) /
        'build_Release/lib/_ttnncpp.so').read_bytes()).hexdigest()
    mesh = None
    owned = []

    def stage(name):
        report['last_stage'] = name
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=name)), flush=True)

    try:
        import torch
        import ttnn
        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)

        def layout(tensor, rows, *, inverse=False, offset=0):
            if args.dma_layout:
                from attention_fold_dma import device_layout_dma
                return device_layout_dma(mesh, tensor, rows, owned, inverse=inverse, offset=offset)
            return device_layout(ttnn, tensor, rows, owned, inverse=inverse, offset=offset)

        def upload(value, dtype=ttnn.bfloat16):
            tensor = ttnn.from_torch(value, device=mesh, dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            owned.append(tensor)
            return tensor

        def host(tensor):
            parts = ttnn.get_device_tensors(tensor)
            if len(parts) != 2:
                raise AssertionError('Two chips required')
            return [ttnn.to_torch(part).clone() for part in parts]

        torch.manual_seed(args.seed)
        query = torch.randn(1, args.rows, 12, 256).bfloat16()
        device_query = upload(query) if args.device_layout else None
        blocks = max(4, args.capacity // 64)
        keys = upload(torch.randn(blocks, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
        values = upload(torch.randn(blocks, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
        page_ids = [2, 0, 3, 1, *range(4, blocks)]
        pages = upload(torch.tensor([page_ids], dtype=torch.int32), ttnn.int32)
        grid = mesh.compute_with_storage_grid_size()
        native_config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)
        outputs = []
        for token in range(args.rows):
            stage(f'native-{token}')
            native = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                upload(query[:, token:token + 1].contiguous()), keys, values, page_table_tensor=pages,
                cur_pos_tensor=upload(torch.tensor([args.start + token], dtype=torch.int32), ttnn.int32),
                scale=0.0625, program_config=native_config, memory_config=ttnn.L1_MEMORY_CONFIG)
            owned.append(native)
            outputs.append(host(native))
        expected = [torch.cat([output[chip] for output in outputs], dim=1) for chip in range(2)]
        groups = chunk_groups(args.start, args.rows, max_group_rows=args.max_group_rows) if args.grouped else [
            dict(offset=0, rows=args.rows, signature=(args.chunk_size, args.capacity))]
        report['groups'] = groups
        grouped_outputs = []
        for group in groups:
            offset, rows = group['offset'], group['rows']
            chunk_size, capacity = group['signature']
            stage(f'folded-{offset}-{rows}')
            if capacity % 64:
                raise ValueError('This probe requires a complete native cache-page view')
            mask = causal_mask(rows, args.start + offset, capacity)
            if args.finite_mask:
                mask.masked_fill_(torch.isneginf(mask), -1e9)
            packed_query = layout(device_query, rows, offset=offset) if args.device_layout else upload(
                fold_query(query[:, offset:offset + rows].contiguous()))
            if args.device_layout and any(not torch.equal(value, fold_query(query[:, offset:offset + rows].contiguous()))
                                          for value in host(packed_query)):
                raise AssertionError('Device query packing differs from the host layout oracle')
            folded = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                packed_query, keys, values,
                page_table_tensor=upload(torch.tensor([page_ids[:capacity // 64]], dtype=torch.int32), ttnn.int32),
                is_causal=False, attn_mask=upload(mask), scale=0.0625,
                program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=chunk_size), memory_config=ttnn.L1_MEMORY_CONFIG)
            owned.append(folded)
            if args.device_layout:
                unpacked = layout(folded, rows, inverse=True)
                grouped_outputs.append(host(unpacked))
            else:
                grouped_outputs.append([unfold_output(value, rows) for value in host(folded)])
        actual = [torch.cat([output[chip] for output in grouped_outputs], dim=1) for chip in range(2)]
        report['checks'] = [dict(chip=chip, exact=torch.equal(reference, candidate),
            differences=int((reference != candidate).sum()), max_abs=float((reference.float() - candidate.float()).abs().max()),
            reference_max=float(reference.abs().max()), candidate_max=float(candidate.abs().max()),
            differences_by_token=[int((reference[:, token] != candidate[:, token]).sum()) for token in range(args.rows)],
            reference_finite=bool(torch.isfinite(reference).all()), candidate_finite=bool(torch.isfinite(candidate).all()))
            for chip, (reference, candidate) in enumerate(zip(expected, actual, strict=True))]
        stage('comparison')
        if any(not check['exact'] for check in report['checks']):
            raise AssertionError('Folded attention differs from native causal B1')
        if args.device_layout and args.grouped and args.rows >= 8 and args.max_group_rows == 4:
            from attention_grouped import GroupedAttentionReader
            stage('prepared-reader-lifetime')
            positions = [upload(torch.tensor([args.start + offset], dtype=torch.int32), ttnn.int32)
                         for offset in range(args.rows)]
            reader = GroupedAttentionReader(ttnn, mesh, args.start, args.rows,
                torch.tensor([page_ids], dtype=torch.int32), positions, pages, upload, dma_layout=args.dma_layout)
            result = reader(device_query, keys, values, page_table_tensor=pages, cur_pos_tensor=positions[0],
                scale=0.0625, program_config=native_config, memory_config=ttnn.L1_MEMORY_CONFIG)
            owned.append(result)
            if any(not torch.equal(reference, actual) for reference, actual in zip(expected, host(result), strict=True)):
                raise AssertionError('Prepared reader differs from native B1')
            if any(not torch.equal(query, actual) for actual in host(device_query)):
                raise AssertionError('Reader changed or released borrowed query')
            report['prepared_reader_exact'] = True
        report['passed'] = True
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
