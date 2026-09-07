"""Simulator-only attention folding probe against native serial causal SDPA."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from attention_head_fold import causal_mask, chunk_groups, device_layout, fold_query, parallel_groups, unfold_output
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
    parser.add_argument('--parallel-groups', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--device-layout', action='store_true')
    parser.add_argument('--dma-layout', action='store_true')
    parser.add_argument('--dynamic-mask', action='store_true')
    parser.add_argument('--replay-reader', action='store_true')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--capacity', type=int, choices=(64, 128, 256, 4352, 16640), default=256)
    parser.add_argument('--chunk-size', type=int, choices=(32, 64, 128, 256), default=32)
    args = parser.parse_args()
    if args.replay_reader and not args.dynamic_mask:
        parser.error('Replay reader requires dynamic-mask composition checks')
    if args.dynamic_mask:
        from attention_mask_replay import validate_ticket
        if not args.dma_layout or args.parallel_groups != 3:
            parser.error('Dynamic mask composition requires three DMA groups')
        validate_ticket(args.start, args.rows, args.capacity)
        validate_ticket(args.start + 7, args.rows, args.capacity)
    if args.parallel_groups > 1 and (not args.grouped or args.max_group_rows != 4 or args.finite_mask):
        parser.error('Parallel probe requires four-row groups with native masks')
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
        parallel_groups=args.parallel_groups,
        device_layout=args.device_layout, dma_layout=args.dma_layout, dynamic_mask=args.dynamic_mask,
        replay_reader=args.replay_reader,
        scope='Query grouping and explicit masks versus native causal B1; optional stock or DMA device layout correctness, no speed certification')
    report['runtime_library_sha256'] = hashlib.sha256((Path(os.environ['TT_METAL_HOME']) /
        'build_Release/lib/_ttnncpp.so').read_bytes()).hexdigest()
    mesh = None
    owned = []
    replay_reader = None

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
        if args.parallel_groups > 1:
            bundles = parallel_groups(args.start, args.rows, max_batches=args.parallel_groups)
            report['parallel_plan'] = bundles
            parallel_outputs = []
            for bundle in bundles:
                stage(f"parallel-{bundle[0]['offset']}-{len(bundle)}")
                count = bundle[0]['rows']
                chunk_size, capacity = bundle[0]['signature']
                queries = torch.cat([fold_query(query[:, group['offset']:group['offset'] + count].contiguous())
                                     for group in bundle], dim=1)
                if args.device_layout:
                    packed = [layout(device_query, count, offset=group['offset']) for group in bundle]
                    stacked = ttnn.concat(packed, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(bundle) > 1 else packed[0]
                    owned.append(stacked)
                    if any(not torch.equal(queries, actual) for actual in host(stacked)):
                        raise AssertionError('Parallel device query packing differs from Torch')
                else:
                    stacked = upload(queries)
                masks = torch.cat([causal_mask(count, args.start + group['offset'], capacity) for group in bundle], dim=0)
                grouped_pages = torch.tensor([page_ids[:capacity // 64]], dtype=torch.int32).repeat(len(bundle), 1)
                result = ttnn.transformer.paged_scaled_dot_product_attention_decode(stacked, keys, values,
                    page_table_tensor=upload(grouped_pages, ttnn.int32), is_causal=False, attn_mask=upload(masks), scale=0.0625,
                    program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=chunk_size), memory_config=ttnn.L1_MEMORY_CONFIG)
                owned.append(result)
                for index in range(len(bundle)):
                    if args.device_layout:
                        selected = ttnn.slice(result, (0, index, 0, 0), (1, index + 1, count * 12, 256),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG) if len(bundle) > 1 else result
                        owned.append(selected)
                        parallel_outputs.append(host(layout(selected, count, inverse=True)))
                    else:
                        parallel_outputs.append([unfold_output(part[:, index:index + 1], count) for part in host(result)])
            actual_parallel = [torch.cat([output[chip] for output in parallel_outputs], dim=1) for chip in range(2)]
            report['parallel_checks'] = [dict(chip=chip, exact=torch.equal(reference, actual),
                differences=int((reference != actual).sum()), max_abs=float((reference.float() - actual.float()).abs().max()))
                for chip, (reference, actual) in enumerate(zip(expected, actual_parallel, strict=True))]
            stage('parallel-comparison')
            if not all(check['exact'] for check in report['parallel_checks']):
                raise AssertionError('Parallel groups differ from native B1')
            if args.device_layout and args.dma_layout:
                from attention_parallel import execute
                metadata = []
                for bundle in bundles:
                    count = bundle[0]['rows']
                    chunk_size, capacity = bundle[0]['signature']
                    mask = torch.cat([causal_mask(count, args.start + group['offset'], capacity) for group in bundle], dim=0)
                    group_pages = torch.tensor([page_ids[:capacity // 64]], dtype=torch.int32).repeat(len(bundle), 1)
                    config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                        exp_approx_mode=False, q_chunk_size=0, k_chunk_size=chunk_size)
                    metadata.append((bundle, upload(group_pages, ttnn.int32), upload(mask), config))
                stage('parallel-adapter')
                result = execute(mesh, ttnn, device_query, keys, values, metadata, owned,
                    scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)
                if any(not torch.equal(reference, actual) for reference, actual in zip(expected, host(result), strict=True)):
                    raise AssertionError('Reusable parallel adapter differs from native B1')
                report['parallel_adapter_exact'] = True
                if args.dynamic_mask:
                    from attention_mask_replay import execute as refresh_mask, prepare, source_hashes, validate_ticket
                    report['mask_sources'] = source_hashes()
                    words = torch.zeros(8, dtype=torch.int32)
                    words[0] = args.start
                    position_input = upload(words, ttnn.int32)
                    mask_programs = [prepare(mesh, position_input, mask, rows=bundle[0]['rows'],
                        batches=len(bundle), offset=bundle[0]['offset'], capacity=args.capacity)
                        for bundle, group_pages, mask, config in metadata]
                    before_addresses = [tuple(part.buffer_address() for part in ttnn.get_device_tensors(mask))
                                        for bundle, group_pages, mask, config in metadata]
                    report['dynamic_checks'] = []
                    if args.replay_reader:
                        from attention_replay import ReplayAttentionReader
                        report['replay_reader_sha256'] = hashlib.sha256((Path(__file__).resolve().parents[2] /
                            'scripts/ci/attention_replay.py').read_bytes()).hexdigest()
                        def upload_replay(value, dtype=ttnn.bfloat16):
                            return ttnn.from_torch(value, device=mesh, dtype=dtype,
                                layout=ttnn.ROW_MAJOR_LAYOUT if dtype == ttnn.int32 else ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                        replay_reader = ReplayAttentionReader(ttnn, mesh, args.rows, args.capacity,
                            torch.tensor([page_ids], dtype=torch.int32), upload_replay)
                    for delta in (7, 0):
                        start = args.start + delta
                        validate_ticket(start, args.rows, args.capacity)
                        stage(f'dynamic-native-{delta}')
                        if delta:
                            shifted_outputs = []
                            for token in range(args.rows):
                                native = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                                    upload(query[:, token:token + 1].contiguous()), keys, values,
                                    page_table_tensor=pages,
                                    cur_pos_tensor=upload(torch.tensor([start + token], dtype=torch.int32), ttnn.int32),
                                    scale=0.0625, program_config=native_config, memory_config=ttnn.L1_MEMORY_CONFIG)
                                owned.append(native)
                                shifted_outputs.append(host(native))
                            dynamic_expected = [torch.cat([value[chip] for value in shifted_outputs], dim=1) for chip in range(2)]
                        else:
                            dynamic_expected = expected
                        words[0] = start
                        staged = ttnn.from_torch(words, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                        ttnn.copy_host_to_device_tensor(staged, position_input)
                        for entry, program in zip(metadata, mask_programs, strict=True):
                            refresh_mask(position_input, entry[2], program)
                        result = execute(mesh, ttnn, device_query, keys, values, metadata, owned,
                            scale=0.0625, memory_config=ttnn.L1_MEMORY_CONFIG)
                        for chip, (reference, actual) in enumerate(zip(dynamic_expected, host(result), strict=True)):
                            if not torch.equal(reference, actual):
                                raise AssertionError('Refreshed mask attention differs from changed-position native B1')
                            report['dynamic_checks'].append(dict(start=start, chip=chip, exact=True))
                        if before_addresses != [tuple(part.buffer_address() for part in ttnn.get_device_tensors(mask))
                                                for bundle, group_pages, mask, config in metadata]:
                            raise AssertionError('Dynamic mask refresh replaced existing mask buffers')
                        if replay_reader is not None:
                            replay_reader.stage(start)
                            reader_result = replay_reader(device_query, keys, values, scale=0.0625,
                                memory_config=ttnn.L1_MEMORY_CONFIG)
                            owned.append(reader_result)
                            if any(not torch.equal(reference, actual) for reference, actual in
                                   zip(dynamic_expected, host(reader_result), strict=True)):
                                raise AssertionError('Prepared replay reader differs from changed-position native B1')
                            if any(not torch.equal(query, actual) for actual in host(device_query)):
                                raise AssertionError('Prepared replay reader changed or released borrowed query')
                    if replay_reader is not None:
                        if replay_reader.calls != 2 or replay_reader.refresh_calls != 2 * len(replay_reader.metadata):
                            raise AssertionError('Every replay must refresh every prepared mask')
                        report['replay_reader_exact'] = True
                        report['replay_reader_refresh_calls'] = replay_reader.refresh_calls
        if args.device_layout and args.grouped and args.rows >= 8 and args.max_group_rows == 4 and (args.parallel_groups == 1 or args.dma_layout):
            from attention_grouped import GroupedAttentionReader
            stage('prepared-reader-lifetime')
            positions = [upload(torch.tensor([args.start + offset], dtype=torch.int32), ttnn.int32)
                         for offset in range(args.rows)]
            reader = GroupedAttentionReader(ttnn, mesh, args.start, args.rows,
                torch.tensor([page_ids], dtype=torch.int32), positions, pages, upload,
                dma_layout=args.dma_layout, parallel=args.parallel_groups > 1)
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
            if replay_reader is not None:
                replay_reader.close()
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
