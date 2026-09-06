"""Simulator-only attention folding probe against native serial causal SDPA."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from attention_head_fold import causal_mask, fold_query, unfold_output


def main():
    spec = importlib.util.spec_from_file_location('sim_guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.require_simulator(os.environ)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--start', type=int, choices=(15, 31, 63, 127), default=63)
    parser.add_argument('--finite-mask', action='store_true')
    args = parser.parse_args()
    path = Path(os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, backend='ttsim', rows=args.rows, start=args.start, finite_mask=args.finite_mask,
        scope='Host-folded query and explicit mask versus native causal B1; no device layout or speed certification')
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

        torch.manual_seed(0)
        query = torch.randn(1, args.rows, 12, 256).bfloat16()
        keys = upload(torch.randn(4, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
        values = upload(torch.randn(4, 2, 64, 256).bfloat16() * 0.1, ttnn.bfloat8_b)
        pages = upload(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32), ttnn.int32)
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
        stage('folded-masked')
        mask = causal_mask(args.rows, args.start, 256)
        if args.finite_mask:
            mask.masked_fill_(torch.isneginf(mask), -1e9)
        folded = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            upload(fold_query(query)), keys, values, page_table_tensor=pages, is_causal=False,
            attn_mask=upload(mask), scale=0.0625,
            program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                exp_approx_mode=False, q_chunk_size=0, k_chunk_size=32), memory_config=ttnn.L1_MEMORY_CONFIG)
        owned.append(folded)
        actual = [unfold_output(value, args.rows) for value in host(folded)]
        report['checks'] = [dict(chip=chip, exact=torch.equal(reference, candidate),
            differences=int((reference != candidate).sum()), max_abs=float((reference.float() - candidate.float()).abs().max()),
            reference_max=float(reference.abs().max()), candidate_max=float(candidate.abs().max()),
            reference_finite=bool(torch.isfinite(reference).all()), candidate_finite=bool(torch.isfinite(candidate).all()))
            for chip, (reference, candidate) in enumerate(zip(expected, actual, strict=True))]
        stage('comparison')
        if any(not check['exact'] for check in report['checks']):
            raise AssertionError('Folded attention differs from native causal B1')
        report['passed'] = True
    finally:
        if mesh is not None:
            for tensor in reversed(owned):
                ttnn.deallocate(tensor)
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
