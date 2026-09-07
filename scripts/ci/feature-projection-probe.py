"""Learned projection diagnostic with optional device fabric reduction and RMSNorm; not a draft-model benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_projection_fixture import MODEL, REVISION
from draft_projection_full_fixture import load_projection
from feature_projection import concatenate_local_features, projection_shards, sparse_input_permutation, require_projection_environment
from projection_rounding import grouped_projection_reference
from feature_collective import gather_add_projection
from feature_normalization import bf16_ulp_distance, rms_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--full-projection', action='store_true')
    parser.add_argument('--device-tail', action='store_true')
    parser.add_argument('--rows', type=int, choices=(1, 8, 32))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--k-block', type=int, choices=(4, 100), default=4)
    parser.add_argument('--active-k', type=int, choices=(1, 2, 4, 8, 16, 32, 128, 12800), default=12800)
    parser.add_argument('--k-stride', type=int, choices=(1, 4, 8, 16, 32), default=1)
    parser.add_argument('--fidelity', choices=('LoFi', 'HiFi2', 'HiFi3', 'HiFi4'), default='HiFi4')
    parser.add_argument('--reference', choices=('float64', 'blackhole-accumulation'), default='float64')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if options.device_tail and (not options.full_projection or options.rows != 1):
        parser.error('Device tail requires full projection and one explicit row')
    if options.device_tail and not options.hardware and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Device tail simulator requires shared BDF loading')
    if options.full_projection and options.rows is None:
        parser.error('Full projection requires explicit rows')
    if (options.active_k - 1) * options.k_stride >= 12800:
        parser.error('Active terms with the selected stride exceed the local input width')
    import torch
    import ttnn

    if options.full_projection:
        manifest, weight, norm_weight = load_projection(options.fixture)
    else:
        manifest = json.loads((options.fixture / 'manifest.json').read_text())
        data = (options.fixture / 'fc-first32.bf16').read_bytes()
        if (manifest['model'] != MODEL or manifest['revision'] != REVISION or manifest['slice_shape'] != [32, 25600]
                or manifest['dtype'] != 'BF16' or len(data) != 32 * 25600 * 2
                or hashlib.sha256(data).hexdigest() != '882b3405c0eb1e9bc0d502e0ff241e43c25f405c8e10c4ba469129008f161a33'
                or hashlib.sha256(data).hexdigest() != manifest['data_sha256']):
            raise ValueError('Audited learned-weight fixture required')
        weight = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(32, 25600).clone()
    outputs = weight.shape[0]
    row_counts = (options.rows,) if options.rows is not None else (1, 8, 32)
    if not torch.isfinite(weight).all():
        raise ValueError('Finite learned weights required')
    packed = projection_shards(weight)
    permutation = torch.tensor(sparse_input_permutation(12800, options.active_k, options.k_stride))
    packed = [shard[permutation].contiguous() for shard in packed]
    report = dict(passed=False, scope=__doc__, backend='hardware' if options.hardware else 'simulator', checkpoint=manifest, checks=[],
        tolerance=dict(rtol=1e-4, atol=1e-4), output_width=outputs, normalization_tested=False,
        device_tail=options.device_tail, device_tail_checks=[],
        active_k=options.active_k, k_stride=options.k_stride,
        fidelity=options.fidelity, reference=options.reference, matched_operands=True, packer_l1_acc=False, sources={name:
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('feature-projection-probe.py', 'feature_projection.py', 'draft_projection_fixture.py', 'draft_projection_full_fixture.py', 'projection_rounding.py', 'feature_collective.py', 'feature_normalization.py')})
    mesh = device_weight = device_norm = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if options.hardware or options.device_tail else ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        device_weight = ttnn.from_torch(torch.cat(packed, dim=0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        for chip, value in enumerate(ttnn.get_device_tensors(device_weight)):
            if not torch.equal(ttnn.to_torch(value), packed[chip]):
                raise AssertionError('Projection weight sharding changed the learned coefficients')
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, options.fidelity),
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        if options.device_tail:
            from models.tt_transformers.tt.ccl import TT_CCL

            collectives = TT_CCL(mesh)
            device_norm = ttnn.from_torch(norm_weight.reshape(1, 1, 160, 32), device=mesh,
                dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        grid = (8, 10) if options.full_projection else (1, 1)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=grid,
            in0_block_w=options.k_block, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=2 if options.full_projection else 1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        report['program'] = dict(kind='Explicit1D', grid=grid, k_block=options.k_block)
        for rows in row_counts:
            generator = torch.Generator().manual_seed(1234 + rows)
            features = [torch.randn((1, 1, rows, 5120), generator=generator).bfloat16() for tap in range(5)]
            for tap, value in enumerate(features):
                local_mask = torch.arange(2560) + tap * 2560 < options.active_k
                value.mul_(local_mask.repeat(2))
            reference = torch.cat(features, dim=-1).double() @ weight.T.double()
            local_features = [torch.cat([value[..., chip * 2560:(chip + 1) * 2560] for value in features], dim=-1)
                              [..., permutation] for chip in range(2)]
            features = [torch.cat([value[..., tap * 2560:(tap + 1) * 2560] for value in local_features], dim=-1)
                        for tap in range(5)]
            tensors = []
            tail_tensors = []
            joined = output = None
            try:
                for value in features:
                    tensors.append(ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)))
                joined = concatenate_local_features(ttnn, tensors)
                local_inputs = [torch.cat([value[..., chip * 2560:(chip + 1) * 2560] for value in features], dim=-1)
                                for chip in range(2)]
                for chip, value in enumerate(ttnn.get_device_tensors(joined)):
                    if not torch.equal(ttnn.to_torch(value), local_inputs[chip]):
                        raise AssertionError('Local feature concatenation changed tap order')
                output = ttnn.matmul(joined, device_weight, dtype=ttnn.float32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kernel, program_config=program)
                if options.device_tail:
                    reduced = gather_add_projection(ttnn, mesh, collectives, output)
                    tail_tensors.append(reduced)
                    norm_input = ttnn.typecast(reduced, ttnn.bfloat16)
                    tail_tensors.append(norm_input)
                    normalized = ttnn.rms_norm(norm_input, epsilon=1e-6, weight=device_norm,
                        compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    tail_tensors.append(normalized)
                ttnn.synchronize_device(mesh)
                actual = [ttnn.to_torch(value) for value in ttnn.get_device_tensors(output)]
                if len(actual) != 2:
                    raise AssertionError('Two partial projections required')
                if options.device_tail:
                    reference_sum = actual[0] + actual[1]
                    reference_norm = rms_reference(reference_sum.bfloat16(), norm_weight)
                    sums = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(reduced)]
                    normalized_values = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(normalized)]
                    if len(sums) != 2 or len(normalized_values) != 2:
                        raise AssertionError('Both ranks must retain normalized projection features')
                    for chip, (summed, norm_value) in enumerate(zip(sums, normalized_values, strict=True)):
                        distance = bf16_ulp_distance(norm_value, reference_norm)
                        check = dict(chip=chip, rows=rows, sum_exact=bool(torch.equal(summed, reference_sum)),
                            max_bf16_ulps=int(distance.max()), passed=bool(torch.equal(summed, reference_sum) and (distance <= 2).all()))
                        report['device_tail_checks'].append(check)
                    if not all(check['passed'] for check in report['device_tail_checks']):
                        raise AssertionError('Device-resident projection tail failed exact sum or BF16 normalization')
                    report['normalization_tested'] = True
                partials = [value.double() @ shard.double() for value, shard in zip(local_inputs, packed, strict=True)]
                torch.testing.assert_close(partials[0] + partials[1], reference, rtol=1e-12, atol=1e-12)
                wrong = torch.cat(local_inputs, dim=-1).double() @ weight.T.double()
                if torch.allclose(wrong, reference, rtol=1e-4, atol=1e-4):
                    raise AssertionError('Fixture cannot distinguish rank-major from tap-major feature order')
                for chip, (value, expected) in enumerate(zip(actual, partials, strict=True)):
                    phase_count = dict(LoFi=1, HiFi2=2, HiFi3=3, HiFi4=4)[options.fidelity]
                    grouped = grouped_projection_reference(local_inputs[chip], packed[chip], phases=phase_count)
                    reversed_grouped = grouped_projection_reference(local_inputs[chip], packed[chip],
                        phases=phase_count, reverse_sources=True)
                    accumulated = grouped_projection_reference(local_inputs[chip], packed[chip],
                        phases=phase_count, destination_rounding=True)
                    check = dict(rows=rows, chip=chip, shape=list(value.shape), dtype=str(value.dtype),
                        accumulated_reference_max_error=float((value.double() - accumulated).abs().max()),
                        accumulated_reference_exact=bool(torch.equal(value.double(), accumulated)),
                        float64_gate_passed=bool(torch.allclose(value.double(), expected, rtol=1e-4, atol=1e-4)),
                        grouped_product_max_error=float((value.double() - grouped).abs().max()),
                        reversed_grouped_product_max_error=float((value.double() - reversed_grouped).abs().max()),
                        grouped_reference_scope='Diagnostic only; excludes FP32 destination rounding',
                        max_abs_error=float((value.double() - expected).abs().max()), layout_exact=True,
                        wrong_order_detected=True, passed=False,
                        bf16_representable=bool(torch.equal(value, value.bfloat16().float())),
                        actual_first_row=value.reshape(-1, outputs)[0].tolist(),
                        expected_first_row=expected.reshape(-1, outputs)[0].tolist())
                    report['checks'].append(check)
                    try:
                        selected_reference = accumulated if options.reference == 'blackhole-accumulation' else expected
                        torch.testing.assert_close(value.double(), selected_reference, rtol=1e-4, atol=1e-4)
                        check['passed'] = True
                    except AssertionError as error:
                        check['error'] = str(error)
            finally:
                for tensor in reversed(tail_tensors):
                    ttnn.deallocate(tensor)
                if output is not None:
                    ttnn.deallocate(output)
                if joined is not None:
                    ttnn.deallocate(joined)
                for value in tensors:
                    ttnn.deallocate(value)
        if len(report['checks']) != len(row_counts) * 2 or not all(check['passed'] for check in report['checks']):
            raise AssertionError(f'Learned projection slice fails the {options.reference} numerical gate')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            if device_norm is not None:
                ttnn.deallocate(device_norm)
            if device_weight is not None:
                ttnn.deallocate(device_weight)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
