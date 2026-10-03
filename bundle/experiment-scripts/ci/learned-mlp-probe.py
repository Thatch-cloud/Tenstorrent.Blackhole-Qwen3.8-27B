"""Layer-zero learned SwiGLU with optional convolution/residual integration; no full-drafter claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_mlp import split_mlp_weights, swiglu_reference, swiglu_device
from draft_mlp_fixture import load_mlp
from draft_convolution import grouped_causal_convolution, convolution_reference
from draft_convolution_fixture import load_convolution
from feature_collective import gather_add_projection
from feature_normalization import bf16_ulp_distance, rms_reference
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from projection_rounding import grouped_projection_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--convolution-fixture', type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if not options.hardware and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Connected simulator required')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    manifest, weights = load_mlp(options.fixture)
    convolution_manifest, convolution_weights = load_convolution(options.convolution_fixture) if options.convolution_fixture else (None, None)
    shards = split_mlp_weights(*(weights[f'layers.0.mlp.{name}_proj.weight'] for name in ('gate', 'up', 'down')))
    report = dict(passed=False, scope=__doc__, checkpoint=manifest, checks=[], block_rows=8,
        backend='hardware' if options.hardware else 'simulator',
        convolution_checkpoint=convolution_manifest, integrated_branch=bool(options.convolution_fixture),
        projection_fidelity_span=32,
        tolerance=dict(projection_rtol=1e-4, projection_atol=1e-4, activation_ulps=2),
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('learned-mlp-probe.py', 'draft_mlp.py', 'draft_mlp_fixture.py', 'feature_collective.py',
                'projection_rounding.py', 'draft_convolution.py', 'draft_convolution_fixture.py', 'feature_normalization.py')})
    mesh = None
    tensors = []

    def retain(value):
        tensors.append(value)
        return value

    def checkpoint(stage, **details):
        report['progress'] = dict(stage=stage, **details)
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['progress']), flush=True)

    def upload(value, sharded=False, layout=None):
        return retain(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT if layout is None else layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh)))

    def host(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip])

    try:
        checkpoint('opening_mesh')
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        collectives = TT_CCL(mesh)
        inputs = torch.zeros((1, 1, 32, 5120), dtype=torch.bfloat16)
        inputs[..., :8, :] = (torch.randn((1, 1, 8, 5120), generator=torch.Generator().manual_seed(673)) * .125).bfloat16()
        device_input = upload(inputs)
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        residual = device_input
        if convolution_weights is not None:
            norm_weight = convolution_weights['layers.0.post_attention_layernorm.weight']
            conv_weight = convolution_weights['layers.0.mlp_conv.kernel_projection.weight'].T.contiguous()
            base_weight = convolution_weights['layers.0.mlp_conv.base_kernel']
            device_norm = upload(norm_weight.reshape(1, 1, 160, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
            normalized = retain(ttnn.rms_norm(residual, epsilon=1e-6, weight=device_norm,
                compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            conv_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 5),
                in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
                fuse_batch=True, fused_activation=None, mcast_in0=True)
            conv_projection = retain(ttnn.matmul(normalized, upload(conv_weight), dtype=ttnn.float32,
                program_config=conv_program, compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            rounded_dynamic = retain(ttnn.typecast(conv_projection, ttnn.bfloat16))
            dynamic = [retain(ttnn.slice(rounded_dynamic, (0, 0, 0, offset * 320), (1, 1, 32, (offset + 1) * 320)))
                for offset in range(4)]
            bases = [upload(base_weight[phase, offset].reshape(1, 1, 1, 5120))
                for phase in range(2) for offset in range(2)]
            device_input = retain(grouped_causal_convolution(ttnn, mesh, normalized, dynamic[:2], bases[:2],
                fp32_intermediates=True))
            checkpoint('convolution_prepare_complete')
        projections = []
        for index, name in enumerate(('gate', 'up')):
            device_weight = upload(torch.cat([rank[index] for rank in shards], dim=0), True)
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 10),
                in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=4,
                fuse_batch=True, fused_activation=None, mcast_in0=True)
            projections.append(retain(ttnn.matmul(device_input, device_weight, dtype=ttnn.float32,
                program_config=program, compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
            checkpoint('projection_dispatched', projection=name)
        activated = swiglu_device(ttnn, *projections, retain)
        down_weight = upload(torch.cat([rank[2] for rank in shards], dim=0), True)
        down_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 10),
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=2,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        partial = retain(ttnn.matmul(activated, down_weight, dtype=ttnn.float32,
            program_config=down_program, compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        output = retain(gather_add_projection(ttnn, mesh, collectives, partial))
        if convolution_weights is not None:
            rounded_output = retain(ttnn.typecast(output, ttnn.bfloat16))
            finished = retain(grouped_causal_convolution(ttnn, mesh, rounded_output, dynamic[2:], bases[2:],
                fp32_intermediates=True))
            wide_finished = retain(ttnn.typecast(finished, ttnn.float32))
            wide_residual = retain(ttnn.typecast(residual, ttnn.float32))
            branch_sum = retain(ttnn.add(wide_finished, wide_residual, dtype=ttnn.float32))
            branch_output = retain(ttnn.typecast(branch_sum, ttnn.bfloat16))
        ttnn.synchronize_device(mesh)
        checkpoint('device_pipeline_complete')
        partials = [host(partial, chip) for chip in range(2)]
        for chip in range(2):
            actual_input = host(device_input, chip)
            if convolution_weights is not None:
                actual_norm = host(normalized, chip)
                norm_ulps = int(bf16_ulp_distance(actual_norm, rms_reference(inputs, norm_weight)).max())
                if norm_ulps > 2:
                    raise AssertionError('Integrated MLP normalization exceeds two BF16 ULPs')
                actual_conv_projection = host(conv_projection, chip)
                expected_conv_projection = grouped_projection_reference(actual_norm, conv_weight, destination_rounding=True, fidelity_span=32)
                torch.testing.assert_close(actual_conv_projection.double(), expected_conv_projection, rtol=1e-4, atol=1e-4)
                expected_dynamic = actual_conv_projection.bfloat16().split(320, dim=-1)
                if any(not torch.equal(host(value, chip), expected) for value, expected in zip(dynamic, expected_dynamic, strict=True)):
                    raise AssertionError('Dynamic convolution kernel slicing must be exact')
                expected_prepared = convolution_reference(actual_norm, expected_dynamic[:2],
                    [base_weight[0, offset].reshape(1, 1, 1, 5120) for offset in range(2)])
                expected_finished = convolution_reference(host(output, chip).bfloat16(), expected_dynamic[2:],
                    [base_weight[1, offset].reshape(1, 1, 1, 5120) for offset in range(2)])
                if not torch.equal(actual_input, expected_prepared) or not torch.equal(host(finished, chip), expected_finished):
                    raise AssertionError('Integrated MLP convolution arithmetic must be exact')
                if not torch.equal(host(branch_output, chip), (host(finished, chip).float() + inputs.float()).bfloat16()):
                    raise AssertionError('Integrated MLP residual must be exact')
                report['checks'].append(dict(chip=chip, stage='convolution/residual', norm_ulps=norm_ulps,
                    prepare_exact=True, finish_exact=True, residual_exact=True))
                checkpoint('convolution_reference_complete', chip=chip)
            for index, name in enumerate(('gate', 'up')):
                expected = grouped_projection_reference(actual_input[..., :8, :], shards[chip][index], destination_rounding=True, fidelity_span=32)
                actual = host(projections[index], chip)[..., :8, :].double()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                report['checks'].append(dict(chip=chip, stage=name, max_error=float((actual - expected).abs().max())))
                checkpoint('projection_reference_complete', chip=chip, projection=name)
            expected_activation = swiglu_reference(*(host(value, chip) for value in projections))
            actual_activation = host(activated, chip)
            distance = int(bf16_ulp_distance(actual_activation, expected_activation).max())
            if distance > 2:
                raise AssertionError(f'SwiGLU activation differs by {distance} BF16 ULPs')
            expected_down = grouped_projection_reference(actual_activation[..., :8, :], shards[chip][2], destination_rounding=True, fidelity_span=32)
            actual_down = partials[chip][..., :8, :].double()
            mismatched = ~torch.isclose(actual_down, expected_down, rtol=1e-4, atol=1e-4)
            if mismatched.any():
                ideal_down = actual_activation[..., :8, :].double() @ shards[chip][2].double()
                failure_path = options.output.with_suffix(f'.rank{chip}-down-failure.pt')
                torch.save(dict(activation=actual_activation[..., :8, :].contiguous(), actual=actual_down,
                    reference=expected_down, ideal=ideal_down, chip=chip, checkpoint=manifest, fidelity_span=32), failure_path)
                indices = mismatched.nonzero().tolist()
                report['down_failure'] = dict(chip=chip, activation_ulps=distance, capture=str(failure_path),
                    mismatched_elements=len(indices), samples=[dict(index=index, actual=float(actual_down[tuple(index)]),
                        reference=float(expected_down[tuple(index)]), ideal=float(ideal_down[tuple(index)])) for index in indices[:16]])
                checkpoint('down_projection_mismatch', chip=chip)
            torch.testing.assert_close(actual_down, expected_down, rtol=1e-4, atol=1e-4)
            if not torch.equal(host(output, chip), partials[0] + partials[1]):
                raise AssertionError('MLP fabric sum must be exact')
            report['checks'].append(dict(chip=chip, stage='activation/down', activation_ulps=distance, sum_exact=True,
                max_down_error=float((partials[chip][..., :8, :].double() - expected_down).abs().max())))
            checkpoint('rank_checks_complete', chip=chip)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            release_owned(ttnn, tensors)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
