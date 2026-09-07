"""Layer-zero learned convolution plumbing; no attention/MLP transform or draft-model claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_convolution import convolution_reference, grouped_causal_convolution
from draft_convolution_fixture import load_convolution
from feature_normalization import rms_reference, bf16_ulp_distance
from feature_projection import require_projection_environment
from projection_rounding import grouped_projection_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--fp32-intermediates', action='store_true')
    parser.add_argument('--inspect-arithmetic', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    manifest, weights = load_convolution(options.fixture)
    report = dict(passed=False, scope=__doc__, backend='simulator', checkpoint=manifest, checks=[],
        fp32_intermediates=options.fp32_intermediates,
        arithmetic_checks=[],
        tolerance=dict(rtol=1e-4, atol=1e-4, max_bf16_ulps=2), sources={name:
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('learned-convolution-probe.py', 'draft_convolution.py', 'draft_convolution_fixture.py', 'projection_rounding.py')})
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 5),
            in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        for branch, norm_name in (('attention', 'input_layernorm'), ('mlp', 'post_attention_layernorm')):
            tensors = []

            def inspect(operation, left, right, result, rounded):
                values = [ttnn.to_torch(ttnn.get_device_tensors(value)[0]) for value in (left, right, result, rounded)]
                expected = values[0].float() * values[1].float() if operation == 'multiply' else values[0].float() + values[1].float()
                report['arithmetic_checks'].append(dict(branch=branch, operation=operation,
                    fp32_max_error=float((values[2].float() - expected).abs().max()),
                    bf16_max_ulps=int(bf16_ulp_distance(values[3], expected.bfloat16()).max())))

            def retain(value):
                tensors.append(value)
                return value

            def upload(value, layout=ttnn.TILE_LAYOUT):
                return retain(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                    layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))

            try:
                hidden = torch.randn((1, 1, 8, 5120), generator=torch.Generator().manual_seed(2027)).bfloat16()
                norm_weight = weights[f'layers.0.{norm_name}.weight']
                projection_weight = weights[f'layers.0.{branch}_conv.kernel_projection.weight'].T.contiguous()
                base_weight = weights[f'layers.0.{branch}_conv.base_kernel']
                device_hidden = upload(hidden)
                device_norm = upload(norm_weight.reshape(1, 1, 160, 32), ttnn.ROW_MAJOR_LAYOUT)
                normalized = retain(ttnn.rms_norm(device_hidden, epsilon=1e-6, weight=device_norm,
                    compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                device_weight = upload(projection_weight)
                projected = retain(ttnn.matmul(normalized, device_weight, dtype=ttnn.float32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kernel, program_config=program))
                rounded = retain(ttnn.typecast(projected, ttnn.bfloat16))
                dynamic = [retain(ttnn.slice(rounded, (0, 0, 0, offset * 320), (1, 1, 8, (offset + 1) * 320)))
                    for offset in range(4)]
                bases = [upload(base_weight[phase, offset].reshape(1, 1, 1, 5120))
                    for phase in range(2) for offset in range(2)]
                prepared = retain(grouped_causal_convolution(ttnn, mesh, normalized, dynamic[:2], bases[:2],
                    fp32_intermediates=options.fp32_intermediates, inspect=inspect if options.inspect_arithmetic else None))
                finished = retain(grouped_causal_convolution(ttnn, mesh, prepared, dynamic[2:], bases[2:],
                    fp32_intermediates=options.fp32_intermediates, inspect=inspect if options.inspect_arithmetic else None))
                ttnn.synchronize_device(mesh)
                for chip in range(2):
                    actual_norm = ttnn.to_torch(ttnn.get_device_tensors(normalized)[chip])
                    actual_projection = ttnn.to_torch(ttnn.get_device_tensors(projected)[chip])
                    expected_projection = grouped_projection_reference(actual_norm, projection_weight, destination_rounding=True)
                    torch.testing.assert_close(actual_projection.double(), expected_projection, rtol=1e-4, atol=1e-4)
                    actual_dynamic = actual_projection.bfloat16().split(320, dim=-1)
                    actual_prepared = ttnn.to_torch(ttnn.get_device_tensors(prepared)[chip])
                    actual_finished = ttnn.to_torch(ttnn.get_device_tensors(finished)[chip])
                    expected_values = [rms_reference(hidden, norm_weight),
                        convolution_reference(actual_norm, actual_dynamic[:2],
                            [base_weight[0, offset].reshape(1, 1, 1, 5120) for offset in range(2)]),
                        convolution_reference(actual_prepared, actual_dynamic[2:],
                            [base_weight[1, offset].reshape(1, 1, 1, 5120) for offset in range(2)])]
                    actual_values = [actual_norm, actual_prepared, actual_finished]
                    distances = [bf16_ulp_distance(actual, expected) for actual, expected in zip(actual_values, expected_values, strict=True)]
                    check = dict(branch=branch, chip=chip, rows=8, max_ulps=[int(value.max()) for value in distances],
                        projection_max_error=float((actual_projection.double() - expected_projection).abs().max()))
                    device_dynamic = [ttnn.to_torch(ttnn.get_device_tensors(value)[chip]) for value in dynamic]
                    check['dynamic_cast_ulps'] = [int(bf16_ulp_distance(actual, expected).max())
                        for actual, expected in zip(device_dynamic, actual_dynamic, strict=True)]
                    check['max_abs_errors'] = [float((actual.float() - expected.float()).abs().max())
                        for actual, expected in zip(actual_values, expected_values, strict=True)]
                    check['mismatch_samples'] = []
                    for actual, expected, distance in zip(actual_values, expected_values, distances, strict=True):
                        indices = (distance.flatten() > 2).nonzero().flatten()[:8]
                        check['mismatch_samples'].append([dict(index=int(index), actual=float(actual.flatten()[index]),
                            expected=float(expected.flatten()[index])) for index in indices])
                    check['passed'] = all(value <= 2 for value in check['max_ulps'])
                    report['checks'].append(check)
                    if not check['passed']:
                        raise AssertionError('Learned convolution exceeds predefined BF16 arithmetic bound')
            finally:
                for tensor in reversed(tensors):
                    ttnn.deallocate(tensor)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
