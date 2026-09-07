"""TTsim learned-weight projection slice; host reduction only, no fabric or draft-model claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_projection_fixture import MODEL, REVISION
from feature_projection import concatenate_local_features, projection_shards


def main():
    if not os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch simulator required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--k-block', type=int, choices=(4, 100), default=4)
    options = parser.parse_args()
    import torch
    import ttnn

    manifest = json.loads((options.fixture / 'manifest.json').read_text())
    data = (options.fixture / 'fc-first32.bf16').read_bytes()
    if (manifest['model'] != MODEL or manifest['revision'] != REVISION or manifest['slice_shape'] != [32, 25600]
            or manifest['dtype'] != 'BF16' or len(data) != 32 * 25600 * 2
            or hashlib.sha256(data).hexdigest() != '882b3405c0eb1e9bc0d502e0ff241e43c25f405c8e10c4ba469129008f161a33'
            or hashlib.sha256(data).hexdigest() != manifest['data_sha256']):
        raise ValueError('Audited learned-weight fixture required')
    weight = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(32, 25600).clone()
    if not torch.isfinite(weight).all():
        raise ValueError('Finite learned weights required')
    packed = projection_shards(weight)
    report = dict(passed=False, scope=__doc__, checkpoint=manifest, checks=[],
        tolerance=dict(rtol=1e-4, atol=1e-4), packer_l1_acc=False, sources={name:
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('feature-projection-probe.py', 'feature_projection.py', 'draft_projection_fixture.py')})
    mesh = device_weight = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        device_weight = ttnn.from_torch(torch.cat(packed, dim=0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        for chip, value in enumerate(ttnn.get_device_tensors(device_weight)):
            if not torch.equal(ttnn.to_torch(value), packed[chip]):
                raise AssertionError('Projection weight sharding changed the learned coefficients')
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(1, 1),
            in0_block_w=options.k_block, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        report['program'] = dict(kind='Explicit1D single-output-tile', k_block=options.k_block)
        for rows in (1, 8, 32):
            generator = torch.Generator().manual_seed(1234 + rows)
            features = [torch.randn((1, 1, rows, 5120), generator=generator).bfloat16() for tap in range(5)]
            tensors = []
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
                ttnn.synchronize_device(mesh)
                actual = [ttnn.to_torch(value) for value in ttnn.get_device_tensors(output)]
                if len(actual) != 2:
                    raise AssertionError('Two partial projections required')
                partials = [value.double() @ shard.double() for value, shard in zip(local_inputs, packed, strict=True)]
                reference = torch.cat(features, dim=-1).double() @ weight.T.double()
                torch.testing.assert_close(partials[0] + partials[1], reference, rtol=1e-12, atol=1e-12)
                wrong = torch.cat(local_inputs, dim=-1).double() @ weight.T.double()
                if torch.allclose(wrong, reference, rtol=1e-4, atol=1e-4):
                    raise AssertionError('Fixture cannot distinguish rank-major from tap-major feature order')
                for chip, (value, expected) in enumerate(zip(actual, partials, strict=True)):
                    check = dict(rows=rows, chip=chip, shape=list(value.shape), dtype=str(value.dtype),
                        max_abs_error=float((value.double() - expected).abs().max()), layout_exact=True,
                        wrong_order_detected=True, passed=False,
                        bf16_representable=bool(torch.equal(value, value.bfloat16().float())),
                        actual_first_row=value.reshape(-1, 32)[0].tolist(),
                        expected_first_row=expected.reshape(-1, 32)[0].tolist())
                    report['checks'].append(check)
                    try:
                        torch.testing.assert_close(value.double(), expected, rtol=1e-4, atol=1e-4)
                        check['passed'] = True
                    except AssertionError as error:
                        check['error'] = str(error)
            finally:
                if output is not None:
                    ttnn.deallocate(output)
                if joined is not None:
                    ttnn.deallocate(joined)
                for value in tensors:
                    ttnn.deallocate(value)
        if len(report['checks']) != 6 or not all(check['passed'] for check in report['checks']):
            raise AssertionError('Learned projection slice fails the unchanged float32 numerical gate')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            if device_weight is not None:
                ttnn.deallocate(device_weight)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
