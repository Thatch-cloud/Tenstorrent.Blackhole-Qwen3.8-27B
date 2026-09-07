"""Layer-zero learned SwiGLU on synthetic inputs; no convolution, residual or full-drafter claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_mlp import split_mlp_weights, swiglu_reference, swiglu_device
from draft_mlp_fixture import load_mlp
from feature_collective import gather_add_projection
from feature_normalization import bf16_ulp_distance
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from projection_rounding import grouped_projection_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--hardware', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if not options.hardware and os.environ.get('QWEN_SIM_SHARED_BDF') != '1':
        parser.error('Connected simulator required')
    import torch
    import ttnn
    from models.tt_transformers.tt.ccl import TT_CCL

    manifest, weights = load_mlp(options.fixture)
    shards = split_mlp_weights(*(weights[f'layers.0.mlp.{name}_proj.weight'] for name in ('gate', 'up', 'down')))
    report = dict(passed=False, scope=__doc__, checkpoint=manifest, checks=[], block_rows=8,
        backend='hardware' if options.hardware else 'simulator',
        tolerance=dict(projection_rtol=1e-4, projection_atol=1e-4, activation_ulps=2),
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('learned-mlp-probe.py', 'draft_mlp.py', 'draft_mlp_fixture.py', 'feature_collective.py', 'projection_rounding.py')})
    mesh = None
    tensors = []

    def retain(value):
        tensors.append(value)
        return value

    def checkpoint(stage, **details):
        report['progress'] = dict(stage=stage, **details)
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['progress']), flush=True)

    def upload(value, sharded=False):
        return retain(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
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
        ttnn.synchronize_device(mesh)
        checkpoint('device_pipeline_complete')
        partials = [host(partial, chip) for chip in range(2)]
        for chip in range(2):
            for index, name in enumerate(('gate', 'up')):
                expected = grouped_projection_reference(inputs[..., :8, :], shards[chip][index], destination_rounding=True)
                actual = host(projections[index], chip)[..., :8, :].double()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
                report['checks'].append(dict(chip=chip, stage=name, max_error=float((actual - expected).abs().max())))
                checkpoint('projection_reference_complete', chip=chip, projection=name)
            expected_activation = swiglu_reference(*(host(value, chip) for value in projections))
            actual_activation = host(activated, chip)
            distance = int(bf16_ulp_distance(actual_activation, expected_activation).max())
            if distance > 2:
                raise AssertionError(f'SwiGLU activation differs by {distance} BF16 ULPs')
            expected_down = grouped_projection_reference(actual_activation[..., :8, :], shards[chip][2], destination_rounding=True)
            torch.testing.assert_close(partials[chip][..., :8, :].double(), expected_down, rtol=1e-4, atol=1e-4)
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
