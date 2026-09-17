"""Diagnose the failed learned Markov score at vocabulary columns1312:1376; not a qualification gate."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from dspark_markov_fixture import load_fixture
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from projection_rounding import grouped_projection_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    spec = importlib.util.spec_from_file_location('dspark_native_source', Path(__file__).with_name('dspark-markov-probe.py'))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    root = Path(os.environ['TT_METAL_HOME'])
    manifest, predecessor, successor = load_fixture(options.fixture)
    latent = predecessor[1596:1597].contiguous()
    weight = successor[1312:1376].T.contiguous()
    base = (torch.randn((1, 1, 7, 248320), generator=torch.Generator().manual_seed(38256)) / 8)[..., :1, 1312:1376].contiguous()
    exact = latent.float() @ weight.float()
    grouped = grouped_projection_reference(latent, weight, destination_rounding=True, fidelity_span=32).float()
    report = dict(completed=False, closed_cleanly=False, accuracy_qualified=False, eligible_for_hardware=False,
        scope=__doc__, fixture=manifest, native_sources=native.fingerprints(root), checks=[],
        sources={name: native.digest(Path(__file__).with_name(name)) for name in
            ('dspark-markov-rounding-probe.py', 'dspark_markov_device.py', 'dspark_markov_fixture.py', 'projection_rounding.py')})
    mesh, owned = None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(value, dtype):
            tensor = ttnn.from_torch(value, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            owned.append(tensor)
            return tensor

        device_latent = upload(latent.reshape(1, 1, 1, 256), ttnn.bfloat16)
        device_weight = upload(weight.reshape(1, 1, 256, 64), ttnn.bfloat16)
        device_base = upload(base, ttnn.float32)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(2, 1),
            in0_block_w=1, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        bias = ttnn.matmul(device_latent, device_weight, dtype=ttnn.float32, program_config=program,
            compute_kernel_config=kernel, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.append(bias)
        default = ttnn.add(device_base, bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.append(default)
        explicit = ttnn.add(device_base, bias, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.append(explicit)
        ttnn.synchronize_device(mesh)
        saved = dict(latent=latent, weight=weight, base=base, exact=exact, grouped=grouped, chips=[])
        for chip in range(2):
            actual_bias, default_scores, explicit_scores = [ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).float().reshape(1, 64)
                for tensor in (bias, default, explicit)]
            if any(not torch.isfinite(value).all() for value in (actual_bias, default_scores, explicit_scores)):
                raise ValueError('Nonfinite diagnostic output')
            expected_scores = base.reshape(1, 64) + exact
            additive = base.reshape(1, 64) + actual_bias
            report['checks'].append(dict(chip=chip, columns=[1312, 1376], anchor=1596,
                matmul_vs_fp32_max_abs=float((actual_bias - exact).abs().max()),
                matmul_vs_grouped_max_abs=float((actual_bias - grouped).abs().max()),
                matmul_grouped_exact=bool(torch.equal(actual_bias, grouped)),
                default_add_vs_fp32_max_abs=float((default_scores - additive).abs().max()),
                explicit_add_vs_fp32_max_abs=float((explicit_scores - additive).abs().max()),
                dtype_flag_changes_result=not torch.equal(default_scores, explicit_scores),
                whole_vs_fp32_max_abs=float((default_scores - expected_scores).abs().max()),
                whole_fp32_gate_passed=bool(torch.isclose(default_scores, expected_scores, rtol=1e-4, atol=1e-4).all()),
                global1340_error=float((default_scores - expected_scores)[0, 1340 - 1312])))
            saved['chips'].append(dict(bias=actual_bias, default_scores=default_scores, explicit_scores=explicit_scores))
        payload = options.output.with_suffix('.operands.pt')
        torch.save(saved, payload)
        report['operands_sha256'] = hashlib.sha256(payload.read_bytes()).hexdigest()
        report['completed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
            report['native_sources_after'] = native.fingerprints(root)
            if report['native_sources_after'] != report['native_sources']:
                raise ValueError('Native sources changed during diagnostic')
        finally:
            if not report['closed_cleanly'] or report.get('native_sources_after') != report['native_sources']:
                report['completed'] = False
            options.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({key: report.get(key) for key in ('completed', 'closed_cleanly', 'checks', 'error')}), flush=True)


if __name__ == '__main__':
    main()
