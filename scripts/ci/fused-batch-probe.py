"""Multi-row fusion check using pinned draft MLP weights as geometry-matched operands, not target quality."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from draft_mlp_fixture import load_mlp
from feature_projection import require_projection_environment
from fused_1d import FusedProjection, native_gate_up_control
from gdn_multitoken_conv import release_owned


def pair_pack(gate, up):
    import torch

    if gate.shape != (17408, 5120) or up.shape != gate.shape or gate.dtype != torch.bfloat16 or up.dtype != gate.dtype:
        raise ValueError('Pinned BF16 17408-by-5120 gate/up weights required')
    gate_parts = [part.T.contiguous() for part in gate.chunk(2, dim=0)]
    up_parts = [part.T.contiguous() for part in up.chunk(2, dim=0)]
    packed = [torch.stack((first.reshape(5120, 272, 32), second.reshape(5120, 272, 32)), dim=2).reshape(5120, 17408)
        for first, second in zip(gate_parts, up_parts, strict=True)]
    return tuple(torch.stack(parts).unsqueeze(1) for parts in (gate_parts, up_parts, packed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-weight-check', action='store_true')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--timing', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, options.hardware)
    if options.timing and not options.hardware:
        parser.error('Latency measurements require allocated hardware')
    if options.hardware and not options.device_weight_check:
        parser.error('Hardware promotion requires byte-exact device weight checks')
    import torch
    import ttnn

    manifest, weights = load_mlp(options.fixture)
    gate, up, packed = pair_pack(weights['layers.0.mlp.gate_proj.weight'], weights['layers.0.mlp.up_proj.weight'])
    report = dict(passed=False, scope=__doc__, fixture=manifest, checks=[],
        backend='hardware' if options.hardware else 'simulator', timings=[],
        timing_scope='Eager paired ABBA projection calls including dispatch and allocation; excludes uploads, validation and deallocation; not traced or full-model latency',
        precision='BF4 gate/up, native LoFi FP32 destination accumulation and BF16 epilogue; not target-model quality')
    packer = Path(os.environ['TT_METAL_HOME']) / 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'
    report['packer_header_sha256'] = hashlib.sha256(packer.read_bytes()).hexdigest()
    report['packer_zero_graft'] = os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') == '1'
    mesh, owned = None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()

        def upload(value, dtype, sharded=False):
            result = ttnn.from_torch(value, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if sharded else ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh))
            owned.append(result)
            return result

        device_gate, device_up, device_packed = [upload(value, ttnn.bfloat4_b, True) for value in (gate, up, packed)]
        report['weight_checks'] = []
        if options.device_weight_check:
            from packed_weight_check import compare_packed_weights, read_comparison
            report['weight_check_backend'] = 'Byte-exact device comparison with coverage and complementary counter validation'
            report['weight_check_sources'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('packed_weight_check.py', 'packed_weight_check.cpp')}
            for offset, separate in enumerate((device_gate, device_up)):
                result = compare_packed_weights(mesh, device_packed, separate, offset, owned)
                for check in read_comparison(ttnn, result, 160 * 272):
                    chip = check['chip']
                    source = (gate, up)[offset][chip, 0]
                    source_packed = packed[chip, 0].reshape(5120, 272, 2, 32)[:, :, offset].reshape(5120, 8704)
                    report['weight_checks'].append(dict(check, source_exact=torch.equal(source, source_packed),
                        projection=('gate', 'up')[offset]))
        for chip in (() if options.device_weight_check else range(2)):
            unpacked = ttnn.to_torch(ttnn.get_device_tensors(device_packed)[chip]).reshape(5120, 272, 2, 32)
            for offset, value in enumerate((device_gate, device_up)):
                observed = unpacked[:, :, offset].reshape(5120, 8704)
                reference = ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).reshape(5120, 8704)
                source = (gate, up)[offset][chip, 0]
                source_packed = packed[chip, 0].reshape(5120, 272, 2, 32)[:, :, offset].reshape(5120, 8704)
                mismatch = observed != reference
                coordinates = mismatch.nonzero()[:8]
                report['weight_checks'].append(dict(chip=chip, projection=('gate', 'up')[offset],
                    source_exact=torch.equal(source, source_packed), exact=torch.equal(observed, reference),
                    mismatches=int(mismatch.sum()), observed_finite=bool(torch.isfinite(observed).all()),
                    reference_finite=bool(torch.isfinite(reference).all()),
                    max_abs=float((observed.float() - reference.float()).abs().max()),
                    examples=[dict(coordinate=coordinate.tolist(), packed=float(observed[tuple(coordinate)]),
                        separate=float(reference[tuple(coordinate)]), source=float(source[tuple(coordinate)]))
                        for coordinate in coordinates]))
                if coordinates.numel():
                    repeated_packed = ttnn.to_torch(ttnn.get_device_tensors(device_packed)[chip]).clone().reshape(5120, 272, 2, 32)
                    repeated_separate = ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone().reshape(5120, 8704)
                    diagnostics = []
                    for coordinate in coordinates:
                        row, column = coordinate.tolist()
                        row_start, column_start = row // 32 * 32, column // 32 * 32
                        tile = source[row_start:row_start + 32, column_start:column_start + 32].contiguous()
                        host_tile = ttnn.from_torch(tile, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
                        quantized = ttnn.to_torch(host_tile).clone()
                        diagnostics.append(dict(coordinate=[row, column],
                            host_quantized=float(quantized[row % 32, column % 32]),
                            repeated_packed=float(repeated_packed[row, column // 32, offset, column % 32]),
                            repeated_separate=float(repeated_separate[row, column])))
                    report['weight_checks'][-1]['readback_diagnostics'] = diagnostics
        if not all(check['source_exact'] and check['exact'] for check in report['weight_checks']):
            raise AssertionError('Pair packing changed BF4 quantization; see weight_checks')
        report['phase'] = 'weight_checks_passed'
        options.output.write_text(json.dumps(report, indent=2))
        kernel = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        report['control_epilogue'] = 'Native gate linear fused SILU before BF16 pack; separate BF16 up; BF16 multiply'
        for rows in (1, 2, 4, 8, 16, 32):
            generator = torch.Generator().manual_seed(3891 + rows)
            inputs = upload(torch.randn((1, 1, rows, 5120), generator=generator).bfloat16(), ttnn.bfloat16)
            report.update(phase='native_projection', active_rows=rows)
            options.output.write_text(json.dumps(report, indent=2))
            expected = native_gate_up_control(ttnn, inputs, device_gate, device_up, kernel, owned)
            operation = FusedProjection(mesh, device_packed, pairs_per_worker=3, token_rows=rows,
                source_root=os.environ['TT_METAL_HOME'])
            report['phase'] = 'fused_projection'
            options.output.write_text(json.dumps(report, indent=2))
            actual = operation(inputs)
            owned.append(actual)
            report.setdefault('kernels', []).append(operation.manifest)
            references = []
            for chip in range(2):
                observed = ttnn.to_torch(ttnn.get_device_tensors(actual)[chip])
                reference = ttnn.to_torch(ttnn.get_device_tensors(expected)[chip])
                if not torch.equal(observed, reference):
                    raise AssertionError(f'Fused multi-row output differs: rows={rows}, chip={chip}, mismatches={int((observed != reference).sum())}')
                report['checks'].append(dict(rows=rows, chip=chip, exact=True))
                references.append(reference.clone())
            if options.timing and rows in (1, 8, 32):
                for block in range(3):
                    samples = dict(control=[], fused=[])
                    for arm in ('control', 'fused', 'fused', 'control'):
                        temporary = []
                        try:
                            ttnn.synchronize_device(mesh)
                            started = time.perf_counter()
                            if arm == 'control':
                                value = native_gate_up_control(ttnn, inputs, device_gate, device_up, kernel, temporary)
                            else:
                                value = operation(inputs)
                                temporary.append(value)
                            ttnn.synchronize_device(mesh)
                            elapsed = (time.perf_counter() - started) * 1000
                            if not math.isfinite(elapsed) or elapsed <= 0:
                                raise AssertionError('Positive finite hardware timing required')
                            for chip in range(2):
                                if not torch.equal(ttnn.to_torch(ttnn.get_device_tensors(value)[chip]), references[chip]):
                                    raise AssertionError('Timed fusion/control output changed')
                            samples[arm].append(elapsed)
                        finally:
                            release_owned(ttnn, temporary)
                    report['timings'].append(dict(rows=rows, block=block, samples_ms=samples,
                        control_ms=sum(samples['control']) / 2, fused_ms=sum(samples['fused']) / 2,
                        both_chips_exact=True))
            options.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(dict(rows=rows, both_chips_exact=True)), flush=True)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    if len(report['checks']) != 12:
        raise AssertionError('All six widths and both chips required')
    if len(report['timings']) != (9 if options.timing else 0):
        raise AssertionError('Three paired timing blocks at T1/T8/T32 required')
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
