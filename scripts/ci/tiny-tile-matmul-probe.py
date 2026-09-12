"""Simulator-only T8 quantized target-shape projections, including retile and live trace replay."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from tiny_tile_matmul import PROJECTIONS, project
from tiny_tile_dma import copy_live_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projection', choices=tuple(PROJECTIONS), default='gate')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg

    input_width, output_width, cores, dtype_name, silu = PROJECTIONS[options.projection]
    report = dict(passed=False, scope=__doc__, projection=options.projection, rows=8, input_width=input_width,
        output_width_per_chip=output_width, weight_dtype=dtype_name, activation_tiles=[32, 16],
        weights='Deterministic synthetic local matrices; not learned target weights or coding quality',
        timing_scope='No simulator timing or throughput claim', dma_checks=[], eager_checks=[], trace_checks=[], negative_controls=[],
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('tiny_tile_matmul.py', 'tiny-tile-matmul-probe.py', 'tiny_tile_dma.py', 'tiny_tile_dma.cpp', 'attention_batch.py')})
    native = Path(os.environ['TT_METAL_HOME'])
    report['native_sources'] = {name: hashlib.sha256((native / name).read_bytes()).hexdigest() for name in (
        'models/demos/blackhole/qwen36/tt/tp_common.py',
        'ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_device_operation.cpp',
        'ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp',
        'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h')}
    report['packer_zero_graft'] = os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') == '1'
    report['packer_l1_acc'] = True
    report['torch_version'] = torch.__version__
    persistent, temporary, captured, traces = [], [], [], []
    mesh = None
    def retain(values, value):
        values.append(value)
        return value
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)
    def read(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=67108864)
        mesh.enable_program_cache()
        report['grid'] = [mesh.compute_with_storage_grid_size().x, mesh.compute_with_storage_grid_size().y]
        if report['grid'] != [11, 10]:
            raise ValueError('Match the qualified worker-dispatch grid')
        generator = torch.Generator().manual_seed(1641)
        host_weight = (torch.randn((1, 1, input_width, output_width * 2), generator=generator)
            / math.sqrt(input_width)).bfloat16()
        weight = retain(persistent, ttnn.from_torch(host_weight, device=mesh, dtype=getattr(ttnn, dtype_name),
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3)))
        del host_weight
        patterns = [torch.randn((1, 1, 8, input_width), generator=generator).bfloat16() for unused in range(2)]
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in patterns]
        source = retain(persistent, ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
        small_source = retain(persistent, ttnn.from_torch(torch.zeros_like(patterns[0]), device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((16, 32)), memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
        restored = retain(persistent, ttnn.empty((1, 1, 8, output_width), device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG))
        roundtrip = retain(persistent, ttnn.empty((1, 1, 8, input_width), device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG))
        bindings = [addresses(ttnn, value) for value in persistent]
        program = create_matmul_1d_decode_progcfg(8, input_width, output_width, cores,
            fused_activation=ttnn.UnaryOpType.SILU if silu else None, grid_w=11)
        report['native_program'] = str(program)
        compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        def retile(value, height):
            return copy_live_rows(mesh, value, small_source if height == 16 else restored)
        def execute(tiny, storage):
            return project(ttnn, source, weight, program, compute, lambda value: retain(storage, value), tiny=tiny,
                retile=retile if tiny else None)
        references = []
        progress('weights_ready')
        for pattern, host_input in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(host_input, source)
            copy_live_rows(mesh, source, small_source)
            copy_live_rows(mesh, small_source, roundtrip)
            for chip in range(2):
                for height, converted in ((16, small_source), (32, roundtrip)):
                    if not torch.equal(read(converted, chip).view(torch.int16), patterns[pattern].view(torch.int16)):
                        raise AssertionError('Tile DMA changed active BF16 bits')
                    report['dma_checks'].append(dict(pattern=pattern, chip=chip, tile_height=height, exact_bits=True))
            progress(f'dma_pattern_{pattern}_passed')
            baseline = execute(False, temporary)
            expected = [read(baseline, chip) for chip in range(2)]
            references.append(expected)
            candidate = execute(True, temporary)
            for chip in range(2):
                if not torch.equal(read(candidate, chip), expected[chip]):
                    raise AssertionError(f'16-row projection differs from native rounding: pattern={pattern} chip={chip}')
                if not torch.equal(read(source, chip), patterns[pattern]):
                    raise AssertionError('Projection changed caller-owned input')
                report['eager_checks'].append(dict(pattern=pattern, chip=chip, exact=True))
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, temporary)
            temporary.clear()
            progress(f'eager_pattern_{pattern}_passed')
        for chip in range(2):
            if torch.equal(references[0][chip], references[1][chip]):
                raise AssertionError('Changed-input negative control is not discriminating')
            report['negative_controls'].append(dict(chip=chip, stale_input_detected=True))
        outputs = []
        for tiny in (False, True):
            trace, output = capture_operation(ttnn, mesh, lambda tiny=tiny: execute(tiny, captured))
            traces.append(trace)
            outputs.append(output)
        for pattern, host_input in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(host_input, source)
            for arm, (trace, output) in enumerate(zip(traces, outputs, strict=True)):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for chip in range(2):
                    if not torch.equal(read(output, chip), references[pattern][chip]):
                        raise AssertionError('Captured projection used stale input or changed numerical results')
                    report['trace_checks'].append(dict(pattern=pattern, arm=arm, chip=chip, exact=True))
            if bindings != [addresses(ttnn, value) for value in persistent]:
                raise AssertionError('Input or weight storage moved under live traces')
            if any(not torch.equal(read(source, chip), patterns[pattern]) for chip in range(2)):
                raise AssertionError('Captured projection changed caller-owned input')
            progress(f'trace_pattern_{pattern}_passed')
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            release_owned(ttnn, temporary + captured + persistent)
            ttnn.close_mesh_device(mesh)
    report['passed'] = True
    progress('complete')


if __name__ == '__main__':
    main()
