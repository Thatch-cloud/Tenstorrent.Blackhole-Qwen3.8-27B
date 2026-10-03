"""Simulator-only streamed T8 projection against native lead and expanded grids; no TG claim."""

import argparse
import json
import math
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from tensix_projection_raw import copy_raw, raw_shape
from tensix_projection_gate import NATIVE_SOURCES, SOURCES, hashes
from tensix_stream_projection import execute_projection, prepare_projection
from tensix_weight_stream import stream_geometry
from tiny_tile_matmul import PROJECTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--projection', required=True, choices=('gate', 'up', 'down'))
    parser.add_argument('--blocks', type=int, default=5)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') != '1':
        parser.error('Explicit scoped simulator packer compatibility graft required')
    geometry = stream_geometry(options.projection, options.blocks)
    native_root = Path(os.environ['TT_METAL_HOME'])
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg

    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__, geometry=geometry,
        weights='Two synthetic compressed patterns, independent data on each chip', token_rows=8,
        compared_rows=32, packer_zero_graft=True, native_compute_unchanged=True,
        sources=hashes(Path(__file__).parent, SOURCES), native_sources=hashes(native_root, NATIVE_SOURCES),
        control_checks=[], eager_checks=[], replay_checks=[], negative_controls=[])
    mesh = None
    owned, transient, pipelines, traces = [], [], [], []
    def progress(stage):
        print(json.dumps(dict(stage=stage, projection=options.projection, blocks=options.blocks)), flush=True)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3)
        patterns = []
        for seed in (1927, 1928):
            generator = torch.Generator().manual_seed(seed)
            patterns.append([torch.randn((1, 1, 8, 2 * geometry['rows']), generator=generator).bfloat16(),
                (torch.randn((1, 1, geometry['rows'], 2 * geometry['width']), generator=generator)
                    / math.sqrt(geometry['rows'])).bfloat16()])
        dtypes = (ttnn.bfloat16, getattr(ttnn, geometry['dtype']))
        payloads = [[ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value, dtype in zip(pattern, dtypes, strict=True)] for pattern in patterns]
        for value, dtype, memory in zip(patterns[0], dtypes, (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG), strict=True):
            owned.append(ttnn.from_torch(value, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                memory_config=memory, mesh_mapper=mapper))
        activation, weight = owned
        for unused in range(2):
            owned.append(ttnn.from_torch(torch.zeros((1, 1, 8, geometry['width']), dtype=torch.bfloat16), device=mesh,
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        outputs = owned[2:]
        for output in outputs:
            pipelines.append(prepare_projection(ttnn, mesh, activation, weight, output, geometry, native_root))
        for value, tile_bytes in ((activation, 2048), (weight, geometry['tile_bytes']), (outputs[0], 2048)):
            owned.append(ttnn.from_torch(torch.zeros(raw_shape(tuple(value.padded_shape), tile_bytes), dtype=torch.int32),
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        raw_activation, raw_weight, raw_output = owned[4:]
        bindings = [addresses(ttnn, value) for value in owned]
        if any(len({binding[chip] for binding in bindings}) != len(owned) for chip in range(2)):
            raise AssertionError('All input, output and raw audit buffers must be disjoint')
        def raw(value, destination, tile_bytes):
            copy_raw(ttnn, value, destination, tile_bytes)
            ttnn.synchronize_device(mesh)
            return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(destination)]
        def input_words():
            return [raw(activation, raw_activation, 2048), raw(weight, raw_weight, geometry['tile_bytes'])]
        def upload(pattern):
            for payload, destination in zip(payloads[pattern], (activation, weight), strict=True):
                ttnn.copy_host_to_device_tensor(payload, destination)
            ttnn.synchronize_device(mesh)
        compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True)
        programs = [create_matmul_1d_decode_progcfg(8, geometry['rows'], geometry['width'], count,
            fused_activation=ttnn.UnaryOpType.SILU if options.projection == 'gate' else None,
            fp32_acc=True, grid_w=11) for count in (PROJECTIONS[options.projection][2], geometry['receivers'])]
        references, inputs = [], []
        progress('allocated')
        for pattern in range(2):
            upload(pattern)
            inputs.append(input_words())
            controls = []
            for program in programs:
                result = ttnn.linear(activation, weight, program_config=program, compute_kernel_config=compute,
                    memory_config=ttnn.L1_MEMORY_CONFIG)
                transient.append(result)
                controls.append(raw(result, raw_output, 2048))
                release_owned(ttnn, transient)
                transient.clear()
            for chip in range(2):
                if not torch.equal(controls[0][chip], controls[1][chip]):
                    raise AssertionError('Expanded native grid changed physical output words relative to the lead')
                report['control_checks'].append(dict(pattern=pattern, chip=chip, exact_all_32_rows=True))
            references.append(controls[0])
            for arm, prepared in enumerate(pipelines):
                execute_projection(ttnn, prepared)
                actual = raw(prepared.output, raw_output, 2048)
                unchanged = input_words()
                for chip in range(2):
                    if not torch.equal(actual[chip], references[pattern][chip]):
                        raise AssertionError('Streamed matmul differs from native physical output words')
                    if any(not torch.equal(current[chip], expected[chip])
                            for current, expected in zip(unchanged, inputs[pattern], strict=True)):
                        raise AssertionError('Streamed matmul changed input or packed weight words')
                    report['eager_checks'].append(dict(pattern=pattern, arm=arm, chip=chip,
                        exact_all_32_rows=True, inputs_unchanged=True))
            progress(f'eager_{pattern}_complete')
        if (any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2))
                or any(torch.equal(reference[0], reference[1]) for reference in references)):
            raise AssertionError('Changed patterns and independent chip results must differ')
        upload(0)
        for prepared in pipelines:
            trace, unused_output = capture_operation(ttnn, mesh, lambda: execute_projection(ttnn, prepared))
            traces.append(trace)
        for repetition, pattern in enumerate((0, 1, 0)):
            upload(pattern)
            for arm, (trace, prepared) in enumerate(zip(traces, pipelines, strict=True)):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = raw(prepared.output, raw_output, 2048)
                unchanged = input_words()
                if [addresses(ttnn, value) for value in owned] != bindings:
                    raise AssertionError('Captured input, output or audit addresses changed')
                for chip in range(2):
                    if not torch.equal(actual[chip], references[pattern][chip]):
                        raise AssertionError('Changed-input captured matmul differs from native physical words')
                    if any(not torch.equal(current[chip], expected[chip])
                            for current, expected in zip(unchanged, inputs[pattern], strict=True)):
                        raise AssertionError('Captured matmul mutated input or packed weight words')
                    report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, arm=arm, chip=chip,
                        exact_all_32_rows=True, inputs_unchanged=True, bindings_stable=True))
                if repetition == 0:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    stale = raw(prepared.output, raw_output, 2048)
                    for chip in range(2):
                        if not torch.equal(stale[chip], references[0][chip]) or torch.equal(stale[chip], references[1][chip]):
                            raise AssertionError('Stale-input negative control failed')
                        report['negative_controls'].append(dict(arm=arm, chip=chip, stale_detected=True))
            progress(f'replay_{repetition}_complete')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in reversed(traces):
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, transient)
                release_owned(ttnn, owned)
                pipelines.clear()
                prepared = None
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            if not report['closed_cleanly']:
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
