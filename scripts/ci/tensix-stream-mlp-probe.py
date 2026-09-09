"""Full-size pooled streamed MLP simulation, two synthetic weight fixtures and shared workspace; no CCL or TG."""

import argparse
import json
import math
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from tensix_mlp_gate import COMPONENTS, NATIVE_SOURCES, SOURCES, hashes
from tensix_projection_raw import copy_raw, raw_shape
from tensix_stream_mlp import StreamBufferPool, execute_from_dram, prepare_mlp
from tiny_mlp import execute
from tiny_tile_matmul import PROJECTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') != '1':
        parser.error('Explicit scoped simulator packer compatibility graft required')
    native_root = Path(os.environ['TT_METAL_HOME'])
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg

    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__, rows=8, compared_rows=32,
        fixtures=2, components=list(COMPONENTS), packer_zero_graft=True, shared_pool=True, shared_workspace=True,
        full_mlp=True, dram_boundary=True,
        sources=hashes(Path(__file__).parent, SOURCES), native_sources=hashes(native_root, NATIVE_SOURCES),
        control_checks=[], eager_checks=[], replay_checks=[], input_checks=[], negative_controls=[], fixture_checks=[])
    mesh, pool = None, None
    owned, transient, prepared_mlp, traces = [], [], [], []
    def retain(storage, value):
        storage.append(value)
        return value
    def progress(stage):
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3)
        weights = []
        for fixture in range(2):
            generator = torch.Generator().manual_seed(7250 + fixture)
            local = {}
            for name, (inner, width, unused_cores, dtype, unused_silu) in PROJECTIONS.items():
                host = (torch.randn((1, 1, inner, width * 2), generator=generator) / math.sqrt(inner)).bfloat16()
                local[name] = retain(owned, ttnn.from_torch(host, device=mesh, dtype=getattr(ttnn, dtype),
                    layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
            weights.append(local)
        patterns = [torch.randn((1, 1, 8, 10240), generator=torch.Generator().manual_seed(seed)).bfloat16()
            for seed in (3938, 3939)]
        payloads = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in patterns]
        source = retain(owned, ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        local_input = retain(owned, ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
        workspace = {name: retain(owned, ttnn.from_torch(
            torch.zeros((1, 1, 8, 5120 if name == 'partial' else 8704), dtype=torch.bfloat16), device=mesh,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))) for name in COMPONENTS}
        pool = StreamBufferPool(ttnn, mesh)
        for fixture in range(2):
            prepared_mlp.append(prepare_mlp(ttnn, mesh, local_input, weights[fixture], workspace, pool, native_root))
        report['pool_buffers'] = len(pool.entries)
        borrowed = [source, *(local[name] for local in weights for name in ('gate', 'up', 'down'))]
        report['weight_bindings'] = [addresses(ttnn, value) for value in borrowed[1:]]
        raw_buffers = {}
        for value, tile_bytes in ((source, 2048), (weights[0]['gate'], 576), (weights[0]['down'], 1088),
                (workspace['gate'], 2048)):
            shape = raw_shape(tuple(value.padded_shape), tile_bytes)
            raw_buffers[shape] = retain(owned, ttnn.from_torch(torch.zeros(shape, dtype=torch.int32), device=mesh,
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        bindings = [addresses(ttnn, value) for value in owned]
        if any(len({binding[chip] for binding in bindings}) != len(owned) for chip in range(2)):
            raise AssertionError('All weights, workspace and raw audit buffers must be disjoint')
        def upload(pattern):
            ttnn.copy_host_to_device_tensor(payloads[pattern], source)
            ttnn.synchronize_device(mesh)
        def raw(value):
            tile_bytes = 576 if value.dtype == ttnn.bfloat4_b else 1088 if value.dtype == ttnn.bfloat8_b else 2048
            destination = raw_buffers[raw_shape(tuple(value.padded_shape), tile_bytes)]
            copy_raw(ttnn, value, destination, tile_bytes)
            ttnn.synchronize_device(mesh)
            return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(destination)]
        weight_words = [raw(value) for value in borrowed[1:]]
        source_words = []
        for pattern in range(2):
            upload(pattern)
            source_words.append(raw(source))
        def audit_inputs(phase, repetition, pattern):
            for index, (value, expected) in enumerate(zip(borrowed, [source_words[pattern], *weight_words], strict=True)):
                current = raw(value)
                for chip in range(2):
                    if not torch.equal(current[chip], expected[chip]):
                        raise AssertionError('MLP mutated a borrowed input or packed weight word')
                    report['input_checks'].append(dict(phase=phase, repetition=repetition, tensor=index, chip=chip,
                        packed_words_unchanged=True))
        compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True)
        programs = [{name: create_matmul_1d_decode_progcfg(8, inner, width,
            (80 if name == 'down' else 68) if expanded else cores,
            fused_activation=ttnn.UnaryOpType.SILU if silu else None, fp32_acc=True, grid_w=11)
            for name, (inner, width, cores, unused_dtype, silu) in PROJECTIONS.items()} for expanded in (False, True)]
        references = []
        progress('allocated_and_inputs_recorded')
        for pattern in range(2):
            upload(pattern)
            references.append([])
            for fixture in range(2):
                controls = []
                for program in programs:
                    interleaved = retain(transient, ttnn.to_memory_config(source, ttnn.L1_MEMORY_CONFIG))
                    result = execute(ttnn, interleaved, weights[fixture], program, compute,
                        lambda value: retain(transient, value), tiny=False)
                    controls.append({name: raw(result[name]) for name in COMPONENTS})
                    release_owned(ttnn, transient)
                    transient.clear()
                references[pattern].append(controls[0])
                actual = execute_from_dram(ttnn, source, prepared_mlp[fixture])
                for component, name in enumerate(COMPONENTS):
                    current = raw(actual[name])
                    for chip in range(2):
                        if not torch.equal(controls[0][name][chip], controls[1][name][chip]):
                            raise AssertionError('Expanded native MLP differs from the lead')
                        if not torch.equal(current[chip], controls[0][name][chip]):
                            raise AssertionError('Streamed MLP differs from native physical output words')
                        coordinate = dict(pattern=pattern, fixture=fixture, component=component, chip=chip,
                            exact_all_32_rows=True)
                        report['control_checks'].append(coordinate)
                        report['eager_checks'].append(dict(coordinate))
                progress(f'eager_pattern_{pattern}_fixture_{fixture}_complete')
            audit_inputs(0, pattern, pattern)
        for fixture in range(2):
            for chip in range(2):
                if torch.equal(references[0][fixture]['partial'][chip], references[1][fixture]['partial'][chip]):
                    raise AssertionError('Changed-input fixtures must produce different outputs')
        for chip in range(2):
            if torch.equal(references[0][0]['partial'][chip], references[0][1]['partial'][chip]):
                raise AssertionError('Different weight buffers must have distinguishable outputs')
            report['fixture_checks'].append(dict(chip=chip, different_weights_detected=True))
        upload(0)
        for fixture in range(2):
            trace, unused_result = capture_operation(ttnn, mesh,
                lambda fixture=fixture: execute_from_dram(ttnn, source, prepared_mlp[fixture]))
            traces.append(trace)
        for repetition, pattern in enumerate((0, 1, 0)):
            upload(pattern)
            for fixture, trace in enumerate(traces):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if [addresses(ttnn, value) for value in owned] != bindings or len(pool.entries) != 2:
                    raise AssertionError('MLP changed stable bindings or allocated per-layer FIFOs')
                for component, name in enumerate(COMPONENTS):
                    current = raw(workspace[name])
                    for chip in range(2):
                        if not torch.equal(current[chip], references[pattern][fixture][name][chip]):
                            raise AssertionError('Captured shared-workspace MLP differs from native output')
                        report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, fixture=fixture,
                            component=component, chip=chip, exact_all_32_rows=True, bindings_stable=True, pool_reused=True))
                if repetition == 0:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    stale = raw(workspace['partial'])
                    for chip in range(2):
                        if (not torch.equal(stale[chip], references[0][fixture]['partial'][chip])
                                or torch.equal(stale[chip], references[1][fixture]['partial'][chip])):
                            raise AssertionError('Stale-input negative control failed')
                        report['negative_controls'].append(dict(fixture=fixture, chip=chip, stale_detected=True))
            audit_inputs(1, repetition, pattern)
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
                prepared_mlp.clear()
                if pool is not None:
                    pool.entries.clear()
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            if not report['closed_cleanly']:
                report['passed'] = False
            options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
