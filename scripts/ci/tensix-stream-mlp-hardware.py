"""Real-weight pooled streamed T8 MLP ABBA including DRAM input copy and the native four-link TP2 collective."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from sampling_link_policy import audit
from tensix_mlp_collective import reduce_partial
from tensix_mlp_gate import COMPONENTS, NATIVE_SOURCES, ORIGINAL_PACKER, PACKER, SOURCES, hashes, qualify
from tensix_mlp_hardware_gate import HARDWARE_SOURCES, MODEL_SOURCES, read_prerequisite
from tensix_mlp_view_gate import NATIVE_SOURCES as VIEW_NATIVE_SOURCES, prerequisite as view_prerequisite
from tensix_mlp_weight_views import tensor_spec, weight_views
from tensix_mlp_profile import MlpTraceProfile, require_profile_mode
from tensix_projection_raw import copy_raw, raw_shape
from tensix_stream_mlp import StreamBufferPool, execute_from_dram, prepare_mlp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator-report', type=Path, required=True)
    parser.add_argument('--simulator-exit-status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--profile', action='store_true')
    options = parser.parse_args()
    require_projection_environment(os.environ, True)
    require_profile_mode(os.environ, options.profile)
    native_root, root = Path(os.environ['TT_METAL_HOME']), Path(__file__).parent
    report = dict(passed=False, closed_cleanly=False, backend='hardware', scope=__doc__, rows=8, streams=1, layer=0,
        collective_links=4, repeats_per_sample=50, seeds=[1659, 2670, 3781], dram_boundary=True,
        native_collective=True, all_samples_retained=True, eager_checks=[], trace_checks=[], negative_controls=[],
        input_checks=[], timed_checks=[], blocks=[], profile_checks=[],
        instrumented_timing=options.profile, correctness_only=options.profile,
        timing_scope='Captured DRAM-to-L1 copy, gate/up/product/down and native TP2 collective; upload and validation excluded')
    if options.profile:
        report.update(scope='Device attribution of qualified real-weight MLP traces; not a performance benchmark',
            repeats_per_sample=1, all_samples_retained=False,
            timing_scope='One instrumented complete MLP trace per marker; profiler dumps and validation outside markers')
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)
    try:
        report['sources'] = hashes(root, SOURCES)
        report['native_sources'] = hashes(native_root, NATIVE_SOURCES)
        report['hardware_sources'] = hashes(root, HARDWARE_SOURCES)
        report['model_sources'] = hashes(native_root, MODEL_SOURCES)
        prerequisite = read_prerequisite(options.simulator_report, options.simulator_exit_status)
        report['simulator_report_sha256'] = hashlib.sha256(options.simulator_report.read_bytes()).hexdigest()
        report['simulator_exit_sha256'] = hashlib.sha256(options.simulator_exit_status.read_bytes()).hexdigest()
        if (report['native_sources'][PACKER] != ORIGINAL_PACKER or report['model_sources'] != MODEL_SOURCES
                or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT')):
            raise ValueError('Original hardware packer and reviewed native MLP/CCL required; no simulator graft')
        report['simulator_gate'] = qualify(prerequisite, report['sources'], report['native_sources'])
        report['view_prerequisite'] = view_prerequisite(root, hashes(native_root, VIEW_NATIVE_SOURCES))
        report['fabric_sources'] = audit(native_root, os.environ)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        for name in dict.fromkeys((*NATIVE_SOURCES, *VIEW_NATIVE_SOURCES)):
            source = native_root / name
            if source.is_file():
                destination = options.output.parent / 'tensix-mlp-native-sources' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
        progress('preflight_failed')
        raise
    if options.preflight:
        report['preflight_passed'] = True
        progress('preflight_complete_no_device_opened')
        return
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_mlp_layer
    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    mesh, pool, prepared = None, None, None
    persistent, transient, captured, traces = [], [], [], []
    def retain(storage, value):
        storage.append(value)
        return value
    def read(value):
        return [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(value)]
    def compare(actual, expected):
        if len(actual) != 2 or len(expected) != 2:
            raise AssertionError('Both chips must be compared')
        for current, reference in zip(actual, expected, strict=True):
            if not torch.equal(current.view(torch.int16), reference.view(torch.int16)):
                raise AssertionError('Complete streamed MLP differs from the native forward')
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=65536)
        if (args.dim, args.hidden_dim, args.num_devices, args.decode_grid_w) != (5120, 17408, 2, 11):
            raise ValueError('Pinned model and two-card worker grid required')
        collectives = TT_CCL(mesh)
        report['native_requested_links'] = {name: collectives.get_num_links(axis)
            for name, axis in (('default', None), ('axis0', 0), ('axis1', 1))}
        def four_links(cluster_axis=None):
            if cluster_axis is not None and (type(cluster_axis) is not int or cluster_axis not in (0, 1)):
                raise ValueError('Qualified pair collective axis required')
            return 4
        collectives.get_num_links = four_links
        report['matched_requested_links'] = {name: collectives.get_num_links(axis)
            for name, axis in (('default', None), ('axis0', 0), ('axis1', 1))}
        state = load_mlp_layer(args.CKPT_DIR, 0)
        mlp = Qwen36MLP(mesh, state, None, args=args, tt_ccl=collectives)
        del state
        compute = mlp.compute_kernel_config_decode
        if (not mlp._mlp_1d_decode or mlp._dram_sharded or compute.math_fidelity != ttnn.MathFidelity.LoFi
                or not compute.fp32_dest_acc_en or not compute.packer_l1_acc or not compute.math_approx_mode):
            raise ValueError('Unchanged interleaved native LoFi FP32/approximate matmul with L1 accumulation required')
        native_weights = dict(gate=mlp.weights.w1, up=mlp.weights.w3, down=mlp.weights.w2)
        report['native_weights'] = {name: tensor_spec(ttnn, value) for name, value in native_weights.items()}
        progress('native_weights_loaded')
        weights, report['weight_views'] = weight_views(ttnn, native_weights)
        progress('weight_views_validated')
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        patterns = [torch.randn((1, 1, 8, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()
            for seed in report['seeds']]
        payloads = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in patterns]
        source, local_input = [retain(persistent, ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=memory, mesh_mapper=mapper))
            for memory in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)]
        workspace = {name: retain(persistent, ttnn.from_torch(
            torch.zeros((1, 1, 8, 5120 if name == 'partial' else 8704), dtype=torch.bfloat16), device=mesh,
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=mapper)) for name in COMPONENTS}
        pool = StreamBufferPool(ttnn, mesh)
        prepared = prepare_mlp(ttnn, mesh, local_input, weights, workspace, pool, native_root)
        report['pool_buffers'] = len(pool.entries)
        borrowed = [source, *(weights[name] for name in ('gate', 'up', 'down'))]
        raw_buffers = {}
        for value, tile_bytes in ((source, 2048), (weights['gate'], 576), (weights['down'], 1088)):
            shape = raw_shape(tuple(value.padded_shape), tile_bytes)
            raw_buffers[shape] = retain(persistent, ttnn.from_torch(torch.zeros(shape, dtype=torch.int32), device=mesh,
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        bindings = [addresses(ttnn, value) for value in persistent + list(weights.values())]
        if any(len({binding[chip] for binding in bindings}) != len(bindings) for chip in range(2)):
            raise AssertionError('Input, weights and persistent workspace must not alias')
        def raw(value):
            tile_bytes = 576 if value.dtype == ttnn.bfloat4_b else 1088 if value.dtype == ttnn.bfloat8_b else 2048
            destination = raw_buffers[raw_shape(tuple(value.padded_shape), tile_bytes)]
            copy_raw(ttnn, value, destination, tile_bytes)
            ttnn.synchronize_device(mesh)
            return read(destination)
        weight_words = [raw(value) for value in borrowed[1:]]
        def forward(candidate, storage):
            if not candidate:
                return retain(storage, mlp.forward(source))
            result = execute_from_dram(ttnn, source, prepared)
            return retain(storage, reduce_partial(ttnn, mesh, collectives, result['partial'], args.ccl_topology()))
        references, source_words = [], []
        progress('weights_ready')
        for pattern, payload in enumerate(payloads):
            ttnn.copy_host_to_device_tensor(payload, source)
            source_words.append(raw(source))
            references.append(read(forward(False, transient)))
            compare(read(forward(True, transient)), references[-1])
            report['eager_checks'].extend(dict(pattern=pattern, chip=chip, exact=True) for chip in range(2))
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, transient)
            transient.clear()
        if any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2)):
            raise AssertionError('Changed-input fixtures must distinguish stale output')
        outputs = []
        for candidate in (False, True):
            trace, output = capture_operation(ttnn, mesh, lambda candidate=candidate: forward(candidate, captured))
            traces.append(trace)
            outputs.append(output)
        profiler = MlpTraceProfile(ttnn, mesh, traces) if options.profile else None
        def timed(arm):
            ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for repeat in range(report['repeats_per_sample']):
                ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = (time.perf_counter() - started) * 1000 / report['repeats_per_sample']
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise AssertionError('Positive finite timing required')
            return elapsed
        for pattern, payload in enumerate(payloads):
            ttnn.copy_host_to_device_tensor(payload, source)
            for arm, trace in enumerate(traces):
                if profiler is not None:
                    profiler.replay(arm, pattern, 'audit')
                else:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                compare(read(outputs[arm]), references[pattern])
                report['trace_checks'].extend(dict(pattern=pattern, arm=arm, chip=chip, exact=True) for chip in range(2))
                if pattern == 0:
                    if profiler is not None:
                        profiler.replay(arm, pattern, 'stale')
                    else:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    stale = read(outputs[arm])
                    compare(stale, references[0])
                    for chip in range(2):
                        if torch.equal(stale[chip], references[1][chip]):
                            raise AssertionError('Stale-input negative control failed')
                        report['negative_controls'].append(dict(arm=arm, chip=chip, stale_detected=True))
            if profiler is not None:
                for sample, arm in enumerate((0, 1, 1, 0)):
                    profiler.replay(arm, pattern, 'measurement', sample)
                    compare(read(outputs[arm]), references[pattern])
                    report['profile_checks'].extend(dict(pattern=pattern, sample=sample, chip=chip, exact=True)
                        for chip in range(2))
            for block in range(0 if options.profile else 3):
                samples = []
                for sample, arm in enumerate((0, 1, 1, 0)):
                    samples.append(timed(arm))
                    compare(read(outputs[arm]), references[pattern])
                    report['timed_checks'].extend(dict(pattern=pattern, block=block, sample=sample, chip=chip, exact=True)
                        for chip in range(2))
                control, candidate = statistics.mean((samples[0], samples[3])), statistics.mean(samples[1:3])
                report['blocks'].append(dict(pattern=pattern, block=block, samples_ms=samples,
                    order=['control', 'candidate', 'candidate', 'control'], control_ms=control,
                    candidate_ms=candidate, ratio=control / candidate))
            if bindings != [addresses(ttnn, value) for value in persistent + list(weights.values())] or len(pool.entries) != 2:
                raise AssertionError('Persistent input, workspace, weights or pool changed')
            for index, (value, expected) in enumerate(zip(borrowed, [source_words[pattern], *weight_words], strict=True)):
                current = raw(value)
                for chip in range(2):
                    if not torch.equal(current[chip], expected[chip]):
                        raise AssertionError('Complete MLP changed a raw input or packed weight word')
                    report['input_checks'].append(dict(pattern=pattern, tensor=index, chip=chip,
                        packed_words_unchanged=True, bindings_stable=True, pool_reused=True))
            progress(f'pattern_{pattern}_passed')
        if profiler is not None:
            report['profile'] = profiler.summary()
            report['eligible_for_full_model_gate'] = False
        else:
            report['control_ms'] = statistics.mean(block['control_ms'] for block in report['blocks'])
            report['candidate_ms'] = statistics.mean(block['candidate_ms'] for block in report['blocks'])
            report['eligible_for_full_model_gate'] = all(block['ratio'] > 1.02 for block in report['blocks'])
        report['native_weights_after'] = {name: tensor_spec(ttnn, value) for name, value in native_weights.items()}
        if report['native_weights_after'] != report['native_weights']:
            raise AssertionError('Native control weight metadata or storage changed during the experiment')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        progress('failed')
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                prepared = None
                if pool is not None:
                    pool.entries.clear()
                release_owned(ttnn, transient + captured + persistent)
                ttnn.close_mesh_device(mesh)
            report['closed_cleanly'] = True
        finally:
            options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    progress('complete')


if __name__ == '__main__':
    main()
