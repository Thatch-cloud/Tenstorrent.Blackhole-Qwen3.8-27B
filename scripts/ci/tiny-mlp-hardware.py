"""Real-weight T8 MLP ABBA, including boundary conversions and native TP2 collective; not model TG."""

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
from tiny_mlp import execute
from tiny_mlp_gate import NATIVE_SOURCES, SOURCES, qualify
from tiny_tile_dma import copy_live_rows
from tiny_tile_product import multiply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, True)
    native_root = Path(os.environ['TT_METAL_HOME'])
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES}
    native_sources = {name: hashlib.sha256((native_root / name).read_bytes()).hexdigest() for name in NATIVE_SOURCES}
    prerequisite = json.loads(options.simulator_report.read_text())
    report = dict(passed=False, scope=__doc__, layer=0, rows=8, streams=1,
        input='Replicated BF16 DRAM; both arms include native DRAM-to-L1 conversion',
        precision='Unchanged BF4 gate/up, BF8 down, LoFi FP32 matmul accumulation with packer L1 accumulation',
        seeds=[1659, 2670, 3781], repeats_per_sample=50, eager_checks=[], trace_checks=[], blocks=[],
        sources=sources, native_sources=native_sources,
        simulator_gate=qualify(prerequisite, sources, native_sources),
        simulator_report_sha256=hashlib.sha256(options.simulator_report.read_bytes()).hexdigest(),
        hardware_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fabric_sources=audit(native_root, os.environ), collective_links=4,
        timing_scope='Captured complete MLP including both DMA boundaries and native reduce-scatter; upload, capture and validation excluded')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import load_mlp_layer
    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

    mesh, persistent, transient, captured, traces = None, [], [], [], []
    def retain(storage, value):
        storage.append(value)
        return value
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)
    def read(value):
        parts = ttnn.get_device_tensors(value)
        if len(parts) != 2:
            raise AssertionError('Both chip outputs required')
        return [ttnn.to_torch(part).clone() for part in parts]
    def exact(actual, expected):
        return all(torch.equal(found.view(torch.int16), reference.view(torch.int16))
            for found, reference in zip(actual, expected, strict=True))
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        args = Qwen36ModelArgs(mesh, max_batch_size=8, max_seq_len=65536)
        if (args.dim, args.hidden_dim, args.num_devices, args.decode_grid_w) != (5120, 17408, 2, 11):
            raise ValueError('Pinned model and two-card worker grid required')
        collectives = TT_CCL(mesh)
        def four_links(cluster_axis=None):
            if cluster_axis not in (None, 0, 1):
                raise ValueError('Qualified pair collective axis required')
            return 4
        collectives.get_num_links = four_links
        state = load_mlp_layer(args.CKPT_DIR, 0)
        mlp = Qwen36MLP(mesh, state, None, args=args, tt_ccl=collectives)
        del state
        if not mlp._mlp_1d_decode or mlp._dram_sharded:
            raise ValueError('Qualified interleaved 1D decode MLP required')
        compute = mlp.compute_kernel_config_decode
        if (compute.math_fidelity != ttnn.MathFidelity.LoFi
                or not compute.fp32_dest_acc_en or not compute.packer_l1_acc):
            raise ValueError('Unchanged native LoFi FP32 and packer L1 accumulation required')
        weights = dict(gate=mlp.weights.w1, up=mlp.weights.w3, down=mlp.weights.w2)
        programs = dict(gate=args.mlp_w1_decode_1d_progcfg, up=args.mlp_w3_decode_1d_progcfg,
            down=args.mlp_w2_decode_1d_progcfg)
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        patterns = [torch.randn((1, 1, 8, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()
            for seed in report['seeds']]
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in patterns]
        source = retain(persistent, ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        small, hidden = [retain(persistent, ttnn.from_torch(torch.zeros((1, 1, 8, width), dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, tile=ttnn.Tile((16, 32)), layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper)) for width in (5120, 8704)]
        bindings = [addresses(ttnn, value) for value in persistent + list(weights.values())]
        def retile(value, height):
            destination = small if height == 16 else ttnn.empty((1, 1, 8, 5120), device=mesh,
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            return copy_live_rows(mesh, value, destination)
        def forward(tiny, storage):
            if not tiny:
                return retain(storage, mlp.forward(source))
            interleaved = retain(storage, ttnn.to_memory_config(source, ttnn.L1_MEMORY_CONFIG))
            result = execute(ttnn, interleaved, weights, programs, mlp.compute_kernel_config_decode,
                lambda value: retain(storage, value), tiny=True, retile=retile,
                multiply=lambda gate, up: multiply(mesh, gate, up, hidden))
            return retain(storage, tt_all_reduce(result['partial'], mesh, collectives, cluster_axis=0, dim=3,
                topology=args.ccl_topology(), memory_config=ttnn.DRAM_MEMORY_CONFIG))
        references = []
        progress('weights_ready')
        for pattern, host in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(host, source)
            reference = read(forward(False, transient))
            references.append(reference)
            if not exact(read(forward(True, transient)), reference):
                raise AssertionError(f'Small-tile MLP differs from actual native forward at pattern={pattern}')
            report['eager_checks'].extend(dict(pattern=pattern, chip=chip, exact=True) for chip in range(2))
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, transient)
            transient.clear()
        if any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2)):
            raise AssertionError('Changed-input control must distinguish stale outputs on both chips')
        outputs = []
        for tiny in (False, True):
            trace, output = capture_operation(ttnn, mesh, lambda tiny=tiny: forward(tiny, captured))
            traces.append(trace)
            outputs.append(output)
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
        for pattern, host in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(host, source)
            for arm, trace in enumerate(traces):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                if not exact(read(outputs[arm]), references[pattern]):
                    raise AssertionError('Changed-input trace differs from actual native forward')
                report['trace_checks'].extend(dict(pattern=pattern, arm=arm, chip=chip, exact=True) for chip in range(2))
            for block in range(3):
                samples = []
                for arm in (0, 1, 1, 0):
                    samples.append(timed(arm))
                    if not exact(read(outputs[arm]), references[pattern]):
                        raise AssertionError('Timed trace changed output')
                baseline = statistics.mean((samples[0], samples[3]))
                candidate = statistics.mean(samples[1:3])
                report['blocks'].append(dict(pattern=pattern, block=block, samples_ms=samples,
                    control_ms=baseline, candidate_ms=candidate, ratio=baseline / candidate))
            if bindings != [addresses(ttnn, value) for value in persistent + list(weights.values())]:
                raise AssertionError('Persistent input, workspace or weight addresses changed')
            if not exact(read(source), [patterns[pattern]] * 2):
                raise AssertionError('MLP modified caller-owned input')
            progress(f'pattern_{pattern}_passed')
        report['control_ms'] = statistics.mean(block['control_ms'] for block in report['blocks'])
        report['candidate_ms'] = statistics.mean(block['candidate_ms'] for block in report['blocks'])
        report['eligible_for_full_model_gate'] = all(block['ratio'] > 1.02 for block in report['blocks'])
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        progress('failed')
        raise
    finally:
        if mesh is not None:
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            release_owned(ttnn, transient + captured + persistent)
            ttnn.close_mesh_device(mesh)
    report['passed'] = True
    progress('complete')


if __name__ == '__main__':
    main()
