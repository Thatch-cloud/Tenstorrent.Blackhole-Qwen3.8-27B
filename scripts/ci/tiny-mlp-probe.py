"""Simulator-only composed target-precision MLP; synthetic weights, no collective or throughput claim."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from tiny_tile_dma import copy_live_rows
from tiny_tile_matmul import PROJECTIONS
from tiny_mlp import execute
from tiny_tile_product import multiply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg

    report = dict(passed=False, scope=__doc__, rows=8, weights='Independent synthetic local matrices at native TP2 shapes',
        boundaries='32-to-16 input, 16-row gate/up/product/down, 16-to-32 output; both DMA calls inside trace',
        eager_checks=[], trace_checks=[], negative_controls=[],
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
            'tiny_mlp.py', 'tiny-mlp-probe.py', 'tiny_tile_dma.py', 'tiny_tile_dma.cpp', 'tiny_tile_matmul.py', 'attention_batch.py',
            'tiny_tile_product.py', 'tiny_tile_product_compute.cpp', 'tiny_tile_product_io.cpp')},
        packer_zero_graft=os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') == '1', packer_l1_acc=True)
    native = Path(os.environ['TT_METAL_HOME'])
    report['native_sources'] = {name: hashlib.sha256((native/name).read_bytes()).hexdigest() for name in (
        'models/demos/blackhole/qwen36/tt/tp_common.py', 'models/demos/blackhole/qwen36/tt/mlp.py',
        'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h',
        'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp',
        'ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp',
        'tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h')}
    mesh, persistent, transient, captured, traces = None, [], [], [], []
    def retain(storage, value):
        storage.append(value)
        return value
    def progress(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(stage, flush=True)
    def read(value, chip):
        return ttnn.to_torch(ttnn.get_device_tensors(value)[chip]).clone()
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        if (mesh.compute_with_storage_grid_size().x, mesh.compute_with_storage_grid_size().y) != (11, 10):
            raise ValueError('Qualified worker-dispatch grid required')
        weights, programs = {}, {}
        generator = torch.Generator().manual_seed(1659)
        for name, (inner, width, cores, dtype, silu) in PROJECTIONS.items():
            host = (torch.randn((1, 1, inner, width * 2), generator=generator) / math.sqrt(inner)).bfloat16()
            weights[name] = retain(persistent, ttnn.from_torch(host, device=mesh, dtype=getattr(ttnn, dtype),
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3)))
            programs[name] = create_matmul_1d_decode_progcfg(8, inner, width, cores,
                fused_activation=ttnn.UnaryOpType.SILU if silu else None, grid_w=11)
            del host
        patterns = [torch.randn((1, 1, 8, 5120), generator=generator).bfloat16() for unused in range(2)]
        mapper = ttnn.ReplicateTensorToMesh(mesh)
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in patterns]
        source = retain(persistent, ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
        small = retain(persistent, ttnn.from_torch(torch.zeros_like(patterns[0]), device=mesh, dtype=ttnn.bfloat16,
            tile=ttnn.Tile((16, 32)), layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
        restored = retain(persistent, ttnn.empty((1, 1, 8, 5120), device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG))
        hidden = retain(persistent, ttnn.from_torch(torch.zeros((1, 1, 8, 8704), dtype=torch.bfloat16),
            device=mesh, dtype=ttnn.bfloat16, tile=ttnn.Tile((16, 32)), layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
        bindings = [addresses(ttnn, value) for value in persistent]
        compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        def retile(value, height):
            return copy_live_rows(mesh, value, small if height == 16 else restored)
        def forward(tiny, storage):
            return execute(ttnn, source, weights, programs, compute, lambda value: retain(storage, value),
                tiny=tiny, retile=retile if tiny else None,
                multiply=(lambda gate, up: multiply(mesh, gate, up, hidden)) if tiny else None)
        references = []
        progress('weights_ready')
        for pattern, host in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(host, source)
            baseline = forward(False, transient)
            expected = {name: [read(value, chip) for chip in range(2)] for name, value in baseline.items()}
            references.append(expected)
            candidate = forward(True, transient)
            for name, value in candidate.items():
                for chip in range(2):
                    actual = read(value, chip)
                    if not torch.equal(actual.view(torch.int16), expected[name][chip].view(torch.int16)):
                        report['mismatch'] = dict(component=name, pattern=pattern, chip=chip,
                            elements=int((actual.view(torch.int16) != expected[name][chip].view(torch.int16)).sum()),
                            max_absolute=float((actual.float() - expected[name][chip].float()).abs().max()))
                        progress('eager_mismatch')
                        raise AssertionError(f'Composed MLP mismatch: {report["mismatch"]}')
                    report['eager_checks'].append(dict(pattern=pattern, chip=chip, component=name, exact=True))
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, transient)
            transient.clear()
            progress(f'eager_pattern_{pattern}_passed')
        for chip in range(2):
            if torch.equal(references[0]['partial'][chip], references[1]['partial'][chip]):
                raise AssertionError('Changed-input negative control is not discriminating')
            report['negative_controls'].append(dict(chip=chip, stale_input_detected=True))
        outputs = []
        for tiny in (False, True):
            trace, state = capture_operation(ttnn, mesh, lambda tiny=tiny: forward(tiny, captured))
            traces.append(trace)
            outputs.append(state)
        for pattern, host in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(host, source)
            for arm, (trace, state) in enumerate(zip(traces, outputs, strict=True)):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for name, value in state.items():
                    for chip in range(2):
                        if not torch.equal(read(value, chip).view(torch.int16), references[pattern][name][chip].view(torch.int16)):
                            raise AssertionError(f'Composed MLP trace changed {name} at pattern={pattern} arm={arm} chip={chip}')
                        report['trace_checks'].append(dict(pattern=pattern, arm=arm, chip=chip, component=name, exact=True))
            if bindings != [addresses(ttnn, value) for value in persistent]:
                raise AssertionError('Persistent MLP buffers moved under live traces')
            if any(not torch.equal(read(source, chip), patterns[pattern]) for chip in range(2)):
                raise AssertionError('Composed MLP modified caller-owned input')
            progress(f'trace_pattern_{pattern}_passed')
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
