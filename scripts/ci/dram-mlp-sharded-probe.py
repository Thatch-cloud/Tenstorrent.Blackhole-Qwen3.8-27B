"""Simulator T16 shared-staging sharded-product complete local MLP; synthetic weights, no collective or speed claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dram_mlp_sharded import execute
from dram_sharded_projection import configurations
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned
from tensix_projection_raw import copy_raw, raw_shape
from tiny_tile_matmul import PROJECTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists() or os.environ.get('QWEN_SIM_PACKER_ZERO_GRAFT') != '1':
        raise ValueError('Fresh simulator report and owned packer compatibility required')
    import torch
    import ttnn
    from models.demos.blackhole.qwen36.tt.tp_common import create_matmul_1d_decode_progcfg

    root = Path(__file__).parent
    native = Path(os.environ['TT_METAL_HOME'])
    files = ('dram-mlp-sharded-probe.py', 'dram_mlp_sharded.py', 'dram_sharded_projection.py',
        'dram_projection_reload.py', 'attention_batch.py', 'gdn_multitoken_conv.py',
        'tensix_projection_raw.py', 'tensix_projection_raw.cpp', 'tiny_tile_matmul.py')
    native_files = ('models/demos/blackhole/qwen36/tt/tp_common.py',
        'ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp',
        'ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp',
        'ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp',
        'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h')
    def hashes(directory, names):
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=hashes(root, files), native_sources=hashes(native, native_files),
        rows=16, compared_rows=32, streams=1, full_local_mlp=True, collective_qualified=False,
        timing_qualified=False, eager_checks=[], replay_checks=[])
    mesh, trace = None, None
    owned, temporary = [], []
    def keep(value):
        owned.append(value)
        return value
    def transient(value):
        temporary.append(value)
        return value
    def clear():
        ttnn.synchronize_device(mesh)
        release_owned(ttnn, temporary)
        temporary.clear()
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
            trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        generator = torch.Generator().manual_seed(383951)
        weights, sharded, configs, programs = {}, {}, {}, {}
        for name, (inner, width, cores, dtype, activated) in PROJECTIONS.items():
            save('weight_' + name)
            host = (torch.randn((2, 1, inner, width), generator=generator) / inner ** .5).bfloat16()
            weights[name] = keep(ttnn.from_torch(host, device=mesh, dtype=getattr(ttnn, dtype),
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
            configs[name] = configurations(ttnn, mesh, name)
            sharded[name] = keep(ttnn.to_memory_config(weights[name], configs[name]['weights']))
            programs[name] = create_matmul_1d_decode_progcfg(16, inner, width, cores,
                fused_activation=ttnn.UnaryOpType.SILU if activated else None, fp32_acc=True, grid_w=11)
            del host
        patterns = [(torch.randn((2, 1, 16, 5120), generator=generator) * .125).bfloat16()
            for unused in range(2)]
        payloads = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper) for value in patterns]
        source = keep(ttnn.from_torch(patterns[0], device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=mapper))
        raw = keep(ttnn.from_torch(torch.zeros(raw_shape((1, 1, 32, 5120), 2048), dtype=torch.int32),
            device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)))
        compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True)
        def read(value):
            copy_raw(ttnn, value, raw, 2048)
            ttnn.synchronize_device(mesh)
            return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(raw)]
        def control():
            gate = transient(ttnn.linear(source, weights['gate'], program_config=programs['gate'],
                compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG))
            up = transient(ttnn.linear(source, weights['up'], program_config=programs['up'],
                compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG))
            hidden = transient(ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG))
            return transient(ttnn.linear(hidden, weights['down'], program_config=programs['down'],
                compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG))
        references, input_words = [], []
        for pattern, payload in enumerate(payloads):
            save('eager_' + str(pattern))
            ttnn.copy_host_to_device_tensor(payload, source)
            input_words.append(read(source))
            references.append(read(control()))
            clear()
            actual = read(execute(ttnn, source, sharded, configs, compute, transient))
            unchanged = read(source)
            for chip in range(2):
                if not torch.equal(actual[chip], references[pattern][chip]) or not torch.equal(
                        unchanged[chip], input_words[pattern][chip]):
                    raise AssertionError('Complete MLP output or input differs from native physical words')
                report['eager_checks'].append(dict(pattern=pattern, chip=chip, exact=True, input_unchanged=True))
            clear()
        if any(torch.equal(references[0][chip], references[1][chip]) for chip in range(2)):
            raise AssertionError('Changed-input control is not discriminating')
        save('capture')
        trace, output = capture_operation(ttnn, mesh, lambda: execute(ttnn, source, sharded, configs, compute, keep))
        bindings = [addresses(ttnn, value) for value in (source, output)]
        poison = ttnn.from_torch(torch.full((1, 1, 16, 5120), float('nan'), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        for repetition, pattern in enumerate((0, 1, 0)):
            save('replay_' + str(repetition))
            ttnn.copy_host_to_device_tensor(payloads[pattern], source)
            ttnn.copy_host_to_device_tensor(poison, output)
            if not all(torch.isnan(ttnn.to_torch(shard)).all() for shard in ttnn.get_device_tensors(output)):
                raise AssertionError('Output poison missing')
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual, unchanged = read(output), read(source)
            if bindings != [addresses(ttnn, value) for value in (source, output)]:
                raise AssertionError('Trace bindings moved')
            for chip in range(2):
                if not torch.equal(actual[chip], references[pattern][chip]) or not torch.equal(
                        unchanged[chip], input_words[pattern][chip]):
                    raise AssertionError('Complete MLP replay differs from native physical words')
                report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, chip=chip,
                    exact=True, input_unchanged=True, output_poison_replaced=True))
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                clear()
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        except BaseException as error:
            report['passed'] = False
            report['cleanup_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            report['sources_after'] = hashes(root, files)
            report['native_sources_after'] = hashes(native, native_files)
            if (not report['closed_cleanly'] or report['sources_after'] != report['sources']
                    or report['native_sources_after'] != report['native_sources']):
                report['passed'] = False
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
