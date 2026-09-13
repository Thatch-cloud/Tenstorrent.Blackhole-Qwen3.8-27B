"""Rank-256 native sparse/dense dot comparison; synthetic weights, not runtime qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Simulator only and fresh report required')
    import torch
    import ttnn

    root = Path(os.environ['TT_METAL_HOME']) / 'ttnn/cpp/ttnn/operations/matmul'
    paths = [Path(__file__), root / 'device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp',
        root / 'device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp']
    paths += [root / 'device/kernels' / name for name in (
        'compute/bmm_large_block_zm_fused_bias_activation.cpp',
        'dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp',
        'dataflow/reader_bmm_tile_layout_in0_receiver.cpp',
        'dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp')]

    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    report = dict(passed=False, closed_cleanly=False, scope=__doc__, sources=hashes(), checks=[],
        performance_qualified=False, full_vocabulary_qualified=False, cache_payload_qualified=False)
    mesh, trace, owned = None, None, []

    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage, checks=len(report['checks']))), flush=True)

    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=4194304)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def allocate(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            result = ttnn.from_torch(host, dtype=dtype, layout=layout, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            owned.append(result)
            return result

        def upload(host, tensor):
            payload = ttnn.from_torch(host, dtype=tensor.dtype, layout=tensor.layout, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(payload, tensor)

        def fixture(seed):
            generator = torch.Generator().manual_seed(381927 + seed)
            return ((torch.randn((2, 1, 1, 256), generator=generator) * .2).bfloat16(),
                (torch.randn((2, 1, 256, 64), generator=generator) * .2).bfloat16())

        host_left, host_right = fixture(0)
        left, right = allocate(host_left), allocate(host_right)
        mask = allocate(torch.ones((2, 1, 1, 1)), layout=ttnn.ROW_MAJOR_LAYOUT)
        dense = allocate(torch.zeros((2, 1, 1, 64)), dtype=ttnn.float32)
        sparse = allocate(torch.zeros((2, 1, 1, 64)), dtype=ttnn.float32)
        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(2, 1),
            in0_block_w=1, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        math = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)

        def run():
            ttnn.matmul(left, right, dtype=ttnn.float32, program_config=config,
                compute_kernel_config=math, memory_config=ttnn.DRAM_MEMORY_CONFIG, optional_output_tensor=dense)
            ttnn.sparse_matmul(left, right, sparsity=mask, program_config=config,
                compute_kernel_config=math, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                optional_output_tensor=sparse)

        def read(tensor):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            return [ttnn.to_torch(shard).clone() for shard in shards]

        def check(label, enabled):
            references, candidates = read(dense), read(sparse)
            for chip, (reference, actual) in enumerate(zip(references, candidates, strict=True)):
                expected = reference if enabled[chip] else torch.zeros_like(reference)
                exact = torch.equal(expected.view(torch.int32), actual.view(torch.int32))
                report['checks'].append(dict(label=label, chip=chip, active=bool(enabled[chip]), exact=exact,
                    mismatches=int((expected.view(torch.int32) != actual.view(torch.int32)).sum()),
                    max_abs=float((expected - actual).abs().max())))
                if not torch.isfinite(reference).all() or not torch.any(reference != 0):
                    raise AssertionError('Degenerate dense control')
            for tensor, host in ((left, host_left), (right, host_right)):
                for chip, actual in enumerate(read(tensor)):
                    if not torch.equal(actual, host[chip:chip + 1]):
                        raise AssertionError('Input mutated')

        save('eager')
        run()
        ttnn.synchronize_device(mesh)
        check('eager', [1, 1])
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        for seed, enabled in enumerate(([1, 0], [0, 1], [0, 0], [1, 1]), start=1):
            save('replay_' + str(seed))
            host_left, host_right = fixture(seed)
            upload(host_left, left)
            upload(host_right, right)
            upload(torch.tensor(enabled).reshape(2, 1, 1, 1).bfloat16(), mask)
            upload(torch.full((2, 1, 1, 64), 123.0), sparse)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check('replay_' + str(seed), enabled)
        report['passed'] = len(report['checks']) == 10 and all(check['exact'] for check in report['checks'])
        if not report['passed']:
            raise AssertionError('Sparse dot differs from dense native bits or inactive zero contract')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                for tensor in reversed(owned):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
