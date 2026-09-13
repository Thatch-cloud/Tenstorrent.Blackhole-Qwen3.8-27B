"""Synthetic device-decided cache plus sparse dot; no full-model or performance qualification."""

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Simulator only and fresh report required')
    import torch
    import ttnn
    from markov_cache_program import build as controller
    from markov_cache_pipeline import build as plumbing
    from markov_sparse_build import validate

    build = validate(os.environ['TT_METAL_HOME'], '/experiment/results/markov-sparse-build.json')
    names = ('markov-cache-pipeline-probe.py', 'markov_cache_program.py', 'markov_cache_control.cpp',
        'markov_cache_control.hpp', 'markov_cache_pipeline.py', 'markov_cache_mask.cpp', 'markov_cache_payload.cpp')

    def hashes():
        return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    report = dict(passed=False, closed_cleanly=False, scope=__doc__, sources=hashes(), build=build,
        checks=[], stale_checks=[], full_vocabulary_qualified=False, performance_qualified=False)
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

        def allocate(shape, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, fill=0):
            host = torch.full((2,) + shape, fill, dtype=torch.float32 if dtype != ttnn.uint32 else torch.int64)
            result = ttnn.from_torch(host, dtype=dtype, layout=layout, device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            owned.append(result)
            return result

        def upload(host, tensor):
            payload = ttnn.from_torch(host, dtype=tensor.dtype, layout=tensor.layout, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(payload, tensor)

        def read(tensor):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            return [ttnn.to_torch(shard).clone() for shard in shards]

        state, request, commit_request, decision, status = [allocate(shape) for shape in
            ((1, 65, 8), (1, 1, 8), (1, 1, 8), (1, 1, 8), (1, 1, 8))]
        mask = allocate((1, 1, 1), ttnn.bfloat16)
        cache = allocate((1, 64, 64), ttnn.float32, fill=float('nan'))
        left = allocate((1, 1, 256), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        right = allocate((1, 256, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        bias, output, dense = [allocate((1, 1, 64), ttnn.float32, ttnn.TILE_LAYOUT) for unused in range(3)]
        lookup_program = controller(mesh, state, request, decision, status)
        commit_program = controller(mesh, state, commit_request, decision, status)
        mask_program, payload_program = plumbing(mesh, state, decision, bias, cache, output, mask)
        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(2, 1),
            in0_block_w=1, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        math = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
        upload(torch.tensor([1] + [0] * 7).reshape(1, 1, 1, 8).repeat(2, 1, 1, 1), commit_request)

        def run():
            ttnn.generic_op([state, request, decision, status], lookup_program)
            ttnn.generic_op([decision, mask], mask_program)
            ttnn.sparse_matmul(left, right, sparsity=mask, program_config=config,
                is_input_a_sparse=True, is_input_b_sparse=True, compute_kernel_config=math,
                dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG, optional_output_tensor=bias)
            ttnn.generic_op([state, decision, bias, cache, output], payload_program)
            ttnn.generic_op([state, commit_request, decision, status], commit_program)
            ttnn.matmul(left, right, program_config=config, compute_kernel_config=math,
                dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG, optional_output_tensor=dense)

        def inputs(token, epoch):
            latent = torch.cat([torch.randn((1, 1, 1, 256), generator=torch.Generator().manual_seed(381900 + token + chip))
                for chip in range(2)]).mul(.2).bfloat16()
            weights = torch.randn((2, 1, 256, 64), generator=torch.Generator().manual_seed(382000 + epoch)).mul(.2).bfloat16()
            upload(latent, left)
            upload(weights, right)
            upload(torch.tensor([[0, token + chip, epoch] + [0] * 5 for chip in range(2)]).reshape(2, 1, 1, 8), request)
            ttnn.synchronize_device(mesh)

        inputs(999, 1)
        run()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        oracle = OrderedDict()
        sequence = [(token, 2) for token in range(64)] + [(0, 2), (100, 2), (1, 2), (0, 2), (0, 3), (0, 3)]
        previous_epoch = None
        for index, (token, epoch) in enumerate(sequence):
            if epoch != previous_epoch:
                oracle.clear()
                previous_epoch = epoch
            hit = token in oracle
            if hit:
                oracle.move_to_end(token)
            else:
                if len(oracle) == 64:
                    oracle.popitem(last=False)
                oracle[token] = None
            inputs(token, epoch)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            results, references, raw_bias = read(output), read(dense), read(bias)
            decisions, masks, statuses = read(decision), read(mask), read(status)
            for chip in range(2):
                assert torch.equal(results[chip].view(torch.int32), references[chip].view(torch.int32)), (index, chip, 'bias')
                assert int(decisions[chip].reshape(-1)[1]) == int(hit), (index, chip, 'hit')
                assert float(masks[chip].reshape(-1)[0]) == float(not hit), (index, chip, 'mask')
                assert int(statuses[chip].reshape(-1)[0]) == int(not hit), (index, chip, 'commit')
                if hit:
                    assert torch.count_nonzero(raw_bias[chip]) == 0 and torch.count_nonzero(results[chip]) > 0
                report['checks'].append(dict(index=index, chip=chip, token=token + chip, epoch=epoch, hit=hit, exact=True))
            if index % 16 == 0:
                save('replay_' + str(index))
        snapshots = read(cache)
        bad = torch.cat(read(decision), dim=0).to(torch.int64)
        bad[..., 5] -= 1
        upload(bad, decision)
        ttnn.generic_op([state, decision, bias, cache, output], payload_program)
        ttnn.synchronize_device(mesh)
        for chip, actual in enumerate(read(output)):
            assert torch.isnan(actual).all(), (chip, 'stale output not poisoned')
            assert torch.equal(read(cache)[chip].view(torch.int32), snapshots[chip].view(torch.int32)), (chip, 'stale cache write')
            report['stale_checks'].append(dict(chip=chip, rejected=True, cache_unchanged=True))
        report['passed'] = len(report['checks']) == 140 and len(report['stale_checks']) == 2
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
