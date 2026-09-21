#!/usr/bin/env python3
"""Four per-user GDN launches against ONE batched launch, on device, bit for bit.

The whole point of `gdn_user_batch` is that the batched program is the per-user program
four times over, moved onto four disjoint 24-core shares of the grid. This proves it on
hardware: the same random inputs go through both arms and every output byte, every prefix
state byte and every untouched initial state must match exactly. It also times both arms
so the launch saving is a measurement rather than an estimate.

Nothing but the recurrence and the fused norm/gate is exercised - no window DMA, no
conv+gates op, no projection - because those are exactly the parts the change does not
touch, and leaving them out means a failure here can only be the batching.

One card is enough: `--chips 1` opens a 1x1 mesh, which `gdn_user_batch` accepts for this
test alone. `--chips 2` additionally runs the real single-user `gdn_multitoken.execute`
as a third arm, which is the launch the serving stack makes today.

  python3 gdn_user_batch_device_test.py --out results.json --chips 1
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import gdn_multitoken as native
import gdn_user_batch as batch


def same_bits(torch, left, right):
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}.get(left.element_size())
    if width is None:
        return torch.equal(left, right)
    return torch.equal(left.view(width), right.view(width))


def difference(torch, left, right):
    if tuple(left.shape) != tuple(right.shape):
        return dict(shape=[list(left.shape), list(right.shape)])
    gap = (left.float() - right.float()).abs()
    return dict(differing=int((gap > 0).sum()), of=int(gap.numel()), max_abs=float(gap.max()))


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chips', type=int, default=1, choices=(1, 2))
    parser.add_argument('--users', type=int, default=4)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
    return parser.parse_args()


def main():
    arguments = parse()
    report = dict(scope='Uncertified equality and timing of the batched GDN recurrence/norm launch '
                        'against the same kernels launched once per user; no model, no projection, no collective',
                  users=arguments.users, rows=arguments.rows, chips=arguments.chips,
                  iterations=arguments.iterations, seed=arguments.seed,
                  native_sha256=native.HASHES, handoff_sha256=native.HANDOFF_HASHES,
                  module_sha256={name: hashlib.sha256((Path(__file__).with_name(name)).read_bytes()).hexdigest()
                                 for name in ('gdn_user_batch.py', 'gdn_user_batch_conv.py')},
                  result='did-not-run', stages=[])

    def stage(name, **details):
        report['stages'].append(dict(stage=name, **details))
        arguments.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['stages'][-1]), flush=True)

    mesh = None
    try:
        stage('import-runtime')
        import torch
        import ttnn
        report['ttnn_path'] = ttnn.__file__

        stage('load-kernels', root=str(arguments.root))
        kernels = batch.load_kernels(arguments.root)
        report['generated_sha256'] = {role: hashlib.sha256(source.encode()).hexdigest()
                                      for role, source in kernels.items()}

        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, arguments.chips), l1_small_size=24576)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        report['core_shares'] = batch.core_shares(grid.x, grid.y, arguments.users)

        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def host(value):
            shards = ttnn.get_device_tensors(value)
            if len(shards) != arguments.chips:
                raise AssertionError('Expected %d chips, got %d' % (arguments.chips, len(shards)))
            return [ttnn.to_torch(shard).clone() for shard in shards]

        stage('fixture-upload')
        torch.manual_seed(arguments.seed)
        rows = arguments.rows
        # One shared normalization weight, as the layer has; everything else is per user.
        norm_w = upload((1 + torch.randn(1, 1, 128) * 0.1).bfloat16())
        groups, initial_host = [], []
        for index in range(arguments.users):
            initial = (torch.randn(1, 24, 128, 128) * 0.05).bfloat16()
            initial_host.append(initial)
            groups.append((upload(torch.randn(1, rows, 5120).bfloat16()),
                           upload(torch.rand(1, rows, 24).bfloat16()),
                           upload((-torch.rand(1, rows, 24)).bfloat16()),
                           upload(initial),
                           upload(torch.randn(1, rows, 3072).bfloat16()),
                           norm_w))

        def release(produced):
            for output, states in produced:
                ttnn.deallocate(output)
                ttnn.deallocate(states)

        def per_user():
            return [batch.execute(mesh, [group], kernels)[0] for group in groups]

        def batched():
            return batch.execute(mesh, groups, kernels)

        stage('per-user-arm', launches=arguments.users)
        control = per_user()
        ttnn.synchronize_device(mesh)
        control_host = [(host(output), host(states)) for output, states in control]
        release(control)

        stage('batched-arm', launches=1)
        candidate = batched()
        ttnn.synchronize_device(mesh)
        candidate_host = [(host(output), host(states)) for output, states in candidate]
        release(candidate)

        stage('compare')
        mismatches = []
        for index, (expected, actual) in enumerate(zip(control_host, candidate_host)):
            for name, left, right in (('output', expected[0], actual[0]), ('states', expected[1], actual[1])):
                for chip, (one, two) in enumerate(zip(left, right)):
                    if not same_bits(torch, one, two):
                        mismatches.append(dict(user=index, tensor=name, chip=chip,
                                               **difference(torch, one, two)))
        report['bit_exact'] = not mismatches
        report['mismatches'] = mismatches

        stage('initial-state-immutable')
        drift = []
        for index, group in enumerate(groups):
            for chip, shard in enumerate(host(group[3])):
                if not same_bits(torch, shard, initial_host[index]):
                    drift.append(dict(user=index, chip=chip, **difference(torch, shard, initial_host[index])))
        report['initial_state_immutable'] = not drift
        report['initial_state_drift'] = drift

        if arguments.chips == 2:
            stage('native-single-user-arm')
            native_mismatches = []
            for index, group in enumerate(groups):
                output, states = native.execute(mesh, group[0], group[1], group[2], group[3], kernels,
                                                z=group[4], norm_w=group[5])
                ttnn.synchronize_device(mesh)
                for name, left, right in (('output', host(output), candidate_host[index][0]),
                                          ('states', host(states), candidate_host[index][1])):
                    for chip, (one, two) in enumerate(zip(left, right)):
                        if not same_bits(torch, one, two):
                            native_mismatches.append(dict(user=index, tensor=name, chip=chip,
                                                          **difference(torch, one, two)))
                ttnn.deallocate(output)
                ttnn.deallocate(states)
            report['matches_native_single_user'] = not native_mismatches
            report['native_mismatches'] = native_mismatches

        stage('timing', iterations=arguments.iterations)
        timings = {}
        for name, arm in (('per_user', per_user), ('batched', batched)):
            for unused in range(2):
                release(arm())
            ttnn.synchronize_device(mesh)
            samples = []
            for unused in range(arguments.iterations):
                start = time.perf_counter()
                produced = arm()
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - start) * 1000)
                release(produced)
            samples.sort()
            timings[name] = dict(median_ms=samples[len(samples) // 2], best_ms=samples[0],
                                 worst_ms=samples[-1], samples=len(samples))
        timings['speedup'] = timings['per_user']['median_ms'] / timings['batched']['median_ms']
        timings['saving_ms_per_layer'] = timings['per_user']['median_ms'] - timings['batched']['median_ms']
        timings['saving_ms_per_round_48_layers'] = 48 * timings['saving_ms_per_layer']
        report['timings'] = timings

        passed = report['bit_exact'] and report['initial_state_immutable'] \
            and report.get('matches_native_single_user', True)
        report['result'] = 'pass' if passed else 'fail'
        stage('done', result=report['result'])
        print('RESULT', report['result'].upper(), flush=True)
        print(json.dumps(timings, indent=2), flush=True)
        return 0 if passed else 1
    except BaseException as error:
        report['result'] = 'error'
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        arguments.out.write_text(json.dumps(report, indent=2))
        print(report['traceback'], flush=True)
        return 2
    finally:
        arguments.out.write_text(json.dumps(report, indent=2))
        if mesh is not None:
            import ttnn
            ttnn.close_mesh_device(mesh)


if __name__ == '__main__':
    raise SystemExit(main())
