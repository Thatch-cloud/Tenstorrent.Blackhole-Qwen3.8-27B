"""Weight-free T16 shared-Q/K normalization comparison; no recurrence or speed claim."""

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
        raise ValueError('Simulator-only execution and a fresh report required')
    import torch
    import ttnn
    from attention_batch import capture_operation
    from gdn_multitoken import HASHES, KERNEL_ROOT
    from gdn_shared_qk_program import build

    directory, runtime = Path(__file__).parent, Path(os.environ['TT_METAL_HOME'])
    def hashes():
        paths = list(directory.glob('*.py')) + [runtime / KERNEL_ROOT / name for name in HASHES]
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=hashes(), checks=[], immutable_checks=[], sanity_checks=[],
        hardware_qualified=False, timing_qualified=False, recurrence_qualified=False)
    mesh, owned, traces = None, [], []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def retain(tensor):
        owned.append(tensor)
        return tensor
    def read(tensor):
        shards = ttnn.get_device_tensors(tensor)
        if len(shards) != 2:
            raise AssertionError('Both chips required')
        return [ttnn.to_torch(shard).clone() for shard in shards]
    def fixture(seed):
        generator = torch.Generator().manual_seed(381600 + seed)
        host = (torch.randn((2, 16, 5120), generator=generator) * (.1 if seed != 2 else .0001)).bfloat16()
        host[:, 0, :128] = 0
        host[:, 15, 1024:1152] = 0
        return host
    previous = None
    def check(host, inputs, control, candidate, label):
        nonlocal previous
        snapshots = []
        for operand, (reference, result) in enumerate(zip(control, candidate, strict=True)):
            for chip, (expected, actual) in enumerate(zip(read(reference), read(result), strict=True)):
                if not torch.isfinite(expected).all() or not torch.equal(expected, actual):
                    report.setdefault('failures', []).append(dict(label=label, operand=operand, chip=chip,
                        max_abs=float((expected - actual).abs().max()), mismatches=int((expected != actual).sum())))
                    raise AssertionError('Block normalization differs from native-chain serial reference')
                report['checks'].append(dict(label=label, operand=operand, chip=chip, exact=True))
                values = host[chip:chip + 1, :, operand * 1024:(operand + 1) * 1024].float().reshape(1, 16, 8, 128)
                mathematical = values / torch.sqrt((values * values).sum(dim=-1, keepdim=True) + 1e-6)
                if operand == 0:
                    mathematical *= 128 ** -.5
                if not torch.allclose(expected, mathematical.reshape(1, 16, 1024), atol=2e-5, rtol=1e-3):
                    raise AssertionError('Independent head/scale/epsilon sanity check failed')
                report['sanity_checks'].append(dict(label=label, operand=operand, chip=chip, passed=True))
                snapshots.append(actual)
        if previous is not None and all(torch.equal(before, after) for before, after in zip(previous, snapshots, strict=True)):
            raise AssertionError('Changed input did not change normalized outputs')
        previous = snapshots
        for chip, actual in enumerate(read(inputs)):
            if not torch.equal(actual, host[chip:chip + 1]):
                raise AssertionError('Input mutated')
            report['immutable_checks'].append(dict(label=label, chip=chip, exact=True))
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
            trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        host = fixture(0)
        inputs = retain(ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        outputs = [[retain(ttnn.empty((1, 16, 1024), device=mesh, dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)) for operand in range(2)] for arm in range(2)]
        programs = [build(ttnn, mesh, inputs, *values, root=runtime, serial=serial)
            for values, serial in zip(outputs, (True, False), strict=True)]
        def run(arm):
            ttnn.generic_op([inputs, *outputs[arm]], programs[arm])
            return outputs[arm]
        save('eager')
        run(0)
        run(1)
        ttnn.synchronize_device(mesh)
        check(host, inputs, *outputs, 'eager')
        save('capture')
        for arm in range(2):
            trace, unused = capture_operation(ttnn, mesh, lambda arm=arm: run(arm))
            traces.append(trace)
        for seed in (1, 2, 0):
            save('replay_' + str(seed))
            host = fixture(seed)
            payload = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(payload, inputs)
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check(host, inputs, *outputs, 'replay_' + str(seed))
        if len(report['checks']) != 16 or len(report['immutable_checks']) != 8 or len(report['sanity_checks']) != 16:
            raise AssertionError('Incomplete normalization matrix')
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
