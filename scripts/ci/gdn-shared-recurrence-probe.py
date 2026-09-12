"""Synthetic shared-Q/K preparation plus complete recurrence; no speed claims."""

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
    from gdn_shared_qk_pipeline import build as build_pipeline
    import gdn_vsplit_norm_batch as baseline
    from attention_batch import capture_operation
    from gdn_multitoken import HASHES, KERNEL_ROOT
    from gdn_vsplit_prepared import PreparedVSplit

    root = Path(os.environ['TT_METAL_HOME'])
    directory = Path(__file__).parent
    def hashes():
        paths = list(directory.glob('*.py')) + [root / KERNEL_ROOT / name for name in HASHES]
        paths += [root / 'tt_metal/hw/inc/api/compute' / name for name in (
            'eltwise_binary_sfpu.h', 'eltwise_binary.h', 'tile_move_copy.h', 'bcast.h')]
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', sources=hashes(),
        checks=[], immutable_checks=[], scope=__doc__, hardware_qualified=False, timing_qualified=False,
        rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True)
    mesh, inputs, prepared, traces = None, [], [], []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def fixture(seed):
        generator = torch.Generator().manual_seed(389100 + seed)
        return [(torch.randn((2, 16, 5120), generator=generator) * .1).bfloat16(),
            torch.sigmoid(torch.randn((2, 16, 24), generator=generator)).bfloat16(),
            (-torch.rand((2, 16, 24), generator=generator)).bfloat16(),
            (torch.randn((2, 24, 128, 128), generator=generator) * .01).bfloat16(),
            torch.randn((2, 16, 3072), generator=generator).bfloat16(),
            (1 + torch.randn((2, 1, 128), generator=generator) * .1).bfloat16()]
    def read(tensor):
        shards = ttnn.get_device_tensors(tensor)
        if len(shards) != 2:
            raise AssertionError('Both chips required')
        return [ttnn.to_torch(shard).clone() for shard in shards]
    def check(control, candidate, host, mode):
        mismatches = []
        for operand, (reference, observed) in enumerate(zip(control, candidate, strict=True)):
            for chip, (expected, actual) in enumerate(zip(read(reference), read(observed), strict=True)):
                if not torch.isfinite(expected).all() or not torch.equal(expected, actual):
                    difference = (expected.float() - actual.float()).abs()
                    mask = expected != actual
                    indices = mask.flatten().nonzero().flatten()[:8]
                    diagnostic = dict(mode=mode, operand=operand, chip=chip, shape=list(expected.shape),
                        mismatches=int(mask.sum()), elements=expected.numel(),
                        finite_expected=bool(torch.isfinite(expected).all()),
                        finite_actual=bool(torch.isfinite(actual).all()),
                        max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                        prefix_mismatches=mask.reshape(16, -1).sum(dim=1).tolist(),
                        first_indices=indices.tolist(), expected=expected.flatten()[indices].float().tolist(),
                        actual=actual.flatten()[indices].float().tolist())
                    mismatches.append(diagnostic)
                    report.setdefault('numerical_failures', []).append(diagnostic)
                    continue
                report['checks'].append(dict(mode=mode, operand=operand, chip=chip, exact=True))
        for operand, (tensor, expected) in enumerate(zip(inputs, host, strict=True)):
            for chip, (actual, reference) in enumerate(zip(read(tensor), expected.chunk(2, dim=0), strict=True)):
                if not torch.equal(actual, reference):
                    raise AssertionError(f'Input mutation {mode=} {operand=} {chip=}')
                report['immutable_checks'].append(dict(mode=mode, operand=operand, chip=chip, exact=True))
        if mismatches:
            raise AssertionError(f'{len(mismatches)} output/state/bridge comparisons failed exactness in {mode}')
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576,
            trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        host = fixture(0)
        for value in host:
            inputs.append(ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        def prepare():
            operation = PreparedVSplit(mesh, *inputs[:4], z=inputs[4], norm_w=inputs[5],
                root=root, experimental=True, batch_norm=True, output_memory=ttnn.L1_MEMORY_CONFIG)
            prepared.append(operation)
            return operation
        control = prepare()
        class Candidate:
            def __init__(self):
                self.owned = []
                try:
                    def allocate(shape, dtype, layout, memory):
                        tensor = ttnn.empty(shape, device=mesh, dtype=dtype, layout=layout, memory_config=memory)
                        self.owned.append(tensor)
                        return tensor
                    bridge = allocate((16, 1, 96, 32), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
                    states = allocate((16, 24, 128, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
                    output = allocate((1, 16, 3072), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)
                    query = allocate((1, 16, 1024), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
                    key = allocate((1, 16, 1024), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
                    tensors = [*inputs[:4], bridge, states, *inputs[4:], output, query, key]
                    self.programs = build_pipeline(ttnn, mesh, tensors, root=root)
                    self.result = output, states, bridge
                except BaseException:
                    self.close()
                    raise

            def run(self):
                for tensors, program in self.programs:
                    ttnn.generic_op(tensors, program)
                return self.result

            def close(self):
                ttnn.synchronize_device(mesh)
                while self.owned:
                    ttnn.deallocate(self.owned[-1])
                    self.owned.pop()

        candidate = Candidate()
        prepared.append(candidate)
        save('eager')
        reference, actual = control.run(), candidate.run()
        ttnn.synchronize_device(mesh)
        check(reference, actual, host, 'eager')
        save('capture')
        for operation in prepared:
            trace, unused = capture_operation(ttnn, mesh, operation.run)
            traces.append(trace)
        for seed in (1, 2, 0):
            save('replay_' + str(seed))
            host = fixture(seed)
            for value, destination in zip(host, inputs, strict=True):
                payload = ttnn.from_torch(value, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(payload, destination)
            ttnn.synchronize_device(mesh)
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check(reference, actual, host, 'replay_' + str(seed))
        if len(report['checks']) != 24 or len(report['immutable_checks']) != 48:
            raise AssertionError('Incomplete comparison matrix')
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
                for operation in reversed(prepared):
                    operation.close()
                for tensor in reversed(inputs):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
