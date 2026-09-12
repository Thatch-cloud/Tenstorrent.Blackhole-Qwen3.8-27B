"""Native B8 history windows versus compact control and independent CPU indexing."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_conv_windows import build_windows as compact
from gdn_native_slot_windows import build_windows as candidate
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh report required')
    import torch
    import ttnn

    directory = Path(__file__).parent
    sources = ('gdn-native-windows-probe.py', 'gdn_native_slot_windows.py', 'gdn_conv_windows.py',
        'gdn_conv_windows.cpp', 'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')
    def hashes():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sources}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', sources=hashes(),
        checks=[], immutable_checks=[], hardware_qualified=False, scope=__doc__)
    mesh, owned, traces = None, [], []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def fixture(seed):
        generator = torch.Generator().manual_seed(385100 + seed)
        projected = torch.randn((2, 16, 8256), generator=generator).to(torch.bfloat16)
        history = [torch.randn((2, 8, 5120), generator=generator).to(torch.bfloat16) for slot in range(4)]
        for slot, value in enumerate(history):
            for chip in range(2):
                for inactive in range(1, 8):
                    value[chip, inactive].fill_(32 * seed + 8 * chip + inactive + slot)
        return [projected, *history, *(value[:, :1].clone() for value in history)]
    def read(tensor):
        return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]
    def verify(outputs, inputs, host, seed, mode):
        expected = torch.cat([*(value[:, :1] for value in host[1:5]), host[0][..., :5120]], dim=1)
        for arm, values in zip(('control', 'native'), outputs, strict=True):
            for slot, tensor in enumerate(values):
                for chip, actual in enumerate(read(tensor)):
                    if not torch.equal(actual, expected[chip:chip + 1, slot:slot + 16]):
                        raise AssertionError(f'Window mismatch {seed=} {mode=} {arm=} {slot=} {chip=}')
                    report['checks'].append(dict(seed=seed, mode=mode, arm=arm, slot=slot, chip=chip, exact=True))
        for operand, (tensor, value) in enumerate(zip(inputs, host, strict=True)):
            for chip, actual in enumerate(read(tensor)):
                if not torch.equal(actual, value[chip:chip + 1]):
                    raise AssertionError('Projected or native/compact history input changed')
                report['immutable_checks'].append(dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True))
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        host = fixture(0)
        inputs = []
        for value in host:
            tensor = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            inputs.append(tensor)
            owned.append(tensor)
        bindings = [addresses(ttnn, value) for value in inputs]
        def run(native):
            values = candidate(mesh, inputs[0], inputs[1:5], experimental=True) if native else compact(mesh, inputs[0], inputs[5:])
            owned.extend(values)
            return values
        save('eager')
        outputs = [run(False), run(True)]
        ttnn.synchronize_device(mesh)
        verify(outputs, inputs, host, 0, 'eager')
        outputs = []
        for native in (False, True):
            trace, values = capture_operation(ttnn, mesh, lambda: run(native))
            traces.append(trace)
            outputs.append(values)
        for seed in (0, 1, 2):
            save('replay_' + str(seed))
            host = fixture(seed)
            for value, destination in zip(host, inputs, strict=True):
                source = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
                ttnn.copy_host_to_device_tensor(source, destination)
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            verify(outputs, inputs, host, seed, 'replay')
            if [addresses(ttnn, value) for value in inputs] != bindings:
                raise AssertionError('Input bindings changed')
        if len(report['checks']) != 64 or len(report['immutable_checks']) != 72:
            raise AssertionError('Complete eager and refreshed replay matrix required')
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
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
