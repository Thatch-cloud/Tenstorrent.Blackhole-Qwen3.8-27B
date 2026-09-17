"""Direct causal-window convolution versus native windows, math and every checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from gdn_conv_windows import build_windows
from gdn_direct_window_device import execute, sources, HASHES
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Fresh simulator-only direct-window evidence required')
    import torch
    import ttnn

    directory = Path(__file__).parent
    names = (Path(__file__).name, 'gdn_direct_window.py', 'gdn_direct_window_device.py',
             'gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'attention_batch.py', 'gdn_multitoken_conv.py')
    def hashes():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
    native = sources('/opt/tt-metal')
    report = dict(passed=False, closed_cleanly=False, backend='simulator', sources=hashes(),
        native_sources=HASHES, generated_reader_sha256=hashlib.sha256(native['reader'].encode()).hexdigest(),
        checks=[], immutable_checks=[], performance_qualified=False, scope=__doc__)
    mesh, trace, owned = None, None, []
    def retain(value):
        owned.append(value)
        return value
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def fixture(seed):
        generator = torch.Generator().manual_seed(385700 + seed)
        shapes = [(2, 16, 8240)] + [(2, 1, 5120)] * 8 + [(2, 1, 24)] * 2
        result = [(torch.randn(shape, generator=generator) / 4).bfloat16() for shape in shapes]
        result[-1] = -result[-1].abs() - 0.25
        return result
    def read(value):
        return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inputs = [retain(ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)) for value in fixture(0)]
        bindings = [addresses(ttnn, value) for value in inputs]
        def update(seed):
            host = fixture(seed)
            for value, destination in zip(host, inputs, strict=True):
                source = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(source, destination)
            return host
        def control():
            windows = build_windows(mesh, inputs[0], inputs[1:5])
            owned.extend(windows)
            packed = ttnn.transformer.gdn_decode_conv_gates(inputs[0], windows, inputs[5:9], inputs[0], inputs[0],
                inputs[9], inputs[10], batch=16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                channels=5120, a_col=8192, b_col=8216)
            owned.extend(packed)
            return [*packed, *windows]
        def candidate():
            return execute(ttnn, mesh, inputs[0], inputs[1:5], inputs[5:9], inputs[9], inputs[10], retain)
        references = {}
        for seed in (0, 1, 2):
            save('native_' + str(seed))
            update(seed)
            references[seed] = [read(value) for value in control()]
        def verify(outputs, seed, mode):
            if len(outputs) != 7:
                raise AssertionError('Three numerical outputs and four checkpoint windows required')
            for operand, value in enumerate(outputs):
                for chip, actual in enumerate(read(value)):
                    if not torch.equal(actual.view(torch.int16), references[seed][operand][chip].view(torch.int16)):
                        raise AssertionError(f'Native/direct mismatch {seed=} {mode=} {operand=} {chip=}')
                    report['checks'].append(dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True))
            for operand, (value, host) in enumerate(zip(inputs, fixture(seed), strict=True)):
                for chip, actual in enumerate(read(value)):
                    if not torch.equal(actual.view(torch.int16), host[chip:chip + 1].view(torch.int16)):
                        raise AssertionError('Immutable projected/history/weight input changed')
                    report['immutable_checks'].append(dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True))
        save('eager')
        update(0)
        verify(candidate(), 0, 'eager')
        save('capture')
        trace, outputs = capture_operation(ttnn, mesh, candidate)
        for seed in (0, 1, 2):
            save('replay_' + str(seed))
            update(seed)
            for value in outputs:
                shape = (2, *tuple(value.shape)[1:])
                poison = ttnn.from_torch(torch.full(shape, float('nan'), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(poison, value)
                if any(not torch.isnan(part).all() for part in read(value)):
                    raise AssertionError('Output poison not installed')
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            verify(outputs, seed, 'replay')
            if bindings != [addresses(ttnn, value) for value in inputs]:
                raise AssertionError('Input bindings changed')
        if len(report['checks']) != 56 or len(report['immutable_checks']) != 88:
            raise AssertionError('Complete two-chip eager/replay matrix required')
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['native_unchanged'] = sources('/opt/tt-metal') == native
            report['passed'] = (report['passed'] and report['closed_cleanly'] and report['native_unchanged']
                                and report['sources'] == report['sources_after'])
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
