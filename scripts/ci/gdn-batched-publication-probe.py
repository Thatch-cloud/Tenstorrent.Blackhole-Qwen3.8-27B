"""Simulator-only native/candidate GDN publication comparison; no timing claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from feature_projection import require_projection_environment
from gdn_commit_dma import prepare as native_prepare
from gdn_commit_batched_dma import prepare as candidate_prepare
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('gdn-batched-publication-probe.py', 'gdn_commit_dma.py', 'gdn_commit_dma.cpp',
    'gdn_commit_batched_dma.py', 'gdn_commit_batched_dma.cpp', 'attention_batch.py',
    'feature_projection.py', 'gdn_multitoken_conv.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, choices=(1, 48), default=1)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh simulator report required')
    import torch
    import ttnn
    root = Path(__file__).parent
    def hashes():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', rows=16,
        layers=options.layers, sources=hashes(), checks=[], padding_audited=False)
    mesh, tensors, traces = None, [], []
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
        shapes = compact + [(16, 24, 128, 128)] + [(1, 16, 5120)] * 4
        shapes += [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4 + compact
        generator = torch.Generator().manual_seed(389113)
        patterns, uploads = [], []
        save('fixtures')
        for pattern in range(2):
            host_layers, staged_layers = [], []
            for layer in range(options.layers):
                values = [torch.randn((shape[0] * 2, *shape[1:]), generator=generator).bfloat16()
                    for shape in shapes]
                for value in values[15:]:
                    value.fill_(float('nan'))
                host_layers.append(values)
                staged_layers.append([ttnn.from_torch(value, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for value in values])
            patterns.append(host_layers)
            uploads.append(staged_layers)
        layers = []
        for values in patterns[0]:
            local = []
            layers.append(local)
            for value in values:
                tensor = ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
                tensors.append(tensor)
                local.append(tensor)
        bindings = [addresses(ttnn, value) for value in tensors]
        def update(pattern):
            for staged, local in zip(uploads[pattern], layers, strict=True):
                for source, destination in zip(staged, local, strict=True):
                    ttnn.copy_host_to_device_tensor(source, destination)
        def check(pattern, prefix, arm, repetition):
            for layer, local in enumerate(layers):
                for chip in range(2):
                    values = [value.chunk(2, dim=0)[chip] for value in patterns[pattern][layer]]
                    selected = values[:5] if prefix == 0 else [values[5][prefix - 1:prefix]] + [
                        value[:, prefix - 1:prefix] for value in values[6:10]]
                    expected = values[:10] + [value.clone() for value in values[10:15]] + selected
                    expected[10][0:1] = selected[0]
                    for slot in range(1, 5):
                        expected[10 + slot][:, 0:1] = selected[slot]
                    for operand, (tensor, reference) in enumerate(zip(local, expected, strict=True)):
                        actual = ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip])
                        if not torch.equal(actual.view(torch.int16), reference.contiguous().view(torch.int16)):
                            raise AssertionError(f'{arm=} {pattern=} {prefix=} {layer=} {chip=} {operand=}')
                    report['checks'].append(dict(arm=arm, pattern=pattern, prefix=prefix,
                        repetition=repetition, layer=layer, chip=chip, exact=True))
            if bindings != [addresses(ttnn, value) for value in tensors]:
                raise AssertionError('Publication bindings changed')
        prefixes = tuple(range(17)) if options.layers == 1 else (0, 1, 8, 16)
        native = {prefix: native_prepare(mesh, layers, prefix) for prefix in prefixes}
        candidate = {prefix: candidate_prepare(mesh, layers, prefix) for prefix in prefixes}
        for prefix in prefixes:
            for pattern in range(2):
                for arm, operation in (('native', native[prefix]), ('candidate', candidate[prefix])):
                    save(f'eager_{arm}_{prefix}_{pattern}')
                    update(pattern)
                    operation()
                    ttnn.synchronize_device(mesh)
                    check(pattern, prefix, arm, None)
        for prefix in prefixes:
            trace, unused = capture_operation(ttnn, mesh, candidate[prefix])
            traces.append(trace)
            for repetition, pattern in enumerate((0, 1, 0)):
                save(f'replay_{prefix}_{repetition}')
                update(pattern)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                check(pattern, prefix, 'replay', repetition)
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, tensors)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            if report['sources_after'] != report['sources'] or not report['closed_cleanly']:
                report['passed'] = False
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
