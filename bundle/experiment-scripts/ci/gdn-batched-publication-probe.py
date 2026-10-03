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
    'feature_projection.py', 'gdn_multitoken_conv.py', 'gdn_publication_fixture.py',
    'tensor_bit_compare.py', 'tensor_bit_compare.cpp', 'tensor_bit_compare_gate.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, choices=(1, 48), default=1)
    parser.add_argument('--prefix', type=int, choices=range(17),
        help='Single-prefix diagnostic only; never satisfies the complete simulator gate')
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
        layers=options.layers, sources=hashes(), checks=[], padding_checks=[], poison_checks=[], padding_audited=False,
        coverage='full' if options.prefix is None else 'single-prefix-diagnostic',
        diagnostic_prefix=options.prefix, hardware_qualified=False, serving_qualified=False)
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
        from gdn_publication_fixture import host_layer, expected
        from tensor_bit_compare import prepare as compare_prepare
        from tensor_bit_compare_gate import qualify as qualify_compare
        report['comparator'] = qualify_compare(root)
        save('fixtures')
        def upload(value, *, device=False, padding=float('nan')):
            return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                pad_value=padding, mesh_mapper=mapper,
                **(dict(device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG) if device else {}))
        layers = []
        for layer in range(options.layers):
            local = [upload(value, device=True) for value in host_layer(0, layer)]
            layers.append(local)
            tensors.extend(local)
        references = [upload(value, device=True) for value in host_layer(0, 0)]
        tensors.extend(references)
        counter = ttnn.from_torch(torch.zeros((1, 1, 32), dtype=torch.uint32), device=mesh,
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        tensors.append(counter)
        poison = ttnn.from_torch(torch.full((1, 1, 32), 0xffffffff, dtype=torch.uint32),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        comparisons = [[compare_prepare(ttnn, mesh, tensor, reference, counter)
            for tensor, reference in zip(local, references, strict=True)] for local in layers]
        bindings = [addresses(ttnn, value) for value in tensors]
        def compare(layer, operand, value, padding):
            staged = upload(value, padding=padding)
            ttnn.copy_host_to_device_tensor(staged, references[operand])
            ttnn.copy_host_to_device_tensor(poison, counter)
            comparisons[layer][operand]()
            if not report['checks'] and layer == 0:
                print(json.dumps(dict(comparator_stage='enqueued', operand=operand)), flush=True)
            ttnn.synchronize_device(mesh)
            if not report['checks'] and layer == 0:
                print(json.dumps(dict(comparator_stage='fenced', operand=operand)), flush=True)
            for chip, shard in enumerate(ttnn.get_device_tensors(counter)):
                if torch.count_nonzero(ttnn.to_torch(shard).to(torch.int64)):
                    raise AssertionError(f'Physical bit comparison failed: {layer=} {operand=} {chip=}')
        def update(pattern):
            for layer, local in enumerate(layers):
                values = host_layer(pattern, layer)
                staged = [upload(value) for value in values]
                for source, destination in zip(staged, local, strict=True):
                    ttnn.copy_host_to_device_tensor(source, destination)
                for operand in range(15, 20):
                    compare(layer, operand, values[operand], float('nan'))
                for chip in range(2):
                    report['poison_checks'].append(dict(pattern=pattern, layer=layer, chip=chip, exact=True))
        def check(pattern, prefix, arm, repetition):
            for layer, local in enumerate(layers):
                values = expected(host_layer(pattern, layer), prefix)
                for operand, value in enumerate(values):
                    compare(layer, operand, value, 0 if operand >= 15 else float('nan'))
                for chip in range(2):
                    entry = dict(arm=arm, pattern=pattern, prefix=prefix,
                        repetition=repetition, layer=layer, chip=chip, exact=True)
                    report['checks'].append(entry)
                    report['padding_checks'].extend(dict(entry, operand=operand)
                        for operand in range(20) if operand % 5)
            if bindings != [addresses(ttnn, value) for value in tensors]:
                raise AssertionError('Publication/comparator bindings changed')
        prefixes = tuple(range(17)) if options.layers == 1 else (0, 1, 8, 16)
        if options.prefix is not None:
            prefixes = (options.prefix,)
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
        report['padding_audited'] = True
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
