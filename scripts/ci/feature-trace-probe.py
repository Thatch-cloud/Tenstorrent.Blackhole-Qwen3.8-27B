"""Simulator-only changed-input feature-copy trace; synthetic layers, no model claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from attention_batch import capture_operation
from gdn_multitoken_conv import addresses
from full_replay import warm_feature_fixture
from target_features import LayerOutputCapture


def main():
    if not os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
        raise RuntimeError('Fast-dispatch simulator required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix-copy', action='store_true')
    parser.add_argument('--rows', type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--two-publications', action='store_true')
    parser.add_argument('--prior-trace', action='store_true')
    parser.add_argument('--late-prefix-pool', action='store_true')
    options = parser.parse_args()
    if options.two_publications and not options.prefix_copy:
        parser.error('--two-publications requires --prefix-copy')
    if options.prior_trace and not (options.prefix_copy and options.two_publications):
        parser.error('--prior-trace requires --prefix-copy --two-publications')
    if options.late_prefix_pool and not options.prior_trace:
        parser.error('--late-prefix-pool requires --prior-trace')
    import torch
    import ttnn

    taps = (5, 19, 33, 47, 61)
    report = dict(passed=False, scope=__doc__, checks=[], backend='ttsim-fast-dispatch', prepared=options.prepared,
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('feature-trace-probe.py', 'target_features.py', 'prepared_target_features.py', 'full_replay.py')})
    mesh = tensor = output = trace = features = None
    published = []
    second_published = []
    prefix_pool = {}
    prepared_destinations = []
    prior_trace = prior_output = None
    prefixes = tuple(dict.fromkeys((0, 1, min(17, options.rows), options.rows)))
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()

        def create_pool():
            from feature_prefix import allocate_prefix_pool
            return allocate_prefix_pool(ttnn, lambda prefix: ttnn.from_torch(
                torch.zeros((1, 1, prefix, 5120), dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)), prefixes=prefixes[1:])

        if options.prior_trace and not options.late_prefix_pool:
            prefix_pool = create_pool()
        expected = []
        for pattern in range(3):
            value = torch.arange(options.rows).reshape(1, 1, options.rows, 1).expand(1, 1, options.rows, 5120).clone().bfloat16() + pattern * 3
            value[..., 2560:] += 16
            expected.append(value)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1)
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                       for value in expected]
        tensor = ttnn.from_torch(expected[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
        if options.prepared:
            for tap in taps:
                prepared_destinations.append(ttnn.from_torch(torch.zeros_like(expected[0]), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
        model = SimpleNamespace(layers=[SimpleNamespace(forward=lambda value: ttnn.add(value, 1.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)) for index in range(64)])

        def operation():
            value = tensor
            for layer in model.layers:
                previous = value
                value = layer.forward(value)
                if previous is not tensor:
                    ttnn.deallocate(previous)
            return value

        def capture_features():
            if options.prepared:
                from prepared_target_features import PreparedTargetFeatures

                return PreparedTargetFeatures(model, taps, prepared_destinations, copy=ttnn.copy,
                    storage_ids=lambda value: tuple(enumerate(addresses(ttnn, value))))
            return LayerOutputCapture(model, taps,
                snapshot=lambda value: ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                release=ttnn.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(ttnn, value))))

        if options.prior_trace:
            warm_output = operation()
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warm_output)
            prior_trace, prior_output = capture_operation(ttnn, mesh, operation)
            if options.late_prefix_pool:
                prefix_pool = create_pool()

        features = capture_features()
        from feature_prefix import allocate_prefixes
        buffers = warm_feature_fixture(SimpleNamespace(run=operation, close=lambda: None), features, ttnn, mesh,
            prepare_features=(lambda values: allocate_prefixes(ttnn, values,
                prefixes * (2 if options.two_publications else 1))) if options.prefix_copy and not prefix_pool else None)
        if prefix_pool:
            buffers = tuple(prefix_pool[epoch, prefix] for epoch in range(2) for prefix in prefixes)
        if buffers is not None:
            published = list(zip(prefixes, buffers[:len(prefixes)], strict=True))
            if options.two_publications:
                second_published = list(zip(prefixes, buffers[len(prefixes):], strict=True))
        features = None
        features = capture_features()
        with features.capture():
            trace, output = capture_operation(ttnn, mesh, operation)
        original = [addresses(ttnn, value) for value in (tensor, *features.outputs())]
        for pattern, source in enumerate(host_inputs):
            ttnn.copy_host_to_device_tensor(source, tensor)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if [addresses(ttnn, value) for value in (tensor, *features.outputs())] != original:
                raise AssertionError('Input or feature allocations changed')
            for layer, feature in zip(taps, features.outputs(), strict=True):
                parts = ttnn.get_device_tensors(feature)
                if len(parts) != 2:
                    raise AssertionError('Both feature shards required')
                for chip, part in enumerate(parts):
                    gold = expected[pattern][..., chip * 2560:(chip + 1) * 2560] + layer + 1
                    if not torch.equal(ttnn.to_torch(part), gold):
                        raise AssertionError('Captured feature copy remained stale')
                    report['checks'].append(dict(pattern=pattern, layer=layer, chip=chip, exact=True))
            if options.prefix_copy:
                from feature_prefix import publish_prefix
                if pattern == 0:
                    for prefix, copies in published:
                        publish_prefix(ttnn, features.outputs(), copies, prefix)
                        if len(copies) != (len(taps) if prefix else 0):
                            raise AssertionError('Incorrect committed feature tap count')
                    ttnn.synchronize_device(mesh)
                else:
                    if options.two_publications and pattern == 1:
                        for prefix, copies in second_published:
                            publish_prefix(ttnn, features.outputs(), copies, prefix)
                        ttnn.synchronize_device(mesh)
                    if options.prior_trace and pattern == 2:
                        ttnn.release_trace(mesh, trace)
                        trace = None
                        features.close()
                        features = None
                        ttnn.execute_trace(mesh, prior_trace, cq_id=0, blocking=True)
                    for prefix, copies in published:
                        for layer, value in zip(taps if prefix else (), copies, strict=True):
                            for chip, part in enumerate(ttnn.get_device_tensors(value)):
                                gold = expected[0][..., :prefix, chip * 2560:(chip + 1) * 2560] + layer + 1
                                actual = ttnn.to_torch(part)
                                if list(actual.shape) != [1, 1, prefix, 2560] or not torch.equal(actual, gold):
                                    raise AssertionError('Published prefix changed after source trace replay')
                                report.setdefault('prefix_checks', []).append(dict(pattern=pattern, prefix=prefix,
                                    layer=layer, chip=chip, exact=True))
                    if options.two_publications and pattern == 2:
                        for prefix, copies in second_published:
                            for layer, value in zip(taps if prefix else (), copies, strict=True):
                                for chip, part in enumerate(ttnn.get_device_tensors(value)):
                                    gold = expected[1][..., :prefix, chip * 2560:(chip + 1) * 2560] + layer + 1
                                    if not torch.equal(ttnn.to_torch(part), gold):
                                        raise AssertionError('Second published prefix changed after source trace replay')
                                    report.setdefault('second_prefix_checks', []).append(dict(pattern=pattern,
                                        prefix=prefix, layer=layer, chip=chip, exact=True))
        if len(report['checks']) != 30:
            raise AssertionError('Complete changed-input matrix required')
        if options.prefix_copy:
            if len(report.get('prefix_checks', [])) != (len(prefixes) - 1) * 20:
                raise AssertionError('Complete retained-prefix matrix required')
            if options.two_publications and len(report.get('second_prefix_checks', [])) != (len(prefixes) - 1) * 10:
                raise AssertionError('Complete second retained-prefix matrix required')
            report['rows'] = options.rows
            report['two_publications'] = options.two_publications
            report['prior_trace'] = options.prior_trace
            report['late_prefix_pool'] = options.late_prefix_pool
            report['sources']['feature_prefix.py'] = hashlib.sha256(Path(__file__).with_name('feature_prefix.py').read_bytes()).hexdigest()
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            if prior_trace is not None:
                ttnn.release_trace(mesh, prior_trace)
            groups = prefix_pool.values() if prefix_pool else [copies for prefix, copies in published + second_published]
            for copies in groups:
                for value in copies:
                    ttnn.deallocate(value)
            if features is not None:
                features.close()
            for value in prepared_destinations:
                ttnn.deallocate(value)
            if output is not None:
                ttnn.deallocate(output)
            if prior_output is not None:
                ttnn.deallocate(prior_output)
            if tensor is not None:
                ttnn.deallocate(tensor)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
