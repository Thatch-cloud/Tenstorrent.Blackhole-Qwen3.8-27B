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
    options = parser.parse_args()
    import torch
    import ttnn

    taps = (5, 19, 33, 47, 61)
    report = dict(passed=False, scope=__doc__, checks=[], backend='ttsim-fast-dispatch',
        sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in ('feature-trace-probe.py', 'target_features.py', 'full_replay.py')})
    mesh = tensor = output = trace = features = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        expected = []
        for pattern in range(3):
            value = torch.arange(32).reshape(1, 1, 32, 1).expand(1, 1, 32, 5120).clone().bfloat16() + pattern * 3
            value[..., 2560:] += 16
            expected.append(value)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1)
        host_inputs = [ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                       for value in expected]
        tensor = ttnn.from_torch(expected[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
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
            return LayerOutputCapture(model, taps,
                snapshot=lambda value: ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                release=ttnn.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(ttnn, value))))

        features = capture_features()
        warm_feature_fixture(SimpleNamespace(run=operation, close=lambda: None), features, ttnn, mesh)
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
        if len(report['checks']) != 30:
            raise AssertionError('Complete changed-input matrix required')
    finally:
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            if features is not None:
                features.close()
            if output is not None:
                ttnn.deallocate(output)
            if tensor is not None:
                ttnn.deallocate(tensor)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
