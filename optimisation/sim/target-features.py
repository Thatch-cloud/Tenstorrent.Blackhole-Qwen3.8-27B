"""Simulator-only TP-shard feature ownership; synthetic layers, not a Qwen or drafter parity claim."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from gdn_multitoken_conv import addresses, release_owned
from target_features import LayerOutputCapture


def main():
    spec = importlib.util.spec_from_file_location('guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.require_simulator(os.environ)
    import torch
    import ttnn

    path = Path(os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, backend='ttsim', checks=[], tap_ids=[5, 19, 33, 47, 61], scope=__doc__)
    report['sources'] = {str(source.name): hashlib.sha256(source.read_bytes()).hexdigest() for source in (
        Path(__file__), Path(__file__).resolve().parents[2] / 'scripts/ci/target_features.py')}
    mesh, capture = None, None
    owned = []
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        initial = torch.arange(32).reshape(1, 1, 32, 1).expand(1, 1, 32, 5120).clone().bfloat16()
        initial[..., 2560:] += 16
        hidden = ttnn.from_torch(initial, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
        owned.append(hidden)

        def forward(value, *, mode):
            if mode != 'decode':
                raise ValueError('Synthetic decode fixture only')
            result = ttnn.add(value, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owned.append(result)
            return result

        model = SimpleNamespace(layers=[SimpleNamespace(forward=forward) for index in range(64)])
        capture = LayerOutputCapture(model, report['tap_ids'],
            snapshot=lambda value: ttnn.clone(value, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            release=ttnn.deallocate, storage_ids=lambda value: tuple(enumerate(addresses(ttnn, value))))
        with capture.capture():
            for index, layer in enumerate(model.layers):
                report['last_layer'] = index
                path.write_text(json.dumps(report, indent=2))
                hidden = layer.forward(hidden, mode='decode')
        ttnn.synchronize_device(mesh)
        if hasattr(model, '_qwen_target_feature_capture') or any(layer.forward is not forward for layer in model.layers):
            raise AssertionError('Feature scope failed to restore the model')
        for chip, part in enumerate(ttnn.get_device_tensors(hidden)):
            expected = initial[..., chip * 2560:(chip + 1) * 2560] + 64
            if not torch.equal(ttnn.to_torch(part), expected):
                raise AssertionError('Feature capture changed the synthetic model output')
        release_owned(ttnn, owned)
        owned.clear()
        for index, feature in zip(report['tap_ids'], capture.outputs(), strict=True):
            for chip, part in enumerate(ttnn.get_device_tensors(feature)):
                expected = initial[..., chip * 2560:(chip + 1) * 2560] + index + 1
                if not torch.equal(ttnn.to_torch(part), expected):
                    raise AssertionError('Post-layer feature changed after original buffers were released')
                report['checks'].append(dict(layer=index, chip=chip, exact=True))
        if len(report['checks']) != 10:
            raise AssertionError('All five taps on both chips required')
    finally:
        if capture is not None:
            capture.close()
        release_owned(ttnn, owned)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
