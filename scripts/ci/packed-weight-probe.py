"""Simulator-only bytewise packed-weight validation, including single-chip negative controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned
from packed_weight_check import compare_packed_weights, read_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, checks=[], sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('packed_weight_check.py', 'packed_weight_check.cpp')})
    mesh, owned = None, []
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        generator = torch.Generator().manual_seed(3876)
        def upload(value):
            result = ttnn.from_torch(value, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
            owned.append(result)
            return result
        for rows, columns in ((32, 32), (64, 96), (320, 256)):
            gate = torch.randn((2, 1, rows, columns), generator=generator).bfloat16()
            up = torch.randn(gate.shape, generator=generator).bfloat16()
            packed = torch.stack((gate.reshape(2, 1, rows, columns // 32, 32),
                up.reshape(2, 1, rows, columns // 32, 32)), dim=-2).reshape(2, 1, rows, columns * 2)
            device_packed = upload(packed)
            for offset, host in enumerate((gate, up)):
                for changed_chip in (None, 0, 1):
                    operand = host.clone()
                    if changed_chip is not None:
                        operand[changed_chip, 0, rows - 1, columns - 1] = 32
                    separate = upload(operand)
                    result = compare_packed_weights(mesh, device_packed, separate, offset, owned)
                    checks = read_comparison(ttnn, result, rows * columns // 1024)
                    if any(check['exact'] != (check['chip'] != changed_chip) for check in checks):
                        raise AssertionError('Packed-weight positive or single-chip negative control failed')
                    report['checks'].append(dict(rows=rows, columns=columns, offset=offset,
                        changed_chip=changed_chip, chips=checks))
                    options.output.write_text(json.dumps(report, indent=2))
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = len(report['checks']) == 18
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
