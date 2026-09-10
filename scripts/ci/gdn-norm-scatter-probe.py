"""Simulator-only isolated norm comparison; not end-to-end qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import gdn_norm_scatter as scatter
from feature_projection import require_projection_environment
from attention_batch import capture_operation


def hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('gdn-norm-scatter-probe.py', 'gdn_norm_scatter.py',
                         'gdn_vsplit_norm_batch.py', 'gdn_vsplit.py', 'attention_batch.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, closed=False, backend='simulator', output_poisoned=True,
                  checks=[], replay_checks=[], poison_checks=[], before=hashes())
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()
        generator = torch.Generator().manual_seed(381627)
        root = Path(__file__).resolve().parents[2] / 'hardware-evidence.local/34009341359/qwen-hardware-inventory-34009341359/gdn-source'
        kernels = (scatter.batch.load_kernels(root), scatter.load_kernels(root))
        for rows in (1, 16, 32):
            for case in range(2):
                owned = []

                def upload(value, dtype, layout, sharded=False):
                    tensor = ttnn.from_torch(value, dtype=dtype, layout=layout,
                        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if sharded else ttnn.ReplicateTensorToMesh(mesh))
                    owned.append(tensor)
                    return tensor

                try:
                    tensors = [upload(torch.zeros(1, 1, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                               for unused in range(4)]
                    bridge = torch.randn(2 * rows, 1, 96, 32, generator=generator) * (1 if case == 0 else 7)
                    tensors.append(upload(bridge, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, sharded=True))
                    tensors.append(upload(torch.zeros(1, 1, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT))
                    tensors.append(upload(torch.randn(1, rows, 3072, generator=generator), ttnn.bfloat16, ttnn.TILE_LAYOUT))
                    tensors.append(upload(torch.randn(1, 1, 128, generator=generator), ttnn.bfloat16, ttnn.TILE_LAYOUT))
                    tensors.append(upload(torch.zeros(1, 32, 3072), ttnn.bfloat16, ttnn.TILE_LAYOUT))
                    shards = [ttnn.get_device_tensors(value) for value in tensors]
                    poison_value = torch.full((1, 32, 3072), 8192, dtype=torch.bfloat16)
                    poison_host = ttnn.from_torch(poison_value, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

                    def poison_output():
                        ttnn.copy_host_to_device_tensor(poison_host, tensors[8])
                        ttnn.synchronize_device(mesh)
                        verified = all(torch.equal(ttnn.to_torch(value), poison_value) for value in shards[8])
                        report['poison_checks'].append(dict(rows=rows, case=case, verified=verified))
                        if not verified:
                            raise AssertionError('Output poison did not reach both chips')

                    before = [[ttnn.to_torch(value).clone() for value in local] for local in shards[:8]]
                    outputs = []
                    for variant in kernels:
                        program = scatter.batch.split.build_program(ttnn, mesh, shards, variant, 'norm_gate', rows)
                        poison_output()
                        ttnn.generic_op(tensors, program)
                        ttnn.synchronize_device(mesh)
                        outputs.append([ttnn.to_torch(value).clone() for value in shards[8]])
                    for chip in range(2):
                        exact = torch.equal(outputs[0][chip], outputs[1][chip])
                        unchanged = all(torch.equal(before[index][chip], ttnn.to_torch(shards[index][chip])) for index in range(8))
                        finite = bool(torch.isfinite(outputs[1][chip]).all())
                        padding_zero = bool((outputs[1][chip][:, rows:] == 0).all())
                        written = not bool((outputs[1][chip] == 8192).any())
                        report['checks'].append(dict(rows=rows, case=case, chip=chip, exact=exact,
                            unchanged=unchanged, finite=finite, padding_zero=padding_zero, written=written))
                        if not (exact and unchanged and finite and padding_zero and written):
                            raise AssertionError('Norm scatter comparison failed')
                    print(json.dumps(report['checks'][-2:]), flush=True)
                    changed_bridge = bridge.flip(-1).contiguous()
                    host_update = ttnn.from_torch(changed_bridge, dtype=ttnn.float32,
                        layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
                    control_program = scatter.batch.split.build_program(ttnn, mesh, shards, kernels[0], 'norm_gate', rows)
                    trace, unused = capture_operation(ttnn, mesh, lambda: ttnn.generic_op(tensors, program))
                    try:
                        ttnn.copy_host_to_device_tensor(host_update, tensors[4])
                        poison_output()
                        ttnn.generic_op(tensors, control_program)
                        ttnn.synchronize_device(mesh)
                        expected = [ttnn.to_torch(value).clone() for value in shards[8]]
                        for replay in range(2):
                            poison_output()
                            ttnn.execute_trace(mesh, trace, blocking=True)
                            for chip in range(2):
                                actual = ttnn.to_torch(shards[8][chip])
                                exact = torch.equal(actual, expected[chip])
                                stale_detected = not torch.equal(expected[chip], outputs[0][chip])
                                padding_zero = bool((actual[:, rows:] == 0).all())
                                written = not bool((actual == 8192).any())
                                unchanged = all(torch.equal(
                                    changed_bridge[chip * rows:(chip + 1) * rows] if index == 4 else before[index][chip],
                                    ttnn.to_torch(shards[index][chip])) for index in range(8))
                                report['replay_checks'].append(dict(rows=rows, case=case, chip=chip,
                                    replay=replay, exact=exact, stale_detected=stale_detected,
                                    unchanged=unchanged, padding_zero=padding_zero, written=written))
                                if not (exact and stale_detected and unchanged and padding_zero and written):
                                    raise AssertionError('Changed-input norm replay failed')
                    finally:
                        ttnn.release_trace(mesh, trace)
                    print(json.dumps(report['replay_checks'][-4:]), flush=True)
                finally:
                    for tensor in reversed(owned):
                        ttnn.deallocate(tensor)
        report['passed'] = (len(report['checks']) == 12 and len(report['replay_checks']) == 24
                            and len(report['poison_checks']) == 30)
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                report['closed'] = True
        finally:
            report['after'] = hashes()
            report['passed'] = report['passed'] and report['closed'] and report['before'] == report['after']
            options.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
