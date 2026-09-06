"""Simulator-only same-address mask refresh, including backwards rollback positions."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ci'))
from attention_head_fold import causal_mask
from attention_mask_replay import execute, prepare, validate_ticket


def main(*, hardware=False):
    spec = importlib.util.spec_from_file_location('guard', Path(__file__).with_name('gdn-multitoken.py'))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    if hardware:
        if os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1':
            raise RuntimeError('Explicit hardware allocation required')
        if os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_SLOW_DISPATCH_MODE'):
            raise RuntimeError('Fast-dispatch hardware required')
    else:
        guard.require_simulator(os.environ)
    import torch
    import ttnn

    path = Path('/experiment/results/attention-mask-replay.json' if hardware else os.environ['QWEN_SIM_REPORT'])
    report = dict(passed=False, checks=[], backend='hardware' if hardware else 'ttsim',
        trace_replays=0, scope='Same-address causal masks only; no attention or speed claim',
        sources={name: hashlib.sha256((Path(__file__).resolve().parents[2] / 'scripts/ci' / name).read_bytes()).hexdigest()
                 for name in ('attention_mask_replay.py', 'attention_mask_replay.cpp')})
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=16777216 if hardware else 0)
        if hardware:
            mesh.enable_program_cache()
        for capacity in (4352, 16640):
            for rows, batches, offset in ((1, 1, 0), (3, 2, 1), (4, 3, 4)):
                report['last_stage'] = dict(capacity=capacity, rows=rows, batches=batches)
                path.write_text(json.dumps(report, indent=2))
                print(json.dumps(report['last_stage']), flush=True)
                initial_words = torch.zeros(8, dtype=torch.int32)
                initial_words[0] = capacity - 256
                positions = ttnn.from_torch(initial_words, dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                mask = ttnn.from_torch(torch.zeros(batches, 1, rows * 12, capacity, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                captured = None
                try:
                    addresses = [tuple(part.buffer_address() for part in ttnn.get_device_tensors(value))
                                 for value in (positions, mask)]
                    program = prepare(mesh, positions, mask, rows=rows, batches=batches, offset=offset, capacity=capacity)
                    if hardware:
                        from attention_batch import capture_operation
                        execute(positions, mask, program)
                        ttnn.synchronize_device(mesh)
                        captured, _ = capture_operation(ttnn, mesh, lambda: execute(positions, mask, program))
                    starts = (capacity - 256, capacity - offset - rows * batches, capacity - 249, capacity - 256)
                    for start in starts:
                        validate_ticket(start, offset + rows * batches, capacity)
                        words = torch.zeros(8, dtype=torch.int32)
                        words[0] = start
                        staged = ttnn.from_torch(words, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                        ttnn.copy_host_to_device_tensor(staged, positions)
                        if hardware:
                            ttnn.execute_trace(mesh, captured, cq_id=0, blocking=True)
                            report['trace_replays'] += 1
                        else:
                            execute(positions, mask, program)
                        ttnn.synchronize_device(mesh)
                        expected = torch.cat([causal_mask(rows, start + offset + batch * rows, capacity)
                                              for batch in range(batches)], dim=0)
                        for chip, value in enumerate(ttnn.get_device_tensors(mask)):
                            if not torch.equal(ttnn.to_torch(value), expected):
                                raise AssertionError('Refreshed mask differs from native host oracle')
                            report['checks'].append(dict(capacity=capacity, rows=rows, batches=batches,
                                offset=offset, start=start, chip=chip, exact=True))
                        if addresses != [tuple(part.buffer_address() for part in ttnn.get_device_tensors(value))
                                         for value in (positions, mask)]:
                            raise AssertionError('Replay replaced a captured metadata address')
                finally:
                    if captured is not None:
                        ttnn.release_trace(mesh, captured)
                    ttnn.deallocate(mask)
                    ttnn.deallocate(positions)
        if len(report['checks']) != 48:
            raise AssertionError('Incomplete mask replay matrix')
        if hardware and report['trace_replays'] != 24:
            raise AssertionError('Incomplete captured mask refresh coverage')
        report['passed'] = True
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
