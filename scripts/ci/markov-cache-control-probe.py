"""Two-chip metadata-only cache protocol probe; no model weights or speed claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Simulator-only execution and a fresh report required')
    import torch
    import ttnn
    from markov_cache_program import build

    def hashes():
        names = ('markov-cache-control-probe.py', 'markov_cache_control.cpp',
            'markov_cache_control.hpp', 'markov_cache_program.py')
        return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        sources=hashes(), checks=[], payload_qualified=False, conditional_matmul_qualified=False,
        performance_qualified=False, hardware_qualified=False)
    mesh, trace, owned = None, None, []

    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage, checks=len(report['checks']))), flush=True)

    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=1048576)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def allocate(rows):
            tensor = ttnn.from_torch(torch.zeros((2, 1, rows, 8), dtype=torch.int64), device=mesh,
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
            owned.append(tensor)
            return tensor

        state, request, decision, status = allocate(65), allocate(1), allocate(1), allocate(1)
        program = build(mesh, state, request, decision, status)

        def upload(tensor, words):
            host = torch.tensor(words, dtype=torch.int64).reshape(2, 1, -1, 8)
            payload = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(payload, tensor)
            ttnn.synchronize_device(mesh)

        def read(tensor):
            shards = ttnn.get_device_tensors(tensor)
            if len(shards) != 2:
                raise AssertionError('Both chips required')
            return [ttnn.to_torch(shard).to(torch.int64).reshape(-1, 8).tolist() for shard in shards]

        def run():
            ttnn.generic_op([state, request, decision, status], program)

        upload(request, [[99] + [0] * 7] * 2)
        run()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.synchronize_device(mesh)

        def step(label, command, token=0, epoch=1, ok=True, hit=None, slot=None):
            commands = [[command, token + chip if command == 0 else token, epoch] + [0] * 5 for chip in range(2)]
            upload(request, commands)
            before_state, before_decision = read(state), read(decision)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            after_state, after_decision = read(state), read(decision)
            results = read(status)
            if read(request) != [[words] for words in commands]:
                raise AssertionError('Request mutated')
            for chip in range(2):
                words = after_decision[chip][0]
                assert results[chip][0] == [int(ok), command] + [0] * 6, (label, chip, results)
                if command == 0:
                    assert words[0] == int(ok) and words[3:5] == [token + chip, epoch], (label, chip, words)
                    if hit is not None:
                        assert words[1] == int(hit), (label, chip, words)
                    if slot is not None:
                        assert words[2] == slot, (label, chip, words)
                    if ok:
                        entry = after_state[chip][words[2] + 1]
                        assert entry[:4] == [int(bool(hit)), token + chip, epoch, words[5]], (label, chip, entry)
                else:
                    assert after_decision[chip] == before_decision[chip], (label, chip, 'decision mutated')
                    if ok:
                        expected = [row[:] for row in before_state[chip]]
                        expected[words[2] + 1][0] = 1
                        assert after_state[chip] == expected, (label, chip, 'commit writes')
                if not ok:
                    assert after_state[chip] == before_state[chip], (label, chip, 'failed command mutated state')
                report['checks'].append(dict(label=label, chip=chip, exact=True))
            return after_decision

        step('cold', 0, 198, hit=False, slot=0)
        stale = read(decision)
        step('uncommitted', 0, 198, hit=False, slot=0)
        pending = read(decision)
        upload(decision, stale)
        step('stale_commit', 1, ok=False)
        upload(decision, pending)
        step('commit', 1)
        step('double_commit', 1, ok=False)
        step('hit', 0, 198, hit=True, slot=0)
        step('hit_commit', 1, ok=False)
        save('eviction')
        for token in range(64):
            step('fill_' + str(token), 0, token, epoch=2, hit=False, slot=token)
            step('commit_' + str(token), 1)
        step('touch_oldest', 0, 0, epoch=2, hit=True, slot=0)
        step('evict', 0, 100, epoch=2, hit=False, slot=1)
        step('commit_eviction', 1)
        step('evicted_miss', 0, 1, epoch=2, hit=False, slot=2)
        old_epoch = read(decision)
        step('new_weights', 0, 1, epoch=3, hit=False, slot=0)
        current_epoch = read(decision)
        upload(decision, old_epoch)
        step('old_epoch_commit', 1, ok=False)
        upload(decision, current_epoch)
        step('new_epoch_commit', 1)
        step('epoch_rollback', 0, 1, epoch=2, ok=False)
        step('invalid_token', 0, 248320, epoch=3, ok=False)
        step('zero_epoch', 0, 1, epoch=0, ok=False)
        step('unknown_command', 99, ok=False)
        exhausted = read(state)
        for chip in range(2):
            exhausted[chip][0][1] = 0xffffffff
        upload(state, exhausted)
        step('counter_overflow', 0, 1, epoch=3, ok=False)
        step('higher_epoch_recovers', 0, 1, epoch=4, hit=False, slot=0)
        if len(report['checks']) != 296:
            raise AssertionError('Incomplete protocol matrix')
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
                for tensor in reversed(owned):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
