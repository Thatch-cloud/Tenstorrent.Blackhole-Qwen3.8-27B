"""Two-chip exact compact selection versus native score layout and argmax."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from compact_score_device import execute_local_winners, reduce_winners
from compact_markov import execute as compact_feedback, validate_readback
from dspark_markov_score_layout import execute as native_feedback
from dspark_score_layout import execute as native_scores
from gdn_multitoken_conv import addresses, release_owned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocabulary', type=int, choices=(64, 248320), default=248320)
    options = parser.parse_args()
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_CARDS_ALLOCATED') == '1' or options.output.exists()):
        raise ValueError('Simulator-only execution and fresh evidence required')
    import torch
    import ttnn

    directory = Path(__file__).parent
    names = (Path(__file__).name, 'compact_score_device.py', 'compact_score_io.cpp',
        'compact_score_compute.cpp', 'compact_score_reduce.cpp', 'dspark_score_layout.py',
        'dspark_score_layout_io.cpp', 'dspark_score_layout_compute.cpp', 'attention_batch.py',
        'gdn_multitoken_conv.py', 'compact_markov.py', 'dspark_markov_device.py',
        'dspark_markov_score_layout.py')
    def hashes():
        return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', vocabulary=options.vocabulary,
        checks=[], immutable_checks=[], feedback_checks=[], sources=hashes(), performance_qualified=False)
    mesh, owned, traces = None, [], []
    def retain(value):
        owned.append(value)
        return value
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(stage=stage)), flush=True)
    def read(value):
        return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]
    vocabulary = options.vocabulary
    generator = torch.Generator().manual_seed(385270)
    patterns = []
    for pattern in range(4):
        base = torch.randn((2, 1, 15, vocabulary), generator=generator)
        bias = torch.zeros((2, 1, 1, vocabulary))
        if pattern == 1:
            base.fill_(-1)
            base[..., 31] = 2
            base[..., vocabulary - 1] = 2
        elif pattern == 2:
            base.fill_(-1)
            base[..., 17] = -0.0
            base[..., 33] = 0.0
        elif pattern == 3:
            base.zero_()
            base[..., 19] = torch.finfo(torch.float32).tiny / 2
            base[..., vocabulary - 2] = torch.finfo(torch.float32).tiny
        patterns.append((base, bias))
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inputs = [retain(ttnn.from_torch(value, device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)) for value in patterns[0]]
        def update(pattern):
            for value, destination in zip(patterns[pattern], inputs, strict=True):
                host = ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(host, destination)
        def run(step):
            scores = native_scores(ttnn, mesh, *inputs, step, retain)
            token = retain(ttnn.argmax(scores, dim=-1, keepdim=False))
            winners = execute_local_winners(ttnn, mesh, *inputs, step, retain)
            result = reduce_winners(ttnn, mesh, winners, vocabulary, retain)
            return scores, token, winners, result
        def verify(outputs, pattern, step, mode):
            scores, tokens, winners, results = [read(value) for value in outputs]
            for chip in range(2):
                row = scores[chip].reshape(-1)
                reference = int(tokens[chip].reshape(-1)[0])
                result = results[chip].reshape(-1).tolist()
                if (result[1] != 0 or result[0] != reference or result[3] != reference
                        or any(result[4:]) or reference != int(torch.argmax(row))):
                    raise AssertionError(f'Compact/native/CPU argmax mismatch {pattern=} {step=} {chip=} {result=} {reference=}')
                records = winners[chip].reshape(-1, 8)
                tiles, workers = vocabulary // 32, records.shape[0]
                for worker, record in enumerate(records):
                    start, end = worker * tiles // workers * 32, (worker + 1) * tiles // workers * 32
                    token = start + int(torch.argmax(row[start:end]))
                    expected_bits = int(row[token].view(torch.int32)) & 0xffffffff
                    if record.tolist() != [expected_bits, token, 0, 0, 0, 0, 0, 0]:
                        raise AssertionError('Local winner differs from exact native score row')
                report['checks'].append(dict(pattern=pattern, step=step, mode=mode, chip=chip, exact=True))
            for operand, value in enumerate(inputs):
                for chip, actual in enumerate(read(value)):
                    if not torch.equal(actual.view(torch.int32), patterns[pattern][operand][chip:chip + 1].view(torch.int32)):
                        raise AssertionError('Score input changed')
                    report['immutable_checks'].append(dict(pattern=pattern, step=step, mode=mode,
                        operand=operand, chip=chip, exact=True))
        for step in (0, 14):
            save('eager_' + str(step))
            output = run(step)
            ttnn.synchronize_device(mesh)
            verify(output, 0, step, 'eager')
        captured = []
        for step in (0, 14):
            save('capture_' + str(step))
            trace, outputs = capture_operation(ttnn, mesh, lambda step=step: run(step))
            traces.append(trace)
            captured.append((step, outputs))
        bindings = [addresses(ttnn, value) for value in inputs]
        poison = ttnn.from_torch(torch.full((2, 1, 1, 8), 0xffffffff, dtype=torch.int64),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
        for pattern in (1, 2, 3, 0):
            update(pattern)
            for trace, (step, outputs) in zip(traces, captured, strict=True):
                save(f'replay_{pattern}_{step}')
                ttnn.copy_host_to_device_tensor(poison, outputs[-1])
                if any(not torch.all(value == 0xffffffff) for value in read(outputs[-1])):
                    raise AssertionError('Output poison not installed')
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                verify(outputs, pattern, step, 'replay')
                if bindings != [addresses(ttnn, value) for value in inputs]:
                    raise AssertionError('Input bindings changed')
        if len(report['checks']) != 20 or len(report['immutable_checks']) != 40:
            raise AssertionError('Complete two-step eager/replay matrix required')
        for trace in reversed(traces):
            ttnn.release_trace(mesh, trace)
        traces.clear()
        save('feedback_setup')
        feedback_width = 64
        predecessor_host = torch.randn((2, 1, feedback_width, 256), generator=generator).bfloat16() / 16
        successor_host = torch.randn((2, 1, 256, feedback_width), generator=generator).bfloat16() / 16
        base_host = torch.randn((2, 1, 15, feedback_width), generator=generator) / 8
        anchor_host = torch.tensor([1, 63], dtype=torch.int64).reshape(2, 1, 1, 1)
        feedback_inputs = [retain(ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper))
            for value, dtype, layout in ((anchor_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                (base_host, ttnn.float32, ttnn.TILE_LAYOUT),
                (predecessor_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                (successor_host, ttnn.bfloat16, ttnn.TILE_LAYOUT))]
        def feedback_run():
            return compact_feedback(ttnn, mesh, *feedback_inputs, owned)
        def feedback_verify(records, mode):
            native = native_feedback(ttnn, mesh, *feedback_inputs, owned)
            tokens = [[int(value.item()) for value in read(record['token'])] for record in records]
            diagnostics = [[value.reshape(-1).tolist() for value in read(record['diagnostic'])] for record in records]
            validate_readback(diagnostics, tokens, feedback_width)
            for step, record in enumerate(native):
                for chip, value in enumerate(read(record['token'])):
                    if tokens[step][chip] != int(value.item()):
                        raise AssertionError('Compact Markov feedback differs from native')
                    report['feedback_checks'].append(dict(mode=mode, step=step, chip=chip, exact=True))
        save('feedback_eager')
        feedback_verify(feedback_run(), 'eager')
        save('feedback_capture')
        trace, feedback_records = capture_operation(ttnn, mesh, feedback_run)
        traces.append(trace)
        changed_anchor = ttnn.from_torch(torch.tensor([63, 2], dtype=torch.int64).reshape(2, 1, 1, 1),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
        changed_base = ttnn.from_torch(-base_host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        ttnn.copy_host_to_device_tensor(changed_anchor, feedback_inputs[0])
        ttnn.copy_host_to_device_tensor(changed_base, feedback_inputs[1])
        save('feedback_replay')
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        feedback_verify(feedback_records, 'replay')
        if len(report['feedback_checks']) != 60:
            raise AssertionError('Complete two-chip fifteen-step feedback matrix required')
        for trace in reversed(traces):
            ttnn.release_trace(mesh, trace)
        traces.clear()
        save('invalid_feedback_chain')
        invalid_feedback_base = base_host.clone()
        invalid_feedback_base[:, :, 4, :] = float('nan')
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(invalid_feedback_base, dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper), feedback_inputs[1])
        invalid_records = feedback_run()
        invalid_tokens = [[int(value.item()) for value in read(record['token'])] for record in invalid_records]
        invalid_diagnostics = [[value.reshape(-1).tolist() for value in read(record['diagnostic'])]
                               for record in invalid_records]
        if invalid_tokens[4] != [0xffffffff, 0xffffffff]:
            raise AssertionError('Invalid proposal IDs must remain visible to the native proposal reader')
        try:
            validate_readback(invalid_diagnostics, invalid_tokens, feedback_width)
        except ValueError:
            report['invalid_feedback_chain_rejected'] = True
        else:
            raise AssertionError('Invalid feedback chain was accepted')
        save('nonfinite_feedback_safety')
        invalid_base = patterns[0][0].clone()
        invalid_base[..., 0] = float('nan')
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(invalid_base, dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper), inputs[0])
        invalid_winners = execute_local_winners(ttnn, mesh, *inputs, 0, retain)
        invalid_result = reduce_winners(ttnn, mesh, invalid_winners, vocabulary, retain)
        report['invalid_checks'] = []
        for chip, value in enumerate(read(invalid_result)):
            result = value.reshape(-1).tolist()
            if result[0] != 0xffffffff or result[1] != 1 or result[3] != 0:
                raise AssertionError('Nonfinite score must signal invalid while keeping feedback in bounds')
            report['invalid_checks'].append(dict(chip=chip, rejected=True, safe_feedback=True))
        report['passed'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.synchronize_device(mesh)
                for trace in reversed(traces):
                    ttnn.release_trace(mesh, trace)
                release_owned(ttnn, owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            report['passed'] = report['passed'] and report['closed_cleanly'] and report['sources'] == report['sources_after']
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
