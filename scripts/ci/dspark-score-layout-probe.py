"""Exact fused FP32 score-row layout gate; no learned dot-product change or TG claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_score_layout import execute
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('dspark-score-layout-probe.py', 'dspark_score_layout.py', 'dspark_score_layout_io.cpp',
    'dspark_score_layout_compute.cpp', 'attention_batch.py', 'gdn_multitoken_conv.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocabulary', type=int, choices=(64, 248320), default=64)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh score-layout evidence required')
    import torch
    import ttnn
    root = Path(__file__).parent
    def hashes():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__, sources=hashes(),
        vocabulary=options.vocabulary, rows=[0, 6, 14], eager_checks=[], replay_checks=[])
    mesh, owned, temporary, traces = None, [], [], []
    def retain(value):
        owned.append(value)
        return value
    def transient(value):
        temporary.append(value)
        return value
    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage)), flush=True)
    try:
        save('open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=134217728)
        mesh.enable_program_cache()
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        generator = torch.Generator().manual_seed(389112)
        patterns = []
        for pattern in range(2):
            base = torch.randn((2, 1, 15, options.vocabulary), generator=generator)
            bias = torch.randn((2, 1, 1, options.vocabulary), generator=generator) / 8
            for chip in range(2):
                for step in report['rows']:
                    base[chip, 0, step, (options.vocabulary - 1 - step - chip - pattern) % options.vocabulary] = 20
            patterns.append((base, bias))
        host = [[ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value in pair] for pair in patterns]
        inputs = [retain(ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)) for value in patterns[0]]
        def read(value):
            return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(value)]
        def same(actual, expected):
            return torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        def update(pattern):
            for value, destination in zip(host[pattern], inputs, strict=True):
                ttnn.copy_host_to_device_tensor(value, destination)
        def input_check(pattern):
            for operand, value in enumerate(inputs):
                for chip, actual in enumerate(read(value)):
                    if not same(actual, patterns[pattern][operand][chip:chip + 1]):
                        raise AssertionError('Caller-owned score input changed')
        references = {}
        workers = (1, 110) if options.vocabulary == 64 else (110,)
        for pattern in range(2):
            update(pattern)
            for step in report['rows']:
                save(f'eager_{pattern}_{step}')
                row = transient(ttnn.slice(inputs[0], (0, 0, step, 0), (1, 1, step + 1, options.vocabulary)))
                scores = transient(ttnn.add(row, inputs[1], memory_config=ttnn.DRAM_MEMORY_CONFIG))
                native = transient(ttnn.untilize(scores, use_multicore=True))
                expected = read(native)
                tokens = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(
                    transient(ttnn.argmax(native, dim=-1, keepdim=False)))]
                references[(pattern, step)] = (expected, tokens)
                for limit in workers:
                    result = transient(execute(ttnn, mesh, *inputs, step, lambda value: value, worker_limit=limit))
                    actual = read(result)
                    actual_tokens = read(transient(ttnn.argmax(result, dim=-1, keepdim=False)))
                    for chip in range(2):
                        cpu = patterns[pattern][0][chip:chip + 1, :, step:step + 1] + patterns[pattern][1][chip:chip + 1]
                        if not same(actual[chip], expected[chip]) or not same(actual[chip], cpu):
                            raise AssertionError('Fused score differs from native or FP32 reference bits')
                        if not torch.equal(actual_tokens[chip], tokens[chip]):
                            raise AssertionError('Fused score changes native token selection')
                        report['eager_checks'].append(dict(pattern=pattern, step=step, workers=limit, chip=chip,
                            exact=True, native_token_exact=True, cpu_exact=True))
                input_check(pattern)
                ttnn.synchronize_device(mesh)
                release_owned(ttnn, temporary)
                temporary.clear()
        outputs = []
        for step in report['rows']:
            save(f'capture_{step}')
            trace, output = capture_operation(ttnn, mesh, lambda step=step: execute(ttnn, mesh, *inputs, step, retain))
            traces.append(trace)
            outputs.append(output)
        bindings = [addresses(ttnn, value) for value in inputs + outputs]
        poison = ttnn.from_torch(torch.full((1, 1, 1, options.vocabulary), float('nan')),
            dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        for repetition, pattern in enumerate((0, 1, 0)):
            update(pattern)
            for step, trace, output in zip(report['rows'], traces, outputs, strict=True):
                save(f'replay_{repetition}_{step}')
                ttnn.copy_host_to_device_tensor(poison, output)
                if not all(torch.isnan(value).all() for value in read(output)):
                    raise AssertionError('Score output poison missing')
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = read(output)
                for chip in range(2):
                    if not same(actual[chip], references[(pattern, step)][0][chip]):
                        raise AssertionError('Changed-input score replay differs from native reference')
                    report['replay_checks'].append(dict(repetition=repetition, pattern=pattern, step=step,
                        chip=chip, exact=True, output_poison_replaced=True))
                input_check(pattern)
                if bindings != [addresses(ttnn, value) for value in inputs + outputs]:
                    raise AssertionError('Score trace bindings changed')
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
                release_owned(ttnn, temporary + owned)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            if report['sources_after'] != report['sources'] or not report['closed_cleanly']:
                report['passed'] = False
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
