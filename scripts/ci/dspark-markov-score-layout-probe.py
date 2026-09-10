"""Exact 15-step native-versus-fused Markov feedback and replay; no speed or coding-quality claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_markov_device import execute as native
from dspark_markov_score_layout import execute as candidate
from dspark_score_layout_gate import qualify
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


SOURCES = ('dspark-markov-score-layout-probe.py', 'dspark_markov_device.py',
    'dspark_markov_score_layout.py', 'dspark_score_layout.py', 'dspark_score_layout_io.cpp',
    'dspark_score_layout_compute.cpp', 'dspark_score_layout_gate.py',
    'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vocabulary', type=int, choices=(64, 248320), default=64)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    if options.output.exists():
        raise ValueError('Fresh Markov chain evidence required')
    root = Path(__file__).parent
    qualification = qualify(root)
    import torch
    import ttnn

    def hashes():
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}

    report = dict(passed=False, closed_cleanly=False, backend='simulator', scope=__doc__,
        vocabulary=options.vocabulary, proposals=15, sources=hashes(), score_layout_gate=qualification,
        eager_checks=[], replay_checks=[], input_checks=[], weight_checks=[])
    mesh, trace, persistent, temporary = None, None, [], []

    def retain(value):
        persistent.append(value)
        return value

    def save(stage):
        report['stage'] = stage
        options.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(stage=stage)), flush=True)

    try:
        save('opening')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576, trace_region_size=268435456)
        mesh.enable_program_cache()
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        replicate = ttnn.ReplicateTensorToMesh(mesh)
        generator = torch.Generator().manual_seed(389113)
        vocabulary = options.vocabulary
        weights_host = [torch.randn((1, 1, vocabulary, 256), generator=generator).bfloat16() / 16,
            torch.randn((1, 1, 256, vocabulary), generator=generator).bfloat16() / 16]
        weights_host[0][:, :, 0].zero_()
        weights = [retain(ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=layout, device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=replicate))
            for value, layout in zip(weights_host, (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT), strict=True)]
        patterns = []
        for pattern in range(3):
            anchor = torch.tensor([1, vocabulary - 1] if pattern == 0 else [vocabulary - 1, 2]
                if pattern == 1 else [0, 0], dtype=torch.int64).reshape(2, 1, 1, 1)
            base = torch.randn((2, 1, 15, vocabulary), generator=generator) / 8
            if pattern == 2:
                base.zero_()
            patterns.append((anchor, base))
        layouts = (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)
        dtypes = (ttnn.uint32, ttnn.float32)
        hosts = [[ttnn.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=shard)
            for value, dtype, layout in zip(pair, dtypes, layouts, strict=True)] for pair in patterns]
        inputs = [retain(ttnn.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=shard,
            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            for value, dtype, layout in zip(patterns[0], dtypes, layouts, strict=True)]

        def read(value):
            return [ttnn.to_torch(tensor).clone() for tensor in ttnn.get_device_tensors(value)]

        def same(actual, expected):
            return torch.equal(actual.view(torch.int32), expected.view(torch.int32))

        def integrity(pattern, stage):
            for operand, value in enumerate(inputs):
                for chip, actual in enumerate(read(value)):
                    expected = patterns[pattern][operand][chip:chip + 1]
                    exact = torch.equal(actual.long(), expected) if operand == 0 else same(actual, expected)
                    if not exact:
                        raise AssertionError('Caller-owned input changed')
                    report['input_checks'].append(dict(stage=stage, pattern=pattern, operand=operand, chip=chip, exact=True))

        def weight_check(stage):
            for operand, value in enumerate(weights):
                for chip, actual in enumerate(read(value)):
                    if not torch.equal(actual.view(torch.int16), weights_host[operand].view(torch.int16)):
                        raise AssertionError('Caller-owned learned-head-shaped weight changed')
                    report['weight_checks'].append(dict(stage=stage, operand=operand, chip=chip, exact=True))

        def update(pattern):
            for host, device in zip(hosts[pattern], inputs, strict=True):
                ttnn.copy_host_to_device_tensor(host, device)

        def compare(records, expected, checks, pattern, repetition=None):
            if len(records) != 15:
                raise AssertionError('Complete fifteen-token chain required')
            for step, record in enumerate(records):
                for chip, (token, scores) in enumerate(zip(read(record['token']), read(record['scores']), strict=True)):
                    reference_token, reference_scores = expected[step][chip]
                    if not torch.equal(token, reference_token) or not same(scores, reference_scores):
                        raise AssertionError(f'Feedback differs: pattern={pattern} step={step} chip={chip}')
                    checks.append(dict(pattern=pattern, repetition=repetition, step=step, chip=chip,
                        token_exact=True, scores_exact=True))

        weight_check('before')
        references = []
        for pattern in range(3):
            update(pattern)
            save(f'native_{pattern}')
            records = native(ttnn, *inputs, *weights, temporary)
            references.append([list(zip(read(record['token']), read(record['scores']), strict=True)) for record in records])
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, temporary)
            temporary.clear()
            save(f'candidate_{pattern}')
            records = candidate(ttnn, mesh, *inputs, *weights, temporary)
            compare(records, references[pattern], report['eager_checks'], pattern)
            integrity(pattern, 'eager')
            ttnn.synchronize_device(mesh)
            release_owned(ttnn, temporary)
            temporary.clear()
        if any(int(token.item()) != 0 for step in references[2] for token, scores in step):
            raise AssertionError('All-zero tie fixture must preserve native first-index selection')
        save('capture')
        trace, records = capture_operation(ttnn, mesh, lambda: candidate(ttnn, mesh, *inputs, *weights, persistent))
        outputs = [record[name] for record in records for name in ('token', 'scores')]
        bindings = [addresses(ttnn, value) for value in inputs + weights + outputs]
        poison_token = ttnn.from_torch(torch.full(tuple(records[0]['token'].shape), 4294967295, dtype=torch.int64),
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate)
        poison_scores = ttnn.from_torch(torch.full((1, 1, 1, vocabulary), float('nan')),
            dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate)
        for repetition, pattern in enumerate((0, 1, 2, 0)):
            save(f'replay_{repetition}')
            update(pattern)
            for record in records:
                ttnn.copy_host_to_device_tensor(poison_token, record['token'])
                ttnn.copy_host_to_device_tensor(poison_scores, record['scores'])
                if (not all(torch.isnan(value).all() for value in read(record['scores']))
                        or not all((value.long() == 4294967295).all() for value in read(record['token']))):
                    raise AssertionError('Output poison missing')
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            compare(records, references[pattern], report['replay_checks'], pattern, repetition)
            integrity(pattern, f'replay_{repetition}')
            if bindings != [addresses(ttnn, value) for value in inputs + weights + outputs]:
                raise AssertionError('Feedback trace bindings changed')
        weight_check('after')
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
                release_owned(ttnn, temporary + persistent)
                ttnn.close_mesh_device(mesh)
                report['closed_cleanly'] = True
        finally:
            report['sources_after'] = hashes()
            if report['sources_after'] != report['sources'] or not report['closed_cleanly']:
                report['passed'] = False
            save('complete' if report['passed'] else 'failed')


if __name__ == '__main__':
    main()
