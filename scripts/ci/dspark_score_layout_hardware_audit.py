"""Exact learned-weight full-vocabulary feedback gate before any request timing."""

import hashlib
import os
from pathlib import Path

from attention_batch import capture_operation
from dspark_markov_device import execute as native
from dspark_markov_score_layout import execute as candidate
from dspark_score_layout_hardware_gate import qualify
from feature_projection import require_projection_environment
from gdn_multitoken_conv import addresses, release_owned


def audit(operations, mesh, predecessor, successor):
    require_projection_environment(os.environ, True)
    admission = qualify(Path(__file__).parent)
    if list(mesh.shape) != [1, 2]:
        raise ValueError('Allocated two-chip mesh required')
    import torch

    persistent, temporary, trace = [], [], None
    report = dict(passed=False, released_cleanly=False, admission=admission,
        vocabulary=248320, proposals=15, eager_checks=[], replay_checks=[], input_checks=[],
        scope=__doc__, timing_qualified=False)
    report['weight_bindings'] = [addresses(operations, weight) for weight in (predecessor, successor)]

    def read(value):
        shards = operations.get_device_tensors(value)
        if len(shards) != 2:
            raise AssertionError('Both independent replicas required')
        return [operations.to_torch(shard).clone() for shard in shards]

    def weight_hashes():
        return [[hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for value in read(weight)] for weight in (predecessor, successor)]

    def retain(value):
        persistent.append(value)
        return value

    generator = torch.Generator().manual_seed(389114)
    patterns = [(torch.tensor(anchors, dtype=torch.int64).reshape(2, 1, 1, 1),
        torch.randn((2, 1, 15, 248320), generator=generator) / 8)
        for anchors in ([1596, 248319], [248318, 2])]
    mapper = operations.ShardTensorToMesh(mesh, dim=0)
    dtypes = (operations.uint32, operations.float32)
    layouts = (operations.ROW_MAJOR_LAYOUT, operations.TILE_LAYOUT)
    inputs = []
    try:
        report['weight_hashes_before'] = weight_hashes()
        host = [[operations.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=mapper)
            for value, dtype, layout in zip(pattern, dtypes, layouts, strict=True)] for pattern in patterns]
        for value, dtype, layout in zip(patterns[0], dtypes, layouts, strict=True):
            inputs.append(retain(operations.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=mapper,
                device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)))

        def update(pattern):
            for source, destination in zip(host[pattern], inputs, strict=True):
                operations.copy_host_to_device_tensor(source, destination)

        def integrity(pattern, stage):
            for operand, tensor in enumerate(inputs):
                for chip, value in enumerate(read(tensor)):
                    expected = patterns[pattern][operand][chip:chip + 1]
                    exact = (torch.equal(value.long(), expected) if operand == 0 else
                        torch.equal(value.view(torch.int32), expected.view(torch.int32)))
                    if not exact:
                        raise AssertionError('Borrowed feedback input changed')
                    report['input_checks'].append(dict(pattern=pattern, stage=stage, operand=operand, chip=chip, exact=True))

        def compare(records, expected, pattern, repetition=None):
            if len(records) != 15:
                raise AssertionError('Complete fifteen-token feedback required')
            checks = report['eager_checks' if repetition is None else 'replay_checks']
            for step, record in enumerate(records):
                for chip, (token, scores) in enumerate(zip(read(record['token']), read(record['scores']), strict=True)):
                    target_token, target_scores = expected[step][chip]
                    if (not torch.equal(token, target_token) or not torch.equal(
                            scores.view(torch.int32), target_scores.view(torch.int32))):
                        raise AssertionError(f'Learned feedback mismatch: pattern={pattern}, step={step}, chip={chip}')
                    checks.append(dict(pattern=pattern, repetition=repetition, step=step, chip=chip,
                        token_exact=True, scores_exact=True))

        references = []
        for pattern in range(2):
            update(pattern)
            records = native(operations, *inputs, predecessor, successor, temporary)
            references.append([list(zip(read(record['token']), read(record['scores']), strict=True)) for record in records])
            operations.synchronize_device(mesh)
            release_owned(operations, temporary)
            temporary.clear()
            records = candidate(operations, mesh, *inputs, predecessor, successor, temporary)
            compare(records, references[pattern], pattern)
            integrity(pattern, 'eager')
            operations.synchronize_device(mesh)
            release_owned(operations, temporary)
            temporary.clear()
        trace, records = capture_operation(operations, mesh,
            lambda: candidate(operations, mesh, *inputs, predecessor, successor, persistent))
        outputs = [record[name] for record in records for name in ('token', 'scores')]
        bindings = [addresses(operations, value) for value in [*inputs, predecessor, successor, *outputs]]
        poison = operations.from_torch(torch.full((1, 1, 1, 248320), float('nan')),
            dtype=operations.float32, layout=operations.ROW_MAJOR_LAYOUT,
            mesh_mapper=operations.ReplicateTensorToMesh(mesh))
        for repetition, pattern in enumerate((0, 1, 0)):
            update(pattern)
            for record in records:
                operations.copy_host_to_device_tensor(poison, record['scores'])
                if not all(torch.isnan(value).all() for value in read(record['scores'])):
                    raise AssertionError('Replay poison missing')
            operations.execute_trace(mesh, trace, cq_id=0, blocking=True)
            compare(records, references[pattern], pattern, repetition)
            integrity(pattern, f'replay_{repetition}')
            if bindings != [addresses(operations, value) for value in [*inputs, predecessor, successor, *outputs]]:
                raise AssertionError('Replay bindings changed')
        report['weight_hashes_after'] = weight_hashes()
        if report['weight_hashes_after'] != report['weight_hashes_before']:
            raise AssertionError('Borrowed learned weights changed')
        report['passed'] = True
    finally:
        operations.synchronize_device(mesh)
        if trace is not None:
            operations.release_trace(mesh, trace)
        release_owned(operations, temporary + persistent)
        report['released_cleanly'] = True
    return report
