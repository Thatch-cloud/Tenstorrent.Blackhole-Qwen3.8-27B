"""Simulator gate for full-vocabulary chunked candidates, excluding LM-head projection."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from draft_shared_head import local_head_candidates, merge_chunk_candidates
from feature_projection import require_projection_environment
from gdn_multitoken_conv import release_owned


def candidate_fixture(rows=8):
    import torch

    boundaries = torch.tensor([0, 32767, 32768, 65535, 65536, 98303, 98304, 124159,
        124160, 156927, 156928, 189695, 189696, 222463, 222464, 248319])
    if type(rows) is not int or rows not in (8, 32):
        raise ValueError('Explicit eight/32-row fixture required')
    logits = torch.full((rows, 248320), -1., dtype=torch.bfloat16)
    for row in range(rows):
        selected = (boundaries + row * 17) % 248320
        logits[row, selected] = torch.arange(1, 17).bfloat16()
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, choices=(8, 32), default=8)
    options = parser.parse_args()
    require_projection_environment(os.environ, False)
    import torch
    import ttnn

    report = dict(passed=False, scope=__doc__, rows=options.rows, checks=[], hashes={name:
        hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('draft-head-probe.py', 'draft_shared_head.py')})
    owned, mesh = [], None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        mesh.enable_program_cache()
        fixture = candidate_fixture(options.rows)
        expected_scores, expected_tokens = fixture.float().topk(16, dim=-1)
        packed = torch.cat([part.reshape(1, 1, options.rows, 124160) for part in fixture.chunk(2, dim=-1)], dim=0)
        logits = ttnn.from_torch(packed, device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
        owned.append(logits)
        chunks = local_head_candidates(ttnn, logits, owned)
        records = []
        for chunk in chunks:
            values = ttnn.get_device_tensors(chunk['values'])
            indices = ttnn.get_device_tensors(chunk['indices'])
            if len(values) != 2 or len(indices) != 2:
                raise AssertionError('Both chip-local candidate shards required')
            for chip in range(2):
                record = dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                    values=ttnn.to_torch(values[chip]).reshape(options.rows, 16).float(),
                    indices=ttnn.to_torch(indices[chip]).reshape(options.rows, 16).long())
                selected = record['indices']
                if torch.any(selected < 0) or torch.any(selected >= chunk['stop'] - chunk['start']):
                    raise AssertionError('Candidate escaped its unpadded vocabulary chunk')
                local = fixture[:, chip * 124160 + chunk['start']:chip * 124160 + chunk['stop']].float()
                if not torch.equal(record['values'], local.gather(-1, selected)):
                    raise AssertionError('Returned score does not match its local token index')
                if not torch.equal(record['values'], local.topk(16, dim=-1).values):
                    raise AssertionError('Local top16 scores differ')
                records.append(record)
                report['checks'].append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'], exact=True))
        tokens, scores = merge_chunk_candidates(records, block_rows=options.rows)
        if (not torch.equal(tokens, expected_tokens[None, 1:options.rows])
                or not torch.equal(scores, expected_scores[None, 1:options.rows])):
            raise AssertionError('Global proposal IDs or scores differ')
        report['global_ids_exact'] = True
        report['proposal_tokens'] = tokens.tolist()
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if mesh is not None:
            release_owned(ttnn, owned)
            ttnn.close_mesh_device(mesh)
        options.output.write_text(json.dumps(report, indent=2))
    report['passed'] = True
    options.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
