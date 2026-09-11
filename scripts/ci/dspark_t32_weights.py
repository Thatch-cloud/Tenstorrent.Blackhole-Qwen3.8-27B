"""Read only the two byte-pinned Markov matrices from an existing checkpoint."""

import hashlib
import json
from pathlib import Path

from dspark_markov_fixture import TENSORS, expected_manifest
from dspark_intake import HEADER_BYTES, HEADER_SHA256, CHECKPOINT_BYTES


def load_checkpoint(path):
    import torch
    matrices = []
    if Path(path).stat().st_size != CHECKPOINT_BYTES:
        raise ValueError('Pinned checkpoint extent required')
    with Path(path).open('rb') as checkpoint:
        if int.from_bytes(checkpoint.read(8), 'little') != HEADER_BYTES:
            raise ValueError('Pinned checkpoint header extent required')
        header_bytes = checkpoint.read(HEADER_BYTES)
        if hashlib.sha256(header_bytes).hexdigest() != HEADER_SHA256:
            raise ValueError('Pinned checkpoint metadata required')
        header = json.loads(header_bytes)
        for name, expected in TENSORS.items():
            entry = header[name]
            start, end = entry['data_offsets']
            if (entry['dtype'] != 'BF16' or entry['shape'] != expected['shape']
                    or end - start != expected['bytes'] or not 0 <= start < end <= CHECKPOINT_BYTES - 8 - HEADER_BYTES):
                raise ValueError('Complete pinned Markov matrix shape and dtype required')
            checkpoint.seek(8 + HEADER_BYTES + start)
            data = checkpoint.read(expected['bytes'])
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('Checkpoint Markov bytes differ from the learned fixture')
            matrix = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(expected['shape'])
            if not torch.isfinite(matrix).all():
                raise ValueError('Finite learned Markov weights required')
            matrices.append(matrix)
    return expected_manifest(), *matrices
