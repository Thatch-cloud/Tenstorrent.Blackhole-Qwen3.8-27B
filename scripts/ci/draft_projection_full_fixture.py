"""Fetch pinned projection and normalization tensors without checkpoint code."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

from draft_projection_fixture import MODEL, REVISION, URL, read_range


HEADER_SHA256 = '0c2c70601b30f8d1ca7d5794b817779ba2dcf1956cfc7d4f83e87091e1ab7c8c'
TENSORS = {'fc.weight': ([5120, 25600], 'fc.bf16'), 'hidden_norm.weight': ([5120], 'hidden_norm.bf16')}
TENSOR_SHA256 = {
    'fc.weight': '3ccef5ab503ff30aa589ee0c528f462a37c544ef766fa943c5b622aa92d4b2d5',
    'hidden_norm.weight': '998ae0570db09ee6b5549d9a35296a2ba8e4adfa5898135534646dede1a91693',
}


def load_projection(output):
    import torch

    manifest = json.loads((output / 'manifest.json').read_text())
    if (manifest['model'] != MODEL or manifest['revision'] != REVISION or manifest['header_sha256'] != HEADER_SHA256):
        raise ValueError('Pinned full projection manifest required')
    tensors = {}
    for tensor, (shape, filename) in TENSORS.items():
        entry = manifest['tensors'][tensor]
        data = (output / filename).read_bytes()
        elements = 1
        for dimension in shape:
            elements *= dimension
        if (entry['file'] != filename or entry['shape'] != shape or entry['dtype'] != 'BF16'
                or entry['bytes'] != 2 * elements or len(data) != 2 * elements
                or entry['sha256'] != TENSOR_SHA256[tensor]
                or hashlib.sha256(data).hexdigest() != entry['sha256']):
            raise ValueError('Complete hashed BF16 projection tensors required')
        if tensor == 'fc.weight' and hashlib.sha256(data[:32 * 25600 * 2]).hexdigest() != '882b3405c0eb1e9bc0d502e0ff241e43c25f405c8e10c4ba469129008f161a33':
            raise ValueError('Full projection differs from audited first32 outputs')
        tensors[tensor] = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape)
        if not torch.isfinite(tensors[tensor]).all():
            raise ValueError('Finite learned tensors required')
    return manifest, tensors['fc.weight'], tensors['hidden_norm.weight']


def stream_tensor(output, start, length, total, *, chunk_bytes=2097152):
    if (any(type(value) is not int for value in (start, length, total, chunk_bytes))
            or start < 0 or length < 1 or start + length > total or not 1 <= chunk_bytes <= 2097152):
        raise ValueError('Bounded tensor byte geometry required')
    partial = output.with_suffix(output.suffix + '.partial')
    if output.exists() or partial.exists():
        raise ValueError('Refusing to overwrite a tensor or incomplete download')
    digest = hashlib.sha256()
    with partial.open('xb') as destination:
        for offset in range(0, length, chunk_bytes):
            count = min(chunk_bytes, length - offset)
            data, observed_total = read_range(URL, start + offset, count)
            if observed_total != total or len(data) != count:
                raise ValueError('Checkpoint changed or range length is inconsistent')
            destination.write(data)
            digest.update(data)
    partial.rename(output)
    return dict(file=output.name, bytes=length, sha256=digest.hexdigest())


def fetch(output):
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for shape, name in TENSORS.values()] + [output / 'manifest.json']
    if any(path.exists() or path.with_suffix(path.suffix + '.partial').exists() for path in targets):
        raise ValueError('Fresh fixture directory required')
    size_bytes, total = read_range(URL, 0, 8)
    header_size = struct.unpack('<Q', size_bytes)[0]
    header_bytes, header_total = read_range(URL, 8, header_size)
    if header_total != total or hashlib.sha256(header_bytes).hexdigest() != HEADER_SHA256:
        raise ValueError('Audited checkpoint header required')
    header = json.loads(header_bytes)
    selected = {}
    for tensor, (shape, filename) in TENSORS.items():
        entry = header[tensor]
        length = 2
        for dimension in shape:
            length *= dimension
        offsets = entry['data_offsets']
        if (entry['dtype'] != 'BF16' or entry['shape'] != shape or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != length
                or 8 + header_size + offsets[1] > total):
            raise ValueError('Audited tensor geometry required')
        selected[tensor] = (filename, 8 + header_size + offsets[0], length, shape)
    manifest = dict(model=MODEL, revision=REVISION, checkpoint_bytes=total,
        header_sha256=HEADER_SHA256, fetched_bytes=8 + header_size, tensors={},
        scope='Full learned fc and hidden_norm only; no draft layers, embeddings or executable checkpoint code')
    for tensor, (filename, start, length, shape) in selected.items():
        print(f'Fetching {tensor}: {length} bytes in bounded ranges', flush=True)
        manifest['tensors'][tensor] = dict(stream_tensor(output / filename, start, length, total), dtype='BF16', shape=shape)
        manifest['fetched_bytes'] += length
    with (output / 'manifest.json').open('x') as destination:
        json.dump(manifest, destination, indent=2)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(fetch(parser.parse_args().output), indent=2))
