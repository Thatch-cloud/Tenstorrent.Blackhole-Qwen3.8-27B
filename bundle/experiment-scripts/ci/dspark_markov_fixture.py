"""Hash-pinned full DSpark Markov matrices; no backbone or executable checkpoint code."""

import argparse
import hashlib
import json
from pathlib import Path

from draft_projection_fixture import read_range
from dspark_intake import MODEL, REVISION, HEADER_BYTES, HEADER_SHA256, CHECKPOINT_BYTES, validate_header


TENSORS = {
    'markov_head.markov_w1.weight': dict(file='predecessor.bf16', shape=[248320, 256], dtype='BF16', bytes=127139840,
        sha256='80cdfb6122576b2cf02a3df40e83209fd5514aed82ef381d6c551d6c925d3c57'),
    'markov_head.markov_w2.weight': dict(file='successor.bf16', shape=[248320, 256], dtype='BF16', bytes=127139840,
        sha256='39ba4adba9d613e61ba2ff4ef4151fdfda7ac898dab12e4603d3cff6a5e4d2e2'),
}


def expected_manifest():
    return dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
        checkpoint_bytes=CHECKPOINT_BYTES, tensors=TENSORS)


def load_fixture(output):
    import torch

    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest != expected_manifest():
        raise ValueError('Exact pinned Markov fixture manifest required')
    matrices = []
    for entry in TENSORS.values():
        path = output / entry['file']
        if path.stat().st_size != entry['bytes'] or path.with_suffix('.bf16.partial').exists():
            raise ValueError('Complete learned matrix required')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Pinned learned Markov bytes differ')
        matrix = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(entry['shape'])
        if not torch.isfinite(matrix).all():
            raise ValueError('Finite learned Markov matrix required')
        if hashlib.sha256(memoryview(matrix.view(torch.uint8).numpy())).hexdigest() != entry['sha256']:
            raise ValueError('Markov tensor conversion changed learned bytes')
        matrices.append(matrix)
    return manifest, *matrices


def fetch(output):
    if output.exists():
        raise ValueError('Fresh fixture directory required; existing data is never replaced')
    url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors'
    header_data, total = read_range(url, 8, HEADER_BYTES)
    if total != CHECKPOINT_BYTES or hashlib.sha256(header_data).hexdigest() != HEADER_SHA256:
        raise ValueError('Pinned learned checkpoint header required')
    header = json.loads(header_data)
    validate_header(header, HEADER_BYTES, total)
    output.mkdir(parents=True, exist_ok=False)
    for name, entry in TENSORS.items():
        start, stop = header[name]['data_offsets']
        if stop - start != entry['bytes'] or header[name]['shape'] != entry['shape']:
            raise ValueError('Learned Markov extent differs from pinned inventory')
        path = output / entry['file']
        partial = path.with_suffix('.bf16.partial')
        digest = hashlib.sha256()
        with partial.open('xb') as destination:
            for offset in range(0, entry['bytes'], 2097152):
                count = min(2097152, entry['bytes'] - offset)
                data, observed_total = read_range(url, 8 + HEADER_BYTES + start + offset, count)
                if observed_total != total or len(data) != count:
                    raise ValueError('Learned checkpoint range changed')
                destination.write(data)
                digest.update(data)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('Downloaded Markov matrix differs from audited content')
        partial.rename(path)
    manifest = expected_manifest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-verified', action='store_true')
    options = parser.parse_args()
    print(json.dumps(load_fixture(options.output)[0] if options.reuse_verified else fetch(options.output), indent=2))
