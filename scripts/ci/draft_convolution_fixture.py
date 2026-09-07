"""Fetch only layer-zero DFlash2 convolution and input normalization weights."""

import argparse
import hashlib
import json
from pathlib import Path

from draft_projection_fixture import MODEL, REVISION, URL, read_range
from draft_projection_full_fixture import HEADER_SHA256, stream_tensor


TENSORS = {
    'layers.0.attention_conv.base_kernel': ([2, 2, 5120], 'attention-base.bf16'),
    'layers.0.attention_conv.kernel_projection.weight': ([1280, 5120], 'attention-projection.bf16'),
    'layers.0.input_layernorm.weight': ([5120], 'attention-norm.bf16'),
    'layers.0.mlp_conv.base_kernel': ([2, 2, 5120], 'mlp-base.bf16'),
    'layers.0.mlp_conv.kernel_projection.weight': ([1280, 5120], 'mlp-projection.bf16'),
    'layers.0.post_attention_layernorm.weight': ([5120], 'mlp-norm.bf16'),
}

TENSOR_SHA256 = dict(zip(TENSORS, (
    '03b6ba4f1ca1f6a91677e46dead2b1fd043ed744ad7fb20c3df9f8408faf5e6a',
    '9a7edb2897de3c7a6fe49b5db4f67e9dfc8a49ccc1526fe87f363337bd998569',
    '79ffd1e01a2b614c3ce7a4e5b9a2d9ee844090620330b429d5b1172c19fd3bd5',
    'e39a1916a16009c2f82c7c9260dcd63f8428b48ed117a7eae41e054b88a4dccf',
    '0eafc9ebdc9ac03f50b53f6d853168c5622df80419c61005ee417a6345ffb5cb',
    '2ff54ab16f8b1c72a6894bd824fd9bf0815803b294ae56d962824788cb548af2',
), strict=True))


def verified_bytes(output, *, specifications=None, hashes=None):
    specifications = TENSORS if specifications is None else specifications
    hashes = TENSOR_SHA256 if hashes is None else hashes
    manifest = json.loads((output / 'manifest.json').read_text())
    if (manifest['model'] != MODEL or manifest['revision'] != REVISION
            or manifest['header_sha256'] != HEADER_SHA256 or manifest['checkpoint_bytes'] != 3848817896):
        raise ValueError('Pinned convolution manifest required')
    tensors = {}
    for name, (shape, filename) in specifications.items():
        entry = manifest['tensors'][name]
        length = 2
        for dimension in shape:
            length *= dimension
        if (entry['file'] != filename or entry['shape'] != shape or entry['dtype'] != 'BF16'
                or entry['bytes'] != length or entry['sha256'] != hashes[name]
                or (output / filename).stat().st_size != length):
            raise ValueError('Audited convolution metadata required')
        data = (output / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes[name]:
            raise ValueError('Audited convolution content required')
        tensors[name] = data
    return manifest, tensors


def load_convolution(output):
    import torch

    manifest, data = verified_bytes(output)
    tensors = {}
    for name, (shape, filename) in TENSORS.items():
        tensors[name] = torch.frombuffer(bytearray(data[name]), dtype=torch.bfloat16).reshape(shape)
        if not torch.isfinite(tensors[name]).all():
            raise ValueError('Finite learned convolution weights required')
    return manifest, tensors


def ensure_fixture(output):
    if not (output / 'manifest.json').exists():
        fetch(output)
    return verified_bytes(output)[0]


def fetch(output, *, specifications=None, scope=__doc__):
    specifications = TENSORS if specifications is None else specifications
    header_bytes, total = read_range(URL, 8, 8928)
    if total != 3848817896 or hashlib.sha256(header_bytes).hexdigest() != HEADER_SHA256:
        raise ValueError('Pinned audited checkpoint header required')
    header = json.loads(header_bytes)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Empty fixture directory required; existing data is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=MODEL, revision=REVISION, header_sha256=HEADER_SHA256,
        checkpoint_bytes=total, tensors={}, scope=scope)
    for tensor, (shape, filename) in specifications.items():
        entry = header[tensor]
        length = 2
        for dimension in shape:
            length *= dimension
        offsets = entry['data_offsets']
        if (entry['dtype'] != 'BF16' or entry['shape'] != shape or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != length or 8936 + offsets[1] > total):
            raise ValueError('Pinned convolution tensor geometry required')
        manifest['tensors'][tensor] = dict(stream_tensor(output / filename, 8936 + offsets[0], length, total),
            shape=shape, dtype='BF16')
    with (output / 'manifest.json').open('x') as destination:
        json.dump(manifest, destination, indent=2)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-verified', action='store_true')
    options = parser.parse_args()
    print(json.dumps(ensure_fixture(options.output) if options.reuse_verified else fetch(options.output), indent=2))
