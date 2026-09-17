"""Fetch only pinned layer-zero draft Q/K/V/O projections and Q/K normalization."""

import argparse
import json
from pathlib import Path

from draft_convolution_fixture import fetch as fetch_subset, verified_bytes, verified_tensor


TENSORS = {
    'layers.0.self_attn.k_norm.weight': ([128], 'k-norm.bf16'),
    'layers.0.self_attn.k_proj.weight': ([1024, 5120], 'k-projection.bf16'),
    'layers.0.self_attn.o_proj.weight': ([5120, 4096], 'o-projection.bf16'),
    'layers.0.self_attn.q_norm.weight': ([128], 'q-norm.bf16'),
    'layers.0.self_attn.q_proj.weight': ([4096, 5120], 'q-projection.bf16'),
    'layers.0.self_attn.v_proj.weight': ([1024, 5120], 'v-projection.bf16'),
}

TENSOR_SHA256 = dict(zip(TENSORS, (
    '48f9dc189ece3153d140423af7378195160a308137d02d3b708437dbe9a94f51',
    'c6528b4c0ba06f65eb9a0d1ec5935673ddf35cc3c493a34b1809e0aeb5a0df10',
    '19779ddf18736b9573bc7325f5563025353c0c236bce5197025cc14270251fe6',
    '585f2b3355717c45da8aec3b24fa74c02365840d7aeab210778042dfb99a4ff5',
    '781aa614b175728cb277b1d490ec097b80a36e1fcc291256f6e1c15a54a63ec5',
    '83c10afd7011659b588875d723a82a6dc547b4a109f5e90eee5ef068dc28653f',
), strict=True))


def load_attention(output):
    manifest, data = verified_bytes(output, specifications=TENSORS, hashes=TENSOR_SHA256)
    tensors = {}
    for name, (shape, filename) in TENSORS.items():
        tensors[name] = verified_tensor(data[name], shape, TENSOR_SHA256[name], name)
    return manifest, tensors


def fetch(output):
    return fetch_subset(output, specifications=TENSORS, scope=__doc__)


def ensure_fixture(output):
    if not (output / 'manifest.json').exists():
        fetch(output)
    return verified_bytes(output, specifications=TENSORS, hashes=TENSOR_SHA256)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-verified', action='store_true')
    options = parser.parse_args()
    print(json.dumps(ensure_fixture(options.output) if options.reuse_verified else fetch(options.output), indent=2))
