"""Fetch only pinned layer-zero DFlash2 SwiGLU projection weights."""

import argparse
import json
from pathlib import Path

from draft_convolution_fixture import fetch as fetch_subset, verified_bytes, verified_tensor


TENSORS = {
    'layers.0.mlp.down_proj.weight': ([5120, 17408], 'down-projection.bf16'),
    'layers.0.mlp.gate_proj.weight': ([17408, 5120], 'gate-projection.bf16'),
    'layers.0.mlp.up_proj.weight': ([17408, 5120], 'up-projection.bf16'),
}


TENSOR_SHA256 = dict(zip(TENSORS, (
    '5cad30340445d399e3f6a12d0b83a8e40aff31cc42c4535ba34ab53d474f8242',
    'a5555f4d8de040e7cc8141b8b41dfc4f3a9476ddecc95879a7d3f7c6f3545bbd',
    'c9c75bdf71b82f6e7e72b84248a26dbac237f02894dc5e19463dcaf88e781f02',
), strict=True))


def load_mlp(output):
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
