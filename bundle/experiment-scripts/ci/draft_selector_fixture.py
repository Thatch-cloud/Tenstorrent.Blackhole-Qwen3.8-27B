"""Stage pinned DFlash2 selector and final norm tensors for hash audit, without remote code."""

import argparse
import json
from pathlib import Path

from draft_convolution_fixture import fetch as fetch_subset, verified_bytes, verified_tensor


TENSORS = {
    'candidate_selector.hidden_projection.weight': ([256, 5120], 'hidden-projection.bf16'),
    'candidate_selector.predecessor_codebook': ([248320, 256], 'predecessor.bf16'),
    'candidate_selector.successor_codebook': ([248320, 256], 'successor.bf16'),
    'norm.weight': ([5120], 'norm.bf16'),
}

TENSOR_SHA256 = {
    'candidate_selector.hidden_projection.weight': '458127a477af64ce695ea4519bb7040d0776740eecc2a74dd63106d2960fe89e',
    'candidate_selector.predecessor_codebook': 'c312a8351a6be74e9ddeb944aad7288c4f9025cd9b2224379063d5ab5427813b',
    'candidate_selector.successor_codebook': '783e22212a2eb510e27ff007c6967f3f6f947c2e007d76ff7d1f57e3fcac6b92',
    'norm.weight': '4718610fff45fc79bc9b967b79f305f07fb73df860fe3feb5bb0e18c891c7e5f',
}


def load_selector(output):
    manifest, data = verified_bytes(output, specifications=TENSORS, hashes=TENSOR_SHA256)
    tensors = {}
    for name, (shape, filename) in TENSORS.items():
        tensors[name] = verified_tensor(data[name], shape, TENSOR_SHA256[name], name)
    return manifest, tensors


def ensure_fixture(output):
    if not (output / 'manifest.json').exists():
        fetch(output)
    return verified_bytes(output, specifications=TENSORS, hashes=TENSOR_SHA256)[0]


def fetch(output):
    return fetch_subset(output, specifications=TENSORS, scope=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-verified', action='store_true')
    options = parser.parse_args()
    print(json.dumps(ensure_fixture(options.output) if options.reuse_verified else fetch(options.output), indent=2))
