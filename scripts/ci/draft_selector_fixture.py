"""Stage pinned DFlash2 selector and final norm tensors for hash audit, without remote code."""

import argparse
import json
from pathlib import Path

from draft_convolution_fixture import fetch as fetch_subset


TENSORS = {
    'candidate_selector.hidden_projection.weight': ([256, 5120], 'hidden-projection.bf16'),
    'candidate_selector.predecessor_codebook': ([248320, 256], 'predecessor.bf16'),
    'candidate_selector.successor_codebook': ([248320, 256], 'successor.bf16'),
    'norm.weight': ([5120], 'norm.bf16'),
}


def fetch(output):
    return fetch_subset(output, specifications=TENSORS, scope=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(fetch(options.output), indent=2))
