"""Fetch only pinned layer-zero DFlash2 SwiGLU projection weights."""

import argparse
import json
from pathlib import Path

from draft_convolution_fixture import fetch as fetch_subset


TENSORS = {
    'layers.0.mlp.down_proj.weight': ([5120, 17408], 'down-projection.bf16'),
    'layers.0.mlp.gate_proj.weight': ([17408, 5120], 'gate-projection.bf16'),
    'layers.0.mlp.up_proj.weight': ([17408, 5120], 'up-projection.bf16'),
}


def fetch(output):
    return fetch_subset(output, specifications=TENSORS, scope=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(fetch(parser.parse_args().output), indent=2))
