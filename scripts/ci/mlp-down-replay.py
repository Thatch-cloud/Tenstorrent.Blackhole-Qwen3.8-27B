"""Replay a local learned MLP failure against two explicit fidelity schedules."""

import argparse
import json
from pathlib import Path

import torch

from draft_mlp_fixture import load_mlp
from draft_remaining_layers_fixture import load_layer
from draft_mlp import split_mlp_weights
from projection_rounding import grouped_projection_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    captured = torch.load(options.capture, map_location='cpu', weights_only=True)
    layer = captured.get('layer', 0)
    if type(layer) is not int or layer not in range(5):
        raise ValueError('Pinned learned layer index required')
    manifest, weights = load_mlp(options.fixture) if layer == 0 else load_layer(options.fixture, layer)
    if captured['checkpoint'] != manifest or captured['chip'] not in (0, 1):
        raise ValueError('Capture must match the verified checkpoint and TP2 rank')
    ranks = split_mlp_weights(*(weights[f'layers.{layer}.mlp.{name}_proj.weight'] for name in ('gate', 'up', 'down')))
    weight = ranks[captured['chip']][2]
    results = []
    for span in (16, 32):
        expected = grouped_projection_reference(captured['activation'], weight, destination_rounding=True, fidelity_span=span)
        actual = captured['actual']
        if span == captured.get('fidelity_span', 16) and not torch.equal(expected, captured['reference']):
            raise AssertionError('Original captured reference must reproduce exactly')
        results.append(dict(fidelity_span=span, exact=torch.equal(actual, expected),
            max_error=float((actual - expected).abs().max()),
            mismatched=int((~torch.isclose(actual, expected, rtol=1e-4, atol=1e-4)).sum())))
        print(json.dumps(results[-1]), flush=True)
        options.output.write_text(json.dumps(dict(capture=str(options.capture), results=results), indent=2))


if __name__ == '__main__':
    main()
