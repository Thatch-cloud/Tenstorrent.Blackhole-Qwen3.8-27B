"""Replay a local learned MLP failure against two explicit fidelity schedules."""

import argparse
import json
from pathlib import Path

import torch

from draft_mlp_fixture import load_mlp
from draft_remaining_layers_fixture import load_layer
from draft_mlp import split_mlp_weights
from projection_rounding import grouped_projection_reference


def select_failure_columns(captured, weight):
    actual, reference = captured['actual'], captured['reference']
    activation = captured['activation']
    if (actual.shape != reference.shape or actual.shape[:-1] != activation.shape[:-1]
            or weight.shape != (activation.shape[-1], actual.shape[-1])
            or not all(torch.isfinite(value).all() for value in (actual, reference, activation, weight))):
        raise ValueError('Finite shape-matched captured projection required')
    failed = ~torch.isclose(actual, reference, rtol=1e-4, atol=1e-4)
    columns = failed.reshape(-1, actual.shape[-1]).any(dim=0).nonzero().flatten()
    if columns.numel() == 0:
        raise ValueError('No failing output columns in capture')
    return columns, weight[:, columns].contiguous(), actual[..., columns], reference[..., columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mismatch-columns', action='store_true')
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
    actual, original = captured['actual'], captured['reference']
    columns = None
    if options.mismatch_columns:
        columns, weight, actual, original = select_failure_columns(captured, weight)
    diagnostic = dict(capture=str(options.capture),
        scope='Failing output columns only' if options.mismatch_columns else 'All captured output columns',
        columns=None if columns is None else columns.tolist(), results=[])
    if columns is not None:
        diagnostic['actual'] = actual.tolist()
        diagnostic['captured_reference'] = original.tolist()
        diagnostic['fp64_matmul'] = (captured['activation'].double() @ weight.double()).tolist()
    results = []
    for span in (16, 32):
        expected = grouped_projection_reference(captured['activation'], weight, destination_rounding=True, fidelity_span=span)
        if span == captured.get('fidelity_span', 16) and not torch.equal(expected, original):
            raise AssertionError('Original captured reference must reproduce exactly')
        results.append(dict(fidelity_span=span, exact=torch.equal(actual, expected),
            max_error=float((actual - expected).abs().max()),
            mismatched=int((~torch.isclose(actual, expected, rtol=1e-4, atol=1e-4)).sum())))
        print(json.dumps(results[-1]), flush=True)
        diagnostic['results'] = results
        options.output.write_text(json.dumps(diagnostic, indent=2))


if __name__ == '__main__':
    main()
