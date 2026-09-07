"""Stage DFlash2 layers1-4 for hash audit; no tensor loading or checkpoint code execution."""

import argparse
import json
from pathlib import Path

from draft_attention_fixture import TENSORS as ATTENTION
from draft_convolution_fixture import TENSORS as CONVOLUTION, fetch as fetch_subset
from draft_mlp_fixture import TENSORS as MLP


def specifications(layer):
    if type(layer) is not int or layer not in (1, 2, 3, 4):
        raise ValueError('Only remaining learned layers1-4 may be staged')
    return {name.replace('layers.0.', f'layers.{layer}.', 1): (list(shape), filename)
        for name, (shape, filename) in (ATTENTION | CONVOLUTION | MLP).items()}


def fetch_layers(output, layers):
    layers = tuple(layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError('A nonempty unique list of layer indices is required')
    selections = {layer: specifications(layer) for layer in layers}
    return {layer: fetch_subset(output / f'layer-{layer}', specifications=selected, scope=__doc__)
        for layer, selected in selections.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', choices=(1, 2, 3, 4), default=(1, 2, 3, 4))
    options = parser.parse_args()
    print(json.dumps(fetch_layers(options.output, options.layers), indent=2))
