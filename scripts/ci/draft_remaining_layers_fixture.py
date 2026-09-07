"""Stage DFlash2 layers1-4 and load only hash-audited tensors; no checkpoint code execution."""

import argparse
import json
from pathlib import Path

from draft_attention_fixture import TENSORS as ATTENTION
from draft_convolution_fixture import TENSORS as CONVOLUTION, fetch as fetch_subset, verified_bytes
from draft_mlp_fixture import TENSORS as MLP

TENSOR_SHA256 = {
    "1": {
        "layers.1.self_attn.k_norm.weight": "0e2a70cd25dcdeb9e2e3b05bded8b1b599c09a76d3726833e147663c23eefac1",
        "layers.1.self_attn.k_proj.weight": "9b0bbb5877f857862e28dbdfeb4f62c1c1705e735c053e97a2e5974f4617828c",
        "layers.1.self_attn.o_proj.weight": "b1fadca9a20fb36565e4184de559950cdb80a01b29628e5b7efb4d58c13690ca",
        "layers.1.self_attn.q_norm.weight": "a6df4723df929225a2687a92df61b2b7a6766574f93708f4d6488ae668817a60",
        "layers.1.self_attn.q_proj.weight": "505b1c7eb8eb41a819ef2cf893c077be2369d4fe2de2456e03385dbb847e887b",
        "layers.1.self_attn.v_proj.weight": "85dfa6317fe3dd247b7ab82429aea9c51bf03501612c3b5ff8da929a8f984fc4",
        "layers.1.attention_conv.base_kernel": "3937a02df01c6cc4857194f35e7c71e3a2aaedb7cf2fbcd2bfade22ea61956d7",
        "layers.1.attention_conv.kernel_projection.weight": "f91fabe8a23014d0d4640c0ce5671eb8b7481d055d446b4a89e0052ab95d5f08",
        "layers.1.input_layernorm.weight": "12c8d725f1e0922ac66c8b2ef34f7b57060c5fd357d7363910e03092f56a79c0",
        "layers.1.mlp_conv.base_kernel": "73e4d4a455f141bfd5a3733058486455288e49aeb52e16a14f5be9f016b5fecd",
        "layers.1.mlp_conv.kernel_projection.weight": "087c7b28d3741f000b4c4728b383c389c72e370f40484b20137f7ff74af51b4e",
        "layers.1.post_attention_layernorm.weight": "5fbd9d0e35b2c7cde8e8931e818d055e9515abcf44f1a72f1df2135a736465d0",
        "layers.1.mlp.down_proj.weight": "3fc84cd7c3b7c037e7d9fab1a1e4da15bb7220577a197fd8ee66216ded8bb401",
        "layers.1.mlp.gate_proj.weight": "a27f71e8401e32c989c997d9134bf6ac2c680d66bce38e9e1ec39a02a9b9dba8",
        "layers.1.mlp.up_proj.weight": "6aad2cc7183d575b45ba08217efd9c7993e14e38e4f7a074b5487599554e146f"
    }
}


def load_layer(output, layer):
    import torch

    selected = specifications(layer)
    if str(layer) not in TENSOR_SHA256:
        raise ValueError('Layer tensor hashes have not been audited')
    manifest, data = verified_bytes(output, specifications=selected, hashes=TENSOR_SHA256[str(layer)])
    tensors = {}
    for name, (shape, filename) in selected.items():
        value = torch.frombuffer(bytearray(data[name]), dtype=torch.bfloat16).reshape(shape)
        if not torch.isfinite(value).all():
            raise ValueError('Finite learned layer tensors required')
        tensors[name] = value
    return manifest, tensors


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
