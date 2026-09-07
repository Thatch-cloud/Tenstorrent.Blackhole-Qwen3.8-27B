"""Stage DFlash2 layers1-4 and load only hash-audited tensors; no checkpoint code execution."""

import argparse
import json
from pathlib import Path

from draft_attention_fixture import TENSORS as ATTENTION
from draft_convolution_fixture import TENSORS as CONVOLUTION, fetch as fetch_subset, verified_bytes
from draft_mlp_fixture import TENSORS as MLP

TENSOR_SHA256 = {
    "4": {
        "layers.4.self_attn.k_norm.weight": "04396c72894be9c8bf6f189bf377b83661d9e3c412af46d7c345e920415e8589",
        "layers.4.self_attn.k_proj.weight": "3d4be044d7c4262d826098dec2ed2f082b9ee36abe2a447ccfa9fd1968bc1e0b",
        "layers.4.self_attn.o_proj.weight": "c4594dc14ed236143b40e6c9776d031dd22034e691b6dc969278b9462e2548aa",
        "layers.4.self_attn.q_norm.weight": "bcf7be92967c66eb91b2b601bd62004fb443e6235f5c5d9257a9e2d0a0899ef2",
        "layers.4.self_attn.q_proj.weight": "9ae94295d798299ea69c21b87a333d52947a991e8ee006c8a7473350b61e7855",
        "layers.4.self_attn.v_proj.weight": "734e800779ef95cfbfc3b2d073a92c7d69155729194c473137ee715a7ba1ccff",
        "layers.4.attention_conv.base_kernel": "5dcc8ff1ea18226dd73adba8595004e0b158924e5f152d5c7d946d05b7bde5be",
        "layers.4.attention_conv.kernel_projection.weight": "c55856808dd80fc904582f01cb95ded9e84cb6217121974a54178be328b6e395",
        "layers.4.input_layernorm.weight": "3eda34484590995e51443e4c17615a65d6ddd5c98312a77281823ec96dfbb9e3",
        "layers.4.mlp_conv.base_kernel": "5446c824be996cffed97e0b0aeb63795e298de94c20a0feb14fec19d719c577a",
        "layers.4.mlp_conv.kernel_projection.weight": "b19ba28ead45ff3f452545236010d69b9e42a46daa7e705384f164fbd70dcf71",
        "layers.4.post_attention_layernorm.weight": "06560a4965ba4eed3b871dc87fe69ceb19fae1df46e0e503e2f6959a8313d036",
        "layers.4.mlp.down_proj.weight": "d82ac25fd4564ae7f81ea1c0e9747e729e07918ca0a33ec30c3fc1a591a19727",
        "layers.4.mlp.gate_proj.weight": "070eaa99c70fc4afd99eb2aa16dec111eabf12db8318db225236a604a12463ed",
        "layers.4.mlp.up_proj.weight": "1b7d8f81e407fbf188bb31cfd27e01f7fa8461cc2ebf2d56594f1c214f5a043a"
    },
    "3": {
        "layers.3.self_attn.k_norm.weight": "4fa4496a9505b1a05ec10760dfeedcb551ba8a4ef5131b0b2a60c717038be1c1",
        "layers.3.self_attn.k_proj.weight": "484f44745b097cf2f926b68643b794591a17ea4221a17ccac6c6890f9230599d",
        "layers.3.self_attn.o_proj.weight": "298082223ab6e04aed2f5e684492016e878a6e3a9005465392b73e405594166b",
        "layers.3.self_attn.q_norm.weight": "44e76c602b3ab0061775ccb948ac572e3436429a0870f9aff4c46a92fb4f3ed3",
        "layers.3.self_attn.q_proj.weight": "388cb8174c2be8de5c33a870229d1989d304edea38f830b0f5a13299c3fca544",
        "layers.3.self_attn.v_proj.weight": "477f444bc083feef7024ae4700a787163c93663f479f4b94fd82e901e9b2a508",
        "layers.3.attention_conv.base_kernel": "8603058c019f4828246ade8d76d1714a6e7d2e5e030120ef91f1e2e97d357347",
        "layers.3.attention_conv.kernel_projection.weight": "73338963cdcc0c131d2d0f2c108d071f7beeb75ec650dc514ad09f9b0216a62e",
        "layers.3.input_layernorm.weight": "b5a2bd5b747665b4ae5fa81327bbea5f3ab4725cc004b29254c8f5687dda6a8e",
        "layers.3.mlp_conv.base_kernel": "af60ac3b5c08dd5f6c6e7a200d28c4fce905e38c1673f80698ca3a4586993204",
        "layers.3.mlp_conv.kernel_projection.weight": "897e21db07a8771ff5225ec15d36e62a7fad03b9f198b9bc74a8c430c160d5cb",
        "layers.3.post_attention_layernorm.weight": "a5d7481e3e2386642660afe3e7dee0da98b2d597b8e52f165265b36fb3019e4b",
        "layers.3.mlp.down_proj.weight": "c7438d31dbbb0240a4cb4a6ab642ed8f6e8354b29167d258b63bbc6bed6a4a92",
        "layers.3.mlp.gate_proj.weight": "27e00617e2a68014d8a6a9af5044050e264804a7caa4d5f9868723aaf57c88f6",
        "layers.3.mlp.up_proj.weight": "f38d436fb1ce3775d53e3eefcff281ef8de2ac91019e503eef3a88e04daaf16c"
    },
    "2": {
        "layers.2.self_attn.k_norm.weight": "20e8572d4363420f61c2a4dd4180fc021a8e848ef6e8d2ef9a4308824f53e183",
        "layers.2.self_attn.k_proj.weight": "905943a8428b8948293b22f496b918b20746e481f267686cac7d7e6d0c583227",
        "layers.2.self_attn.o_proj.weight": "3622416239dcf25b6242157e270d8ed8ce6b83a3a11aa98b6938748211fe71f7",
        "layers.2.self_attn.q_norm.weight": "7c5f569d8fbdda56acc18606a4c5fa3e2493fd8c7e20a852b64da199b190ae61",
        "layers.2.self_attn.q_proj.weight": "cb578c02c5c30e975fc7bfd1b55d2f2b982b9a014c2793404dab74494526180b",
        "layers.2.self_attn.v_proj.weight": "869c0a5dbaf78c66decf9fa1237b3bf7664f10b78941ad604daadc1c705d9848",
        "layers.2.attention_conv.base_kernel": "c371e1896e88395902cc68de5af17541ec282ebe21847d96d41c671c6d5a42de",
        "layers.2.attention_conv.kernel_projection.weight": "44aa936b3c3d3dda19d7b88ab2cfc898070c7cf1e1120c766aaf584ad252bb58",
        "layers.2.input_layernorm.weight": "63c9b93004c8211b941c89291d600b4d1da580a1b6a2db8f07f352143f3840bf",
        "layers.2.mlp_conv.base_kernel": "0e69560af3eaf128b094c6d655fbd84cc19ba0fa3786ce66298d5f81be395de9",
        "layers.2.mlp_conv.kernel_projection.weight": "a3a1d47ab0a4b58e828b490719a0f152bcdd11d555b4befe766af471a18d3f05",
        "layers.2.post_attention_layernorm.weight": "0aac61f7dee4e95ab70b5e116d75461a31981b71980281cc73e11322ec944810",
        "layers.2.mlp.down_proj.weight": "b8db0cb78bb88a27bd8e6f6cb84cbb5cef2a3105df4e8eeefc361816ebfb7d45",
        "layers.2.mlp.gate_proj.weight": "c84613bb9eaf35d8a73251cbdfbaa2de0838422a8411afcf33c40f476e5fa24c",
        "layers.2.mlp.up_proj.weight": "c3fe3a1a76203428dfd9ed99688d1ce1449b869b2ebc959d98d89019f6326b1a"
    },
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
