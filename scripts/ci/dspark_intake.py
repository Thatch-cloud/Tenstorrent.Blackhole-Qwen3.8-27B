"""Bounded DSpark metadata intake; never execute checkpoint code or claim weight validation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

from draft_projection_fixture import read_range


MODEL = 'RadixArk/Qwen3.8-27B-DSpark'
REVISION = 'b9a5dbdf03bc999c6c73c426b19c2d9041cea393'
CHECKPOINT_BYTES = 3714723322
HEADER_BYTES = 6640
HEADER_SHA256 = '992cdd260cf8761176cb7d6e94a64339189819e74609e7f0c2989ec33ee182f1'
FILES = {
    'config.json': (2448, 'dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd'),
    'dspark.py': (7016, '75bba7c469166bd2d6a6877b9964d035b0cec9a83190ab9f3de8c6175883a114'),
    'dflash.py': (20651, '9825996703de73bd436a6aaf57ae203ef92d599249d9cf49078576d52f4e56a4'),
}
TAPS = [5, 19, 33, 47, 61]


def tensor_shapes():
    result = {
        'confidence_head.proj.bias': [1],
        'confidence_head.proj.weight': [1, 5376],
        'fc.weight': [5120, 25600],
        'hidden_norm.weight': [5120],
        'norm.weight': [5120],
        'markov_head.markov_w1.weight': [248320, 256],
        'markov_head.markov_w2.weight': [248320, 256],
    }
    layer_shapes = {
        'input_layernorm.weight': [5120],
        'post_attention_layernorm.weight': [5120],
        'mlp.gate_proj.weight': [17408, 5120],
        'mlp.up_proj.weight': [17408, 5120],
        'mlp.down_proj.weight': [5120, 17408],
        'self_attn.q_proj.weight': [4096, 5120],
        'self_attn.k_proj.weight': [1024, 5120],
        'self_attn.v_proj.weight': [1024, 5120],
        'self_attn.o_proj.weight': [5120, 4096],
        'self_attn.q_norm.weight': [128],
        'self_attn.k_norm.weight': [128],
    }
    for layer in range(5):
        result.update({f'layers.{layer}.{name}': shape for name, shape in layer_shapes.items()})
    return result


def validate_config(config):
    expected = dict(architectures=['DSparkDraftModel'], block_size=7, training_block_size=16,
        hidden_size=5120, intermediate_size=17408, num_hidden_layers=5, num_attention_heads=32,
        num_key_value_heads=8, head_dim=128, num_target_layers=64, vocab_size=248320,
        draft_vocab_size=248320, target_layer_ids=TAPS, dtype='bfloat16', attention_bias=False,
        layer_types=['full_attention'] * 5, sliding_window=None, use_sliding_window=False,
        markov_head_type='vanilla', markov_rank=256, projector_type='dspark',
        enable_confidence_head=True, confidence_head_with_markov=True,
        mask_token_id=248070, eos_token_id=248044, rms_norm_eps=1e-6)
    if not isinstance(config, dict) or any(key not in config or type(config[key]) is not type(value)
            or config[key] != value for key, value in expected.items()):
        raise ValueError('Pinned DSpark v2 architecture and serving/training widths required')
    nested = dict(attention_mode='gqa', confidence_head_alpha=1.0, confidence_head_with_markov=True,
        enable_confidence_head=True, markov_head_type='vanilla', markov_rank=256,
        mask_token_id=248070, projector_type='dspark', target_layer_ids=TAPS)
    if any(config.get(name) != nested for name in ('dflash_config', 'dspark_config')):
        raise ValueError('Both nested DSpark configuration views must agree')
    rope = dict(beta_fast=32.0, beta_slow=1.0, factor=32.0, original_max_position_embeddings=8192,
        rope_theta=10000000, rope_type='yarn')
    scaling = {key: value for key, value in rope.items() if key != 'rope_theta'}
    if config.get('rope_parameters') != rope or config.get('rope_scaling') != scaling:
        raise ValueError('Pinned YaRN parameters required; default DFlash2 rotary tables are incompatible')
    return dict(serving_proposals=7, target_verify_rows=8, training_future_positions=16,
        wider_serving_qualified=False, attention='full', rotary='yarn', target_feature_layers=TAPS.copy(),
        selector='Sequential full-vocabulary vanilla Markov bias; not DFlash2 top-16 selection')


def validate_header(header, header_size, checkpoint_bytes):
    expected = tensor_shapes()
    if (type(header_size) is not int or header_size != HEADER_BYTES
            or type(checkpoint_bytes) is not int or checkpoint_bytes != CHECKPOINT_BYTES
            or not isinstance(header, dict) or header.get('__metadata__') != {'format': 'pt'}
            or set(header) != set(expected) | {'__metadata__'}):
        raise ValueError('Exact pinned checkpoint size and tensor inventory required')
    ranges = []
    for name, shape in expected.items():
        entry = header[name]
        if (not isinstance(entry, dict) or entry.get('dtype') != 'BF16' or entry.get('shape') != shape
                or any(type(value) is not int for value in entry['shape'])):
            raise ValueError(f'Invalid BF16 geometry for {name}')
        offsets = entry.get('data_offsets')
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != 2 * math.prod(shape)):
            raise ValueError(f'Invalid byte extent for {name}')
        ranges.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError('Tensor extents must not overlap or leave gaps')
        cursor = end
    if cursor + header_size + 8 != checkpoint_bytes:
        raise ValueError('Tensor extents must exactly cover the checkpoint payload')
    return dict(tensors=len(expected), parameters=cursor // 2, dtype='BF16', payload_bytes=cursor,
        weight_payload_fetched=False, weight_payload_hash_verified=False)


def verified_file(name, data):
    size, digest = FILES[name]
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Pinned source bytes changed: {name}')
    return data


def fetch_metadata():
    base = f'https://huggingface.co/{MODEL}'
    files = {}
    for name, (size, _) in FILES.items():
        with urllib.request.urlopen(f'{base}/raw/{REVISION}/{name}', timeout=60) as response:
            if response.status != 200:
                raise ValueError('Successful pinned metadata response required')
            files[name] = verified_file(name, response.read(size + 1))
    url = f'{base}/resolve/{REVISION}/model.safetensors'
    size_bytes, total = read_range(url, 0, 8)
    if total != CHECKPOINT_BYTES or struct.unpack('<Q', size_bytes)[0] != HEADER_BYTES:
        raise ValueError('Pinned header and checkpoint lengths required before fetching header')
    data, header_total = read_range(url, 8, HEADER_BYTES)
    if header_total != total or hashlib.sha256(data).hexdigest() != HEADER_SHA256:
        raise ValueError('Pinned checkpoint header bytes required')
    contract = validate_config(json.loads(files['config.json']))
    inventory = validate_header(json.loads(data), HEADER_BYTES, total)
    files['model.header.json'] = data
    report = dict(passed=True, model=MODEL, revision=REVISION, checkpoint_bytes=total,
        header_sha256=HEADER_SHA256, files={name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
            for name, value in files.items()}, fetched_bytes=8 + sum(map(len, files.values())),
        contract=contract, inventory=inventory, remote_code_executed=False,
        eligible_for_simulator=False, eligible_for_hardware=False, eligible_for_serving=False,
        scope='Metadata compatibility only; no learned arithmetic, acceptance, quality or speed qualification')
    return report, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    output = parser.parse_args().output
    if output.exists():
        raise ValueError('Refusing to overwrite an existing intake')
    report, files = fetch_metadata()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = name + '.source.txt' if name.endswith('.py') else name
        (output / target).write_bytes(data)
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
