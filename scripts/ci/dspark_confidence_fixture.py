"""Fetch only the pinned confidence head, not the multi-gigabyte checkpoint."""

import hashlib
import json

from draft_projection_fixture import read_range
from dspark_intake import MODEL, REVISION, HEADER_BYTES, HEADER_SHA256, CHECKPOINT_BYTES, validate_header


TENSORS = {
    'confidence_head.proj.weight': ('6162560ea0ae067ed6fcb0c46d0850682651196018134fa1a89dc44719b5a44f', (1, 5376)),
    'confidence_head.proj.bias': ('15f140ecfc7c93ed979d499e4295d54667fbfa852baed09eaa224ac7fa63d01d', (1,)),
}


def decode(name, data):
    import torch

    digest, shape = TENSORS[name]
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Confidence bytes differ from the pinned learned head')
    value = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape)
    if not torch.isfinite(value).all():
        raise ValueError('Finite learned confidence weights required')
    return value


def fetch():
    url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors'
    encoded, total = read_range(url, 8, HEADER_BYTES)
    if total != CHECKPOINT_BYTES or hashlib.sha256(encoded).hexdigest() != HEADER_SHA256:
        raise ValueError('Pinned checkpoint header required')
    header = json.loads(encoded)
    validate_header(header, HEADER_BYTES, CHECKPOINT_BYTES)
    result = {}
    for name, (digest, shape) in TENSORS.items():
        entry = header[name]
        if entry['dtype'] != 'BF16' or entry['shape'] != list(shape):
            raise ValueError('Pinned confidence tensor geometry required')
        start, end = entry['data_offsets']
        data, total = read_range(url, 8 + HEADER_BYTES + start, end - start)
        if total != CHECKPOINT_BYTES:
            raise ValueError('Checkpoint size changed during head download')
        result[name] = decode(name, data)
    return result
