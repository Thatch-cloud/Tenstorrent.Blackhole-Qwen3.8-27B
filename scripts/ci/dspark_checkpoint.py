"""Stage the immutable DSpark v2 safetensors object without executing checkpoint code."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import urllib.request

from dspark_intake import MODEL, REVISION, CHECKPOINT_BYTES, HEADER_BYTES, HEADER_SHA256, validate_header
from dspark_markov_fixture import TENSORS


CHECKPOINT_SHA256 = '2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7'
CHUNK_BYTES = 2097152


def copy_response(response, output, expected_bytes, expected_sha256):
    partial = output.with_suffix(output.suffix + '.partial')
    if output.exists() or partial.exists():
        raise ValueError('Refusing to overwrite a checkpoint or incomplete download')
    if response.status != 200 or response.headers.get('Content-Length') != str(expected_bytes):
        raise ValueError('Complete response with exact pinned content length required')
    digest, count = hashlib.sha256(), 0
    with partial.open('xb') as target:
        while data := response.read(min(CHUNK_BYTES, expected_bytes - count + 1)):
            if count + len(data) > expected_bytes:
                raise ValueError('Checkpoint response exceeds pinned size')
            target.write(data)
            digest.update(data)
            count += len(data)
    if count != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError('Incomplete or hash-mismatched checkpoint retained as partial')
    partial.rename(output)


def verify(output):
    if (output.stat().st_size != CHECKPOINT_BYTES
            or output.with_suffix(output.suffix + '.partial').exists()):
        raise ValueError('Complete pinned safetensors object required')
    digest = hashlib.sha256()
    with output.open('rb') as source:
        for data in iter(lambda: source.read(CHUNK_BYTES), b''):
            digest.update(data)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('DSpark checkpoint differs from its pinned LFS object')
        source.seek(0)
        if struct.unpack('<Q', source.read(8))[0] != HEADER_BYTES:
            raise ValueError('Pinned safetensors header length required')
        data = source.read(HEADER_BYTES)
        if hashlib.sha256(data).hexdigest() != HEADER_SHA256:
            raise ValueError('Pinned safetensors header bytes required')
        header = json.loads(data)
        inventory = validate_header(header, HEADER_BYTES, CHECKPOINT_BYTES)
        matrix_hashes = {}
        for name, entry in TENSORS.items():
            start, end = header[name]['data_offsets']
            source.seek(8 + HEADER_BYTES + start)
            matrix_digest = hashlib.sha256()
            for offset in range(0, end - start, CHUNK_BYTES):
                data = source.read(min(CHUNK_BYTES, end - start - offset))
                matrix_digest.update(data)
            matrix_hashes[name] = matrix_digest.hexdigest()
            if matrix_hashes[name] != entry['sha256']:
                raise ValueError('Previously staged Markov matrix differs from the complete checkpoint')
    inventory.update(weight_payload_fetched=True, weight_payload_hash_verified=True)
    return dict(passed=True, model=MODEL, revision=REVISION, checkpoint_bytes=CHECKPOINT_BYTES,
        checkpoint_sha256=CHECKPOINT_SHA256, header_sha256=HEADER_SHA256, inventory=inventory,
        markov_subranges_sha256=matrix_hashes, remote_code_executed=False,
        learned_backbone_executed=False, eligible_for_hardware=False, eligible_for_serving=False,
        scope='Whole-object and previously staged matrix integrity only; not learned arithmetic or performance')


def fetch(output):
    if output.exists() or output.with_suffix(output.suffix + '.partial').exists():
        raise ValueError('Fresh checkpoint path required')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < CHECKPOINT_BYTES + 1073741824:
        raise ValueError('Insufficient free space for pinned checkpoint and1GiB headroom')
    url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors'
    with urllib.request.urlopen(url, timeout=90) as response:
        copy_response(response, output, CHECKPOINT_BYTES, CHECKPOINT_SHA256)
    return verify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reuse-verified', action='store_true')
    options = parser.parse_args()
    report = verify(options.output) if options.reuse_verified else fetch(options.output)
    print(json.dumps(report, indent=2))
