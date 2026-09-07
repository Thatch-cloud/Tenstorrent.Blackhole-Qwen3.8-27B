"""Download only a bounded learned projection slice; never import checkpoint code."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request


MODEL = 'incoai/Qwen3.8-27B-DFlash2'
REVISION = 'dedf8df68adfb1afeaf7b7480c0a0243108177b4'
URL = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/model.safetensors'


def read_range(url, start, length):
    if type(start) is not int or start < 0 or type(length) is not int or not 0 < length <= 2097152:
        raise ValueError('Bounded positive byte range required')
    end = start + length - 1
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        matched = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if response.status != 206 or matched is None:
            raise ValueError('Server must honor the bounded range request')
        first, last, total = map(int, matched.groups())
        if (first, last) != (start, end) or total <= end:
            raise ValueError('Response byte range differs from request')
        data = response.read(length + 1)
        if len(data) != length:
            raise ValueError('Response byte count differs from request')
        return data, total


def fetch(output):
    header_size_bytes, total = read_range(URL, 0, 8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]
    header_bytes, header_total = read_range(URL, 8, header_size)
    header = json.loads(header_bytes)
    feature = header['fc.weight']
    if (header_total != total or feature['dtype'] != 'BF16' or feature['shape'] != [5120, 25600]
            or len(feature['data_offsets']) != 2):
        raise ValueError('Pinned feature projection geometry changed')
    first, last = feature['data_offsets']
    if (type(first) is not int or type(last) is not int or first < 0
            or last - first != 5120 * 25600 * 2 or 8 + header_size + last > total):
        raise ValueError('Invalid feature projection byte bounds')
    data, data_total = read_range(URL, 8 + header_size + first, 32 * 25600 * 2)
    if data_total != total:
        raise ValueError('Checkpoint size changed between range responses')
    manifest = dict(model=MODEL, revision=REVISION, tensor='fc.weight', dtype='BF16',
        full_shape=feature['shape'], slice_shape=[32, 25600], output_rows=[0, 32],
        checkpoint_bytes=total, fetched_bytes=8 + header_size + len(data),
        header_sha256=hashlib.sha256(header_bytes).hexdigest(), data_sha256=hashlib.sha256(data).hexdigest(),
        scope='First32 learned output neurons only, not the complete draft projection or model')
    output.mkdir(parents=True, exist_ok=True)
    data_path, manifest_path = output / 'fc-first32.bf16', output / 'manifest.json'
    if data_path.exists() or manifest_path.exists():
        raise ValueError('Refusing to overwrite an existing checkpoint fixture')
    data_path.write_bytes(data)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(fetch(parser.parse_args().output), indent=2))
