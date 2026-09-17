"""Reuse hash-pinned simulator assets without spending probe time on downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


ASSETS = (
    ('libttsim_bh_x2.so',
     'https://github.com/tenstorrent/ttsim/releases/download/v1.10.3/libttsim_bh_x2.so',
     '79287bd7cc1fc0fab28ca7b82567c39311f0dcc6ec2704ab7c4386dfc71abfd4'),
    ('cluster_descriptor.yaml',
     'https://raw.githubusercontent.com/tenstorrent/tt-umd/115b809170ff762182f925a07636887e5afb910e/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml',
     '27b7ec074f81fe4b4a1be89a5c39fd6c7eaf2c2d68da8f49bd8faf02a7d3df15'),
)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def prepare(cache, destination):
    cache, destination = Path(cache), Path(destination)
    cache.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url, checksum in ASSETS:
        started = time.monotonic()
        stored = cache / checksum
        target = destination / name
        if target.exists() or target.is_symlink():
            raise ValueError('Fresh asset destination required: ' + name)
        hit = stored.is_file() and not stored.is_symlink() and digest(stored) == checksum
        if not hit:
            with tempfile.TemporaryDirectory(prefix='download-', dir=cache) as temporary:
                download = Path(temporary) / name
                subprocess.run(['curl', '--fail', '--location', '--connect-timeout', '5',
                    '--max-time', '20', url, '-o', str(download)], check=True, timeout=25)
                if digest(download) != checksum:
                    raise ValueError('Downloaded simulator asset hash mismatch: ' + name)
                download.replace(stored)
        shutil.copyfile(stored, target)
        if digest(target) != checksum:
            raise ValueError('Staged simulator asset hash mismatch: ' + name)
        record = dict(name=name, sha256=checksum, cache_hit=hit,
            elapsed_seconds=time.monotonic() - started)
        records.append(record)
        print(json.dumps(dict(stage='simulator_asset', **record)), flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    options = parser.parse_args()
    prepare(options.cache, options.destination)
