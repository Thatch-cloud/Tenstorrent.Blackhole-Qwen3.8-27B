"""Content-addressed binary storage without hardware runtime selection."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from dspark_hardware_gate import digest


IMAGE = 'sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465'


def cache_key(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inspect_entry(cache, inputs):
    directory = Path(cache) / cache_key(inputs)
    if not directory.exists():
        return None
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('inputs') != inputs or digest(directory / '_ttnncpp.so') != manifest.get('binary_sha256'):
        raise ValueError('Cached runtime provenance or binary hash changed')
    return manifest


def store_entry(cache, inputs, binary):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    existing = inspect_entry(cache, inputs)
    if existing is not None:
        if existing['binary_sha256'] != digest(binary):
            raise ValueError('Refusing to replace a different binary under an existing cache key')
        return existing
    directory = Path(tempfile.mkdtemp(prefix=cache_key(inputs) + '.partial-', dir=cache))
    shutil.copy2(binary, directory / '_ttnncpp.so')
    manifest = dict(inputs=inputs, binary_sha256=digest(directory / '_ttnncpp.so'))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    directory.rename(cache / cache_key(inputs))
    return manifest
