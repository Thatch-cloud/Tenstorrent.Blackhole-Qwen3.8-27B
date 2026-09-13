"""Opt-in cache factory graft for the combined hardware runtime build."""

import hashlib
from pathlib import Path

from dspark_cached_markov_gate import qualify
from markov_sparse_fp32 import SOURCE, transform


def prepare(root, scripts, *, enabled):
    if type(enabled) is not bool:
        raise ValueError('Explicit cached Markov build selection required')
    if not enabled:
        return None
    admission = qualify(scripts, scripts)
    source = Path(root) / SOURCE
    original = source.read_bytes()
    changed = transform(original)
    source.write_bytes(changed)
    return dict(admission=admission, source=SOURCE,
        source_before=hashlib.sha256(original).hexdigest(),
        factory_sha256=hashlib.sha256(changed).hexdigest())


def completed(root, inputs, binary_sha256, *, import_passed):
    if import_passed is not True:
        raise ValueError('Successful sparse runtime import required')
    root = Path(root)
    if hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() != inputs['factory_sha256']:
        raise ValueError('Sparse factory changed during build')
    binaries = {}
    for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
        checksum = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if checksum != binary_sha256:
            raise ValueError('Sparse runtime binary differs from build cache')
        binaries[name] = checksum
    return dict(passed=True, import_passed=True, factory_inputs=inputs,
        factory_sha256=inputs['factory_sha256'], binaries=binaries,
        performance_qualified=False, full_vocabulary_qualified=False)
