"""Opt-in cache factory graft for the combined hardware runtime build."""

import hashlib
from pathlib import Path

from dspark_cached_markov_gate import qualify
from markov_sparse_fp32 import SOURCE, transform


def admitted_reference(root, scripts, reference, evidence):
    from markov_sparse_fp32 import SOURCE_SHA256, INSERT, CONFIG_REPLACEMENT, CONFIG
    if reference.get(SOURCE) != SOURCE_SHA256:
        raise ValueError('Original pinned sparse reference required')
    inputs = evidence['factory_inputs']
    if inputs.get('admission') != qualify(scripts, scripts) or inputs.get('source_before') != SOURCE_SHA256:
        raise ValueError('Accepted simulator admission and original sparse source required')
    source = (Path(root) / SOURCE).read_bytes()
    original = source.replace(INSERT.encode(), b'').replace(CONFIG_REPLACEMENT.encode(), CONFIG.encode())
    if transform(original) != source:
        raise ValueError('Exact qualified sparse transformation required')
    checked = completed(root, inputs, evidence['binaries']['build_Release/lib/_ttnncpp.so'],
        import_passed=evidence.get('import_passed'))
    if any(evidence.get(key) != value for key, value in checked.items()):
        raise ValueError('Sparse build evidence differs from loaded runtime')
    return dict(reference, **{SOURCE: checked['factory_sha256']})


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
