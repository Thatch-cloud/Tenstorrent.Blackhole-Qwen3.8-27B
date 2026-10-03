from frozen_context_geometry import selected_geometry
"""Qualified factory inputs and evidence for the isolated four-link hardware build."""

import hashlib
from pathlib import Path

from frozen_combined_runtime import qualify, prepare_scratch, verify_scratch
from dspark_fp32_intermediates import SOURCE, transform


def prepare(root, scripts, *, enabled):
    if type(enabled) is not bool:
        raise ValueError('Explicit 8K factory build selection required')
    if not enabled:
        return None
    admission = qualify(scripts)
    scratch = prepare_scratch(root)
    path = Path(root) / SOURCE
    original = path.read_bytes()
    changed = transform(original)
    path.write_bytes(changed)
    return dict(report_sha256=admission['report_sha256'], source=SOURCE,
        source_before=hashlib.sha256(original).hexdigest(),
        factory_sha256=hashlib.sha256(changed).hexdigest(),
        key_chunk_size=256, capacity=selected_geometry()['capacity'], target_tree_scratch=scratch)


def completed(root, inputs, binary_sha256, *, import_passed):
    if import_passed is not True:
        raise ValueError('Successful runtime import check required')
    root = Path(root)
    verify_scratch(root, inputs.get('target_tree_scratch'))
    if hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() != inputs['factory_sha256']:
        raise ValueError('Hardware factory source changed during build/cache restoration')
    binaries = {}
    for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
        checksum = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if checksum != binary_sha256:
            raise ValueError('Hardware binary differs from qualified cache entry')
        binaries[name] = checksum
    return dict(passed=True, import_passed=True, factory_sha256=inputs['factory_sha256'],
        binaries=binaries, factory_inputs=inputs, scope='Build only; no request or performance qualification')
