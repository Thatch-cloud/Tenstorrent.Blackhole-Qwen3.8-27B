"""Qualified 64K factory inputs for the combined CCL/GDN runtime cache."""

import hashlib
import json
from pathlib import Path

from dspark_attention_64k_gate import REPORT_SHA256, qualify
from dspark_8k_build import completed as complete_binaries
import dspark_fp32_build as baseline
from dspark_ladder_build import BUILDERS as LADDER_BUILDERS, factory_scope


BUILDERS = (*LADDER_BUILDERS, 'dspark_64k_build.py', 'dspark_fp32_build.py',
    'dspark_fp32_intermediates.py', 'dspark_attention_64k_gate.py')


def fingerprints(directory):
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() for name in BUILDERS}


def verify_factory(root):
    source = (Path(root) / baseline.SOURCE).read_bytes()
    with factory_scope():
        original = baseline.restore_factory_source(source, baseline.REPLACEMENT)
        if baseline.transform(original, enabled=True) != source:
            raise ValueError('Exact qualified 64K factory transformation required')
    return hashlib.sha256(source).hexdigest()


def prepare(root, directory, report_path):
    evidence = qualify(directory, report_path)
    source_path = Path(root) / baseline.SOURCE
    original = source_path.read_bytes()
    with factory_scope():
        candidate = baseline.transform(original, enabled=True)
    source_path.write_bytes(candidate)
    return dict(report_sha256=evidence['report_sha256'], source=baseline.SOURCE,
        source_before=hashlib.sha256(original).hexdigest(), factory_sha256=verify_factory(root),
        capacity=66560, key_chunk_size=1024, builders=fingerprints(directory))


def completed(root, inputs, binary_sha256, *, import_passed):
    if inputs.get('report_sha256') != REPORT_SHA256 or verify_factory(root) != inputs.get('factory_sha256'):
        raise ValueError('Qualified 64K factory evidence required')
    evidence = complete_binaries(root, inputs, binary_sha256, import_passed=import_passed)
    evidence.update(backend='hardware', experiment='combined-runtime-64k', performance_qualified=False)
    return evidence


def validate_build(root, path, directory):
    evidence = json.loads(Path(path).read_text())
    if (evidence.get('backend') != 'hardware' or evidence.get('experiment') != 'combined-runtime-64k'
            or evidence.get('passed') is not True or evidence.get('import_passed') is not True):
        raise ValueError('Completed combined-runtime 64K build required')
    inputs = evidence.get('factory_inputs', {})
    if inputs.get('builders') != fingerprints(directory):
        raise ValueError('Combined 64K build dependencies changed')
    binaries = evidence.get('binaries', {})
    if set(binaries) != {'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'}:
        raise ValueError('Both combined runtime binaries required')
    checksum = next(iter(binaries.values()))
    expected = completed(root, inputs, checksum, import_passed=True)
    if any(evidence.get(name) != value for name, value in expected.items()):
        raise ValueError('Combined runtime evidence changed')
    return evidence
