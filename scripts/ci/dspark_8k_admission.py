"""Explicit offline admission for the qualified 8K component; not runtime speed acceptance."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from pathlib import Path

from dspark_attention_8k_gate import qualify
from dspark_fp32_intermediates import ANCHOR, REPLACEMENT, SOURCE, transform


_ADMISSION = ContextVar('dspark_8k_admission', default=None)


def history_limit():
    return 8448 if _ADMISSION.get() is not None else 8192


def validate_request(context, output_tokens):
    if type(context) is not int or type(output_tokens) is not int or context != 8192 or output_tokens != 256:
        raise ValueError('8K trial requires exactly 8192 prompt rows and 256 output-token headroom')


def verify_factory(root):
    source = (Path(root) / SOURCE).read_bytes()
    if source.count(REPLACEMENT.encode()) != 1:
        raise ValueError('Qualified 256-key FP32-statistics factory is not installed')
    original = source.replace(REPLACEMENT.encode(), ANCHOR.encode())
    if transform(original) != source:
        raise ValueError('Qualified factory source differs from the pinned transformation')
    return hashlib.sha256(source).hexdigest()


@contextmanager
def admitted_request(directory, *, context, output_tokens, factory_root, build_evidence):
    validate_request(context, output_tokens)
    if _ADMISSION.get() is not None:
        raise ValueError('Nested 8K admission is not supported')
    evidence = qualify(directory)
    factory_sha256 = verify_factory(factory_root)
    if (build_evidence.get('factory_sha256') != factory_sha256
            or build_evidence.get('import_passed') is not True
            or build_evidence.get('passed') is not True):
        raise ValueError('Successful matching hardware factory build evidence required')
    binaries = build_evidence.get('binaries', {})
    expected = {'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'}
    if set(binaries) != expected or len(set(binaries.values())) != 1:
        raise ValueError('Both native binary paths must match the rebuilt hardware runtime')
    for name, checksum in binaries.items():
        if hashlib.sha256((Path(factory_root) / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Hardware factory binary changed')
    admission = dict(component=evidence, factory_sha256=factory_sha256, context=context,
        output_tokens=output_tokens, capacity=8448, key_chunk_size=256,
        full_request_qualified=False, performance_qualified=False)
    token = _ADMISSION.set(admission)
    try:
        yield admission
    finally:
        _ADMISSION.reset(token)
