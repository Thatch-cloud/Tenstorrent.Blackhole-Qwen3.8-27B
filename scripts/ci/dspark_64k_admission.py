"""Request-local admission checks for the experimental 64K combined runtime."""

from contextlib import contextmanager
from contextvars import ContextVar
import os

from dspark_attention_64k_gate import qualify
from dspark_64k_build import validate_build


_ADMISSION = ContextVar('dspark_64k_admission', default=None)


def current_admission():
    value = _ADMISSION.get()
    return None if value is None else dict(value)


def validate_request(context, output_tokens):
    if type(context) is not int or context != 65536 or type(output_tokens) is not int or output_tokens != 256:
        raise ValueError('64K comparison requires 65536 prompt rows and 256 output tokens')


@contextmanager
def admitted_request(directory, report_path, *, context, output_tokens, factory_root, build_path):
    validate_request(context, output_tokens)
    if _ADMISSION.get() is not None:
        raise ValueError('Nested 64K admission is not supported')
    if os.environ.get('QWEN_LADDER_BACKEND') != 'hardware':
        raise ValueError('64K combined runtime requires the hardware ladder backend')
    component = qualify(directory, report_path)
    build = validate_build(factory_root, build_path, directory)
    if build.get('backend') != 'hardware' or build.get('passed') is not True:
        raise ValueError('Matching successful hardware build required')
    admission = dict(context=context, output_tokens=output_tokens, capacity=66560,
        key_chunk_size=1024, native_padded_keys=67584,
        component_report_sha256=component['report_sha256'],
        factory_sha256=build['factory_sha256'], experimental_scope_admitted=True,
        full_request_qualified=False, performance_qualified=False)
    token = _ADMISSION.set(admission)
    try:
        yield dict(admission)
    finally:
        _ADMISSION.reset(token)
