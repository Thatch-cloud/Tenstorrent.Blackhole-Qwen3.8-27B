"""Scoped deployment of simulator-qualified SFPU arithmetic to allocated hardware."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

from native_draft_sdpa import KERNEL_DIRECTORY, SOURCE_HASHES, patched_sources
from projection_link_policy import validate
from t32_attention_admission import PATCHED
from t32_attention_sum_patch import substitutions
from t32_proposal_gate import qualify


RUNTIME = {
    'build_Release/lib/_ttnncpp.so': '31374d9163054482273092050fce7400635264db5d8beaa8d92180176fbb8089',
    'build_Release/ttnn/_ttnncpp.so': '31374d9163054482273092050fce7400635264db5d8beaa8d92180176fbb8089',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h':
        '87b9c251202c28ffd8b3e419699b04de7d3f4cb4176fb8a28f586aa68b18d181',
}


def build_sources(original):
    result = patched_sources(original)
    source = result['compute_common.hpp']
    for before, after in substitutions():
        if source.count(before.encode()) != 1:
            raise ValueError('Unique qualified SFPU substitution required')
        source = source.replace(before.encode(), after.encode())
    result['compute_common.hpp'] = source
    if {name: hashlib.sha256(value).hexdigest() for name, value in result.items()} != PATCHED:
        raise ValueError('Hardware arithmetic differs from retained simulator kernel')
    return result


@contextmanager
def request_admission(root, evidence):
    import t32_attention_admission as admission

    root = Path(root)
    original = admission.require_active

    def require_hardware():
        if (evidence.get('runtime') != RUNTIME or evidence.get('patched') != PATCHED
                or evidence.get('full_request_qualified') is not False
                or evidence.get('proposal', {}).get('run') != 34660555430):
            raise ValueError('Explicit retained hardware experiment admission required')
        if validate(os.environ) != evidence['links']:
            raise ValueError('Hardware link policy changed inside request')
        current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME}
        kernels = {name: hashlib.sha256((root / KERNEL_DIRECTORY / name).read_bytes()).hexdigest() for name in PATCHED}
        if current != RUNTIME or kernels != PATCHED:
            raise ValueError('Admitted hardware runtime changed inside request')
        return evidence

    require_hardware()
    admission.require_active = require_hardware
    try:
        yield evidence
    finally:
        modified = admission.require_active is not require_hardware
        admission.require_active = original
        if modified:
            raise RuntimeError('T32 experiment admission changed during request')


@contextmanager
def installed(root, evidence, directory):
    forbidden = ('QWEN_SIM_ONLY', 'QWEN_T32_SFPU_SUM', 'QWEN_T32_FP32_BUILD',
        'QWEN_T32_PRECISE_RECIP', 'QWEN_T32_EXPLICIT_PACK')
    if (any(os.environ.get(name) == '1' for name in forbidden)
            or any(os.environ.get(name) for name in ('TT_METAL_SIMULATOR', 'TT_METAL_MOCK_CLUSTER_DESC_PATH'))
            or os.environ.get('QWEN_T32_NUMERATOR_TAP', '0') != '0'
            or os.environ.get('QWEN_PROJECTION_LINKS') != '4'):
        raise ValueError('Explicit allocated hardware scope without simulator or diagnostic modes required')
    links = validate(os.environ)
    if links['backend'] != 'hardware':
        raise ValueError('Allocated hardware required')
    report = qualify(evidence, directory)
    root = Path(root)
    runtime = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME}
    if runtime != RUNTIME:
        raise ValueError('Retained hardware binaries and native packer required')
    kernel = root / KERNEL_DIRECTORY
    lock = kernel / '.qwen-precise-draft.lock'
    with lock.open('x'):
        pass
    changed, original = [], {}
    try:
        original = {name: (kernel / name).read_bytes() for name in SOURCE_HASHES}
        candidate = build_sources(original)
        for name, source in candidate.items():
            changed.append(name)
            (kernel / name).write_bytes(source)
        yield dict(proposal=report, runtime=runtime, patched=PATCHED, links=links,
            scope='Hardware correctness experiment only', full_request_qualified=False)
    finally:
        try:
            for name in reversed(changed):
                (kernel / name).write_bytes(original[name])
        finally:
            lock.unlink()
