"""Temporary, hash-pinned precise exponential for DFlash2 SDPA, not target attention."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import sys


KERNEL_DIRECTORY = 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute'
SOURCE_HASHES = {
    'compute_common.hpp': '3fb5da2440c3bf90ebceb8acd55424c7739339de4c6b02db83836e1e3414fa19',
    'sdpa.cpp': 'a3f48af8ba0fd63b136c79a54c8b6f7b4b5b8fb0d7a209bf5701f081ed7fa3e0',
}
SIGNATURE = {0: 1, 1: 16, 2: 4, 4: 4, 5: 4, 6: 1, 23: 0, 24: 1, 30: 0}
ORIGINAL_INIT = '    exp_tile_init<true /* approx */, scale_fp32, InputClamping::None>();'
ORIGINAL_EXP = '                exp_tile<true /* approx */, false /* scale_en */, InputClamping::None, iterations>(j, vector_mode_exp);'
PRECISE_INIT = '''#ifndef QWEN_DRAFT_EXP_APPROX
#define QWEN_DRAFT_EXP_APPROX true
#endif
    exp_tile_init<QWEN_DRAFT_EXP_APPROX, scale_fp32, InputClamping::None>();'''
PRECISE_EXP = '''                exp_tile<QWEN_DRAFT_EXP_APPROX, !QWEN_DRAFT_EXP_APPROX, InputClamping::None, iterations>(
                    j, vector_mode_exp, static_cast<uint16_t>(scale_fp32 >> 16));'''
ORIGINAL_RECIP_INIT = '    recip_tile_init();'
ORIGINAL_RECIP = '        MATH((recip_tile_first_column(0)));'
PRECISE_RECIP_INIT = '''#if defined(QWEN_DRAFT_EXP_APPROX)
    if constexpr (!QWEN_DRAFT_EXP_APPROX) {
        MATH((ckernel::sfpu::sfpu_reciprocal_init<false>()));
    } else {
        recip_tile_init();
    }
#else
    recip_tile_init();
#endif'''
PRECISE_RECIP = '''#if defined(QWEN_DRAFT_EXP_APPROX)
        MATH((recip_tile_first_column<QWEN_DRAFT_EXP_APPROX>(0)));
#else
        MATH((recip_tile_first_column(0)));
#endif'''


def replacements():
    signature = ' && '.join(f'get_compile_time_arg_val({index}) == {value}' for index, value in SIGNATURE.items())
    include = '#include "compute_common.hpp"'
    result = {
        'compute_common.hpp': ((ORIGINAL_INIT, PRECISE_INIT), (ORIGINAL_EXP, PRECISE_EXP)),
        'sdpa.cpp': ((include, f'#define QWEN_DRAFT_EXP_APPROX (EXP_APPROX_MODE || !({signature}))\n{include}'),),
    }
    if os.environ.get('QWEN_T32_PRECISE_RECIP') == '1':
        if (os.environ.get('QWEN_SIM_ONLY') != '1'
                or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))):
            raise ValueError('Experimental reciprocal is simulator-only')
        result['compute_common.hpp'] += ((ORIGINAL_RECIP_INIT, PRECISE_RECIP_INIT),
                                        (ORIGINAL_RECIP, PRECISE_RECIP))
    return result


def patched_sources(original):
    if set(original) != set(SOURCE_HASHES):
        raise ValueError('Complete pinned SDPA kernel sources required')
    patched = {}
    for name, substitutions in replacements().items():
        source = original[name]
        if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError(f'Unaudited SDPA source: {name}')
        for before, after in substitutions:
            if source.count(before.encode()) != 1:
                raise ValueError(f'Unique SDPA source anchor required: {name}')
            source = source.replace(before.encode(), after.encode())
        patched[name] = source
    return patched


def audit_active_kernel(root):
    directory = Path(root) / KERNEL_DIRECTORY
    patched = {name: (directory / name).read_bytes() for name in SOURCE_HASHES}
    original = dict(patched)
    for name, substitutions in replacements().items():
        for before, after in reversed(substitutions):
            if original[name].count(after.encode()) != 1:
                raise ValueError(f'Precise draft kernel is not installed: {name}')
            original[name] = original[name].replace(after.encode(), before.encode())
    if patched_sources(original) != patched:
        raise ValueError('Precise draft source differs from audited transformation')
    if not (directory / '.qwen-precise-draft.lock').is_file():
        raise ValueError('Precise draft kernel requires its owning process')
    return dict(original=SOURCE_HASHES,
        patched={name: hashlib.sha256(source).hexdigest() for name, source in patched.items()},
        signature=SIGNATURE, scope=__doc__)


def run_precise_probe(script):
    root = os.environ['TT_METAL_HOME']
    if os.environ.get('QWEN_PRECISE_DRAFT_ACTIVE') == '1':
        audit = audit_active_kernel(root)
        audit['runtime_sources'] = {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            for name in ('ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp',
                         'build_Release/lib/_ttnncpp.so',
                         'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h')}
        return audit
    with precise_draft_kernel(root):
        environment = dict(os.environ, QWEN_PRECISE_DRAFT_ACTIVE='1')
        result = subprocess.run([sys.executable, str(script), *sys.argv[1:]], env=environment)
    raise SystemExit(result.returncode)


@contextmanager
def precise_draft_kernel(root):
    directory = Path(root) / KERNEL_DIRECTORY
    lock = directory / '.qwen-precise-draft.lock'
    original, changed = {}, []
    with lock.open('x'):
        pass
    try:
        original = {name: (directory / name).read_bytes() for name in SOURCE_HASHES}
        patched = patched_sources(original)
        for name, source in patched.items():
            changed.append(name)
            (directory / name).write_bytes(source)
        yield audit_active_kernel(root)
    finally:
        try:
            for name in reversed(changed):
                (directory / name).write_bytes(original[name])
        finally:
            lock.unlink()
