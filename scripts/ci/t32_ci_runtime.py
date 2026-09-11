"""Admission and resource evidence for the isolated 16-CPU, 64-GiB simulator container."""

from pathlib import Path
import json
import os

from dspark_hardware_gate import digest
from native_draft_sdpa import audit_active_kernel


PINS = {
    'build_Release/lib/_ttnncpp.so': 'f65ac9e332d34ff462a051a021221fc12377b05711dc67d1faa5aa6fe37858c3',
    'build_Release/ttnn/_ttnncpp.so': 'd6c53113a104719a442b4d4a9ec2b344cdd0e00daa1e4d907afb9c13d1e531d9',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h':
        '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7',
}


def build_evidence(root):
    if os.environ.get('QWEN_T32_FP32_BUILD') != '1':
        return None
    if (os.environ.get('QWEN_SIM_ONLY') != '1'
            or any(os.environ.get(name) == '1' for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))):
        raise ValueError('Disposable FP32 build is simulator-only')
    from t32_attention_fp32_patch import SOURCE_PATH, SOURCE_SHA256

    root = Path(root)
    report = json.loads(Path('/experiment/results/t32-fp32-build.json').read_text())
    base = {name: value for name, value in PINS.items() if name.endswith('.so')}
    binaries = {name: digest(root / name) for name in base}
    candidate = '12e5cc308f777ebf01bbfd5b48fee71335af916eead62763e0db8953c720972e'
    variant = os.environ.get('QWEN_T32_FP32_VARIANT', 'fp32')
    if variant not in ('baseline', 'fp32'):
        raise ValueError('Explicit baseline or FP32 build required')
    if variant == 'baseline':
        candidate = SOURCE_SHA256
    if (report.get('stage') != 'built' or report.get('base_binaries') != base
            or report.get('variant', 'fp32') != variant
            or report.get('original_factory') != SOURCE_SHA256
            or report.get('candidate_factory') != candidate or digest(root / SOURCE_PATH) != candidate
            or report.get('binaries') != binaries or len(set(binaries.values())) != 1
            or any(binaries[name] == value for name, value in base.items())
            or report.get('registration_patch') != digest('/simulator-support/sdpa-graft-registration.patch')):
        raise ValueError('Complete matching disposable build provenance required')
    return report


def fingerprints(root):
    root = Path(root)
    audit_active_kernel(root)
    build = build_evidence(root)
    expected = dict(PINS)
    if build is not None:
        expected.update(build['binaries'])
    result = {name: digest(root / name) for name in PINS}
    if result != expected:
        raise ValueError('Exact CI binaries and simulator compatibility packer required')
    directory = root / 'ttnn/cpp/ttnn/operations/transformer/sdpa'
    result.update({str(path.relative_to(root)): digest(path) for path in sorted(directory.rglob('*'))
        if path.is_file() and path.suffix in ('.cpp', '.hpp', '.h')})
    return result


def snapshot(root=Path('/sys/fs/cgroup'), boot=Path('/proc/sys/kernel/random/boot_id')):
    root = Path(root)
    limits = {}
    for name in ('memory.max', 'memory.swap.max', 'memory.current', 'memory.peak', 'memory.swap.current'):
        value = (root / name).read_text().strip()
        limits[name] = None if value == 'max' else int(value)
        if limits[name] is not None and limits[name] < 0:
            raise ValueError('Nonnegative cgroup values required')
    quota, period = (root / 'cpu.max').read_text().split()
    if limits['memory.max'] != 64 * 1024**3 or quota == 'max' or int(period) <= 0 or int(quota) != 16 * int(period):
        raise ValueError('Enforced 16-CPU, 64-GiB simulator quota required')
    events = dict((name, int(value)) for name, value in
        (line.split() for line in (root / 'memory.events').read_text().splitlines()))
    if not {'oom', 'oom_kill'} <= events.keys() or any(value < 0 for value in events.values()):
        raise ValueError('Complete cgroup OOM counters required')
    identifier = Path(boot).read_text().strip()
    if not identifier:
        raise ValueError('Boot identity required')
    return dict(bounded=True, cgroup=str(root.resolve()), boot_id=identifier,
        limits=limits, events=events, cpu_quota=int(quota), cpu_period=int(period))
