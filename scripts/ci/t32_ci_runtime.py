"""Admission and resource evidence for the isolated 16-CPU, 64-GiB simulator container."""

from pathlib import Path

from dspark_hardware_gate import digest
from native_draft_sdpa import audit_active_kernel


PINS = {
    'build_Release/lib/_ttnncpp.so': 'f65ac9e332d34ff462a051a021221fc12377b05711dc67d1faa5aa6fe37858c3',
    'build_Release/ttnn/_ttnncpp.so': 'd6c53113a104719a442b4d4a9ec2b344cdd0e00daa1e4d907afb9c13d1e531d9',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h':
        '8aaf199a2439c5956ee077a5e9451981909e9589d5b81d1c5d7fc65f76e0e5d7',
}


def fingerprints(root):
    root = Path(root)
    audit_active_kernel(root)
    result = {name: digest(root / name) for name in PINS}
    if result != PINS:
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
