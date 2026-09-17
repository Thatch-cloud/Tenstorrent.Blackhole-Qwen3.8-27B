"""Per-process simulator cgroup evidence; no global WSL configuration or arithmetic changes."""

from pathlib import Path


MEMORY_MAX = 3*1024**3
SWAP_MAX = 4*1024**3


def snapshot(*, bounded, membership=Path('/proc/self/cgroup'), root=Path('/sys/fs/cgroup'),
        boot=Path('/proc/sys/kernel/random/boot_id')):
    if type(bounded) is not bool:
        raise ValueError('Explicit simulator memory-budget selection required')
    groups = [line[3:] for line in membership.read_text().splitlines() if line.startswith('0::')]
    if len(groups)!=1:
        raise ValueError('Unified cgroup-v2 membership required')
    directory = (root/groups[0].lstrip('/')).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError('Simulator cgroup must remain inside the controller root')
    limits = {}
    for name in ('memory.max','memory.swap.max','memory.current','memory.peak','memory.swap.current'):
        value = (directory/name).read_text().strip()
        limits[name] = None if value=='max' else int(value)
        if limits[name] is not None and limits[name]<0:
            raise ValueError('Nonnegative memory-controller values required')
    events = {name:int(value) for name,value in (line.split() for line in (directory/'memory.events').read_text().splitlines())}
    if not {'oom','oom_kill'}<=set(events) or any(value<0 for value in events.values()):
        raise ValueError('Complete nonnegative cgroup OOM counters required')
    if bounded and (limits['memory.max']!=MEMORY_MAX or limits['memory.swap.max']!=SWAP_MAX):
        raise ValueError('Simulator requires an enforced 3-GiB resident / 4-GiB swap cgroup budget')
    identifier = boot.read_text().strip()
    if not identifier:
        raise ValueError('WSL boot identity required')
    return dict(bounded=bounded,cgroup=groups[0],boot_id=identifier,limits=limits,events=events)


def require_clean(before, after):
    if (any(before[name]!=after[name] for name in ('bounded','cgroup','boot_id'))
            or any(before['limits'][name]!=after['limits'][name] for name in ('memory.max','memory.swap.max'))
            or set(before['events'])!=set(after['events'])
            or any(after['events'][name]<value for name,value in before['events'].items())
            or any(after['events'][name]!=before['events'][name] for name in ('oom','oom_kill'))):
        raise ValueError('Unchanged boot/budget and no simulator cgroup OOM event required')
