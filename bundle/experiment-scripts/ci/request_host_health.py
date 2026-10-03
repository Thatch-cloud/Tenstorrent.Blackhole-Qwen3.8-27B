"""Read-only Linux request-boundary diagnostics, not a device performance gate."""

from pathlib import Path
import time


FILES = ('cpu.stat', 'cpu.max', 'cpu.pressure', 'memory.current', 'memory.peak',
    'memory.events', 'memory.swap.current', 'memory.pressure', 'io.pressure')


def snapshot(root=Path('/sys/fs/cgroup')):
    result = dict(monotonic_ns=time.monotonic_ns(), cgroup={}, unavailable={})
    for name in FILES:
        try:
            result['cgroup'][name] = (Path(root) / name).read_text().strip()
        except OSError as error:
            result['unavailable'][name] = type(error).__name__
    return result


def counters(text):
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


def summarize(before, after):
    elapsed = after['monotonic_ns'] - before['monotonic_ns']
    if elapsed <= 0:
        raise ValueError('Ordered host snapshots required')
    deltas = {}
    for name in ('cpu.stat', 'memory.events'):
        if name not in before['cgroup'] or name not in after['cgroup']:
            continue
        initial, final = counters(before['cgroup'][name]), counters(after['cgroup'][name])
        deltas[name] = {key: final[key] - initial[key] for key in initial.keys() & final.keys()}
        if any(value < 0 for value in deltas[name].values()):
            raise ValueError('Host counters reset across request; comparison is invalid')
    return dict(before=before, after=after, elapsed_ms=elapsed / 1e6, counter_deltas=deltas,
        scope='Whole request including setup and audits; not isolated decode or proof of exclusive host use')
