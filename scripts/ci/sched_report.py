"""Read a c2 serving container log for what the tp4/sched-1 levers change (the SCH pack's READ, references/fusion-jobs/SCH; stdlib only).

    python3 scripts/ci/sched_report.py <container log> [--second <container log>]

One log: the admission (what a returning turn's engine build cost, and with QWEN_FAST_ADMISSION_LEAN where: the A / B / C split, the readbacks taken and skipped, the backstop's
readings reused), the scheduler (the wall time of the prefill steps and of the decode rounds while a prefill was pending, the rounds the decoders got after each returning turn's
step, the longest stretch without a round, the guard and the cap in force) and the host tier (what each spill cost per block, the largest flush, the blocks the per-flush cap
would have dropped that complete mode spilled). With --second the same figures of the two logs side by side (B over A), for an ABAB's arm against its control.

Everything is read from the lines the engine already logs: the step lines of levern_policy (STEP_LINE and its suffixes), '[PINDIAG] sticky engine built', the lean markers of
serving_request_factory, '[PINDIAG] prefix: host tier spill' lines. A figure with no line behind it prints '-', never 0.
"""

import re
import statistics
import sys

STEP = re.compile(r'\[PINDIAG\] lever N step n=(\d+) kind=(prefill|decode) seats=(\d+) req=(\S+) start=(\S+) tokens=(\d+) end=(\S+) prompt=(\S+) final=(\S+) reason=(\S+) '
                  r'prev=([\w-]+):([\d.-]+)(?:ms)? owed_ms=(\S+) owed_rounds=(\d+)(?: f_eff=(\S+) need=(\S+) gap_ms=(\S+))?(?: base=(\S+) target=(\S+) adaptive=([01]))?'
                  r'(?: short=([01]) cap_ms=(\S+) guard=([01]))?')
STICKY = re.compile(r'\[PINDIAG\] sticky engine built req=(\S+) ms=([\d.]+) frontier=(\d+) prompt=(\d+)(?: kind=(build|rebind))?')
LEAN_SPLIT = re.compile(r'\[PINDIAG\] admission lean split request=(\S+) a_ms=([\d.]+) b_ms=([\d.]+) c_ms=([\d.]+) total_ms=([\d.]+)')
LEAN_ADOPT = re.compile(r'\[PINDIAG\] admission lean adopt request=(\S+) slot=(\d+) readback=([01]) ms=([\d.]+)')
LEAN_REUSED = '[PINDIAG] admission lean backstop reused the admission reading'
DRAM_AFTER = '[PINDIAG] dram after engine'
BACKSTOP_READ = re.compile(r'\[PINDIAG\] dram backstop (?:refused|unavailable)')
SPILL = re.compile(r'host tier spill blocks=(\d+) bytes=(\d+) ms=([\d.]+) held=(\d+)(?: batches=(\d+) lifted=(\d+))?')
SPILL_SLOW = 'host tier spill slow'
SPILL_COMPLETE = 'host tier spill complete on'


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))] if values else None


def _fit(rows):
    """(fixed ms, ms per block) of the least-squares line through (blocks, ms) rows, or None with fewer than three distinct sizes."""
    if len({row[0] for row in rows}) < 3:
        return None
    n = len(rows)
    sx, sy = sum(row[0] for row in rows), sum(row[1] for row in rows)
    sxx, sxy = sum(row[0] ** 2 for row in rows), sum(row[0] * row[1] for row in rows)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n, slope


def read(lines):
    steps, sticky, splits, adopts, spills = [], [], [], [], []
    reused = dram_lines = slow = complete = 0
    for line in lines:
        if 'lever N step' in line:
            match = STEP.search(line)
            if match:
                steps.append(match.groups())
        elif 'sticky engine built' in line:
            match = STICKY.search(line)
            if match:
                sticky.append((match.group(5) or 'build', float(match.group(2))))
        elif 'admission lean split' in line:
            match = LEAN_SPLIT.search(line)
            if match:
                splits.append(tuple(float(value) for value in match.groups()[1:]))
        elif 'admission lean adopt' in line:
            match = LEAN_ADOPT.search(line)
            if match:
                adopts.append((int(match.group(3)), float(match.group(4))))
        elif LEAN_REUSED in line:
            reused += 1
        elif DRAM_AFTER in line:
            dram_lines += 1
        elif SPILL_SLOW in line:
            slow += 1
        elif SPILL_COMPLETE in line:
            complete += 1
        elif 'host tier spill blocks=' in line:
            match = SPILL.search(line)
            if match:
                spills.append((int(match.group(1)), float(match.group(3)), int(match.group(6) or 0), int(match.group(5) or 0)))
    return dict(steps=steps, sticky=sticky, splits=splits, adopts=adopts, reused=reused, dram_lines=dram_lines, spills=spills, slow=slow, complete=complete)


def figures(found):
    """The figures of one log: a flat dict of name -> number or None."""
    out = {}
    rebinds = [ms for kind, ms in found['sticky'] if kind == 'rebind']
    builds = [ms for kind, ms in found['sticky'] if kind == 'build']
    out.update(rebinds=len(rebinds), rebind_p50=_pct(rebinds, 0.5), rebind_p90=_pct(rebinds, 0.9), builds=len(builds), build_p50=_pct(builds, 0.5))
    out.update(split_a_p50=_pct([row[0] for row in found['splits']], 0.5), split_b_p50=_pct([row[1] for row in found['splits']], 0.5),
               split_c_p50=_pct([row[2] for row in found['splits']], 0.5), split_total_p50=_pct([row[3] for row in found['splits']], 0.5), splits=len(found['splits']))
    out.update(adopts=len(found['adopts']), adopts_skipped=sum(1 for taken, _ in found['adopts'] if not taken), adopt_ms_p50=_pct([ms for _, ms in found['adopts']], 0.5),
               backstop_reused=found['reused'], dram_lines=found['dram_lines'])
    # the scheduler: a step line's prev=<kind>:<ms> is the wall time of the step BEFORE it, so the step lines give the time of every step but the last
    steps = [(int(s[0]), s[1], int(s[2]), s[3], s[8], s[10], float(s[11]) if s[11] not in ('-', '') else 0.0, s) for s in found['steps']]
    prefill_ms = sum(ms for *_, kind, ms, _ in steps if kind == 'prefill')
    decode_ms = sum(ms for *_, kind, ms, _ in steps if kind == 'decode')
    out.update(step_lines=len(steps), pending_prefill_share=prefill_ms / (prefill_ms + decode_ms) if prefill_ms + decode_ms else None)
    # rounds after a returning turn's step: the decode steps between a SHORT prefill step beside decoders and the next prefill step (only with the pacing's short= field)
    after, run, open_short = [], 0, False
    longest, stretch = 0.0, 0.0
    for index, (n, kind, seats, req, final, prev_kind, prev_ms, raw) in enumerate(steps):
        if kind == 'prefill':
            if open_short:
                after.append(run)
                open_short = False
            run = 0
            if raw[20] == '1' and seats > 0:
                open_short = True
        else:
            run += 1 if seats > 0 else 0
        # the stretch without a decode round that served a seat, from the wall times the lines carry (a line's prev is the step before it)
        if prev_kind == 'prefill':
            stretch += prev_ms
            longest = max(longest, stretch)
        if kind == 'decode' and seats > 0:
            stretch = 0.0
    out.update(rounds_after_short_p50=_pct(after, 0.5), rounds_after_short_max=max(after) if after else None, shorts_beside_decoders=len(after),
               longest_stretch_ms=longest if steps else None)
    gaps = [float(s[16]) for s in found['steps'] if s[16] not in (None, '-')]
    out.update(gap_ms_p90=_pct(gaps, 0.9), gap_ms_max=max(gaps) if gaps else None)
    guards = [s for s in found['steps'] if s[22] == '1']
    caps = {s[21] for s in found['steps'] if s[21] is not None}
    out.update(guard_steps=len(guards) if any(s[20] is not None for s in found['steps']) else None, cap_ms=float(sorted(caps)[0]) if len(caps) == 1 else None)
    # the host tier
    rows = [(blocks, ms) for blocks, ms, _lifted, _batches in found['spills']]
    fit = _fit(rows)
    big = [ms / blocks for blocks, ms in rows if blocks >= 200]
    out.update(spills=len(rows), spill_blocks=sum(row[0] for row in rows), spill_ms=sum(row[1] for row in rows), spill_max_blocks=max((row[0] for row in rows), default=None),
               spill_max_ms=max((row[1] for row in rows), default=None), spill_fixed_ms=fit[0] if fit else None, spill_ms_per_block=fit[1] if fit else (statistics.median(big) if big else None),
               spill_lifted=sum(row[2] for row in found['spills']) if found['complete'] else None, spill_slow=found['slow'] if found['complete'] else None,
               spill_complete=bool(found['complete']))
    return out


ROWS = (
    ('engine builds of returning turns (kind=rebind)', 'rebinds', 'rebind_p50', 'rebind_p90', 'ms (p50 / p90)'),
    ('cold engine builds', 'builds', 'build_p50', None, 'ms (p50)'),
    ('lean split A: host checks, backstop, adoption', 'splits', 'split_a_p50', None, 'ms (p50)'),
    ('lean split B: placement, refusal check, window join', 'splits', 'split_b_p50', None, 'ms (p50)'),
    ('lean split C: the rebind', 'splits', 'split_c_p50', None, 'ms (p50)'),
    ('slot adoptions with the readback skipped', 'adopts', 'adopts_skipped', None, 'of the adoptions logged'),
    ('backstop readings reused', 'backstop_reused', None, None, 'admissions'),
    ('"dram after engine" lines', 'dram_lines', None, None, 'lines'),
    ('prefill share of the wall time while a prefill is pending', 'step_lines', 'pending_prefill_share', None, 'share'),
    ('rounds after a returning turn beside decoders', 'shorts_beside_decoders', 'rounds_after_short_p50', 'rounds_after_short_max', 'rounds (p50 / max)'),
    ('longest stretch without a decode round', 'step_lines', 'longest_stretch_ms', None, 'ms'),
    ('guard steps', 'guard_steps', None, None, 'steps'),
    ('spills', 'spills', 'spill_ms_per_block', 'spill_fixed_ms', 'ms a block / fixed ms'),
    ('largest spill', 'spills', 'spill_max_blocks', 'spill_max_ms', 'blocks / ms'),
    ('blocks complete mode spilled beyond the cap', 'spill_complete', 'spill_lifted', 'spill_slow', 'blocks / slow flushes'),
)


def _fmt(value):
    if value is None:
        return '-'
    if isinstance(value, bool):
        return 'on' if value else '-'
    if isinstance(value, float):
        return '%.3f' % value if abs(value) < 10 else '%.1f' % value
    return str(value)


def report(found, out=sys.stdout):
    values = figures(found)
    out.write('%-62s %10s %12s %12s\n' % ('figure', 'count', 'first', 'second'))
    for label, count, first, second, unit in ROWS:
        out.write('%-62s %10s %12s %12s   %s\n' % (label, _fmt(values.get(count)), _fmt(values.get(first)) if first else '', _fmt(values.get(second)) if second else '', unit))
    return values


def compare(first, second, out=sys.stdout):
    """Side by side, B over A, for the figures that are numbers in both."""
    a, b = figures(first), figures(second)
    out.write('%-62s %12s %12s %9s\n' % ('figure', 'A', 'B', 'B / A'))
    for label, _count, key, second_key, unit in ROWS:
        for name in (key, second_key):
            if name is None or a.get(name) is None or b.get(name) is None or isinstance(a[name], bool):
                continue
            ratio = '%.2f' % (b[name] / a[name]) if a[name] else '-'
            out.write('%-62s %12s %12s %9s   %s\n' % (label + (' [%s]' % name if name == second_key else ''), _fmt(a[name]), _fmt(b[name]), ratio, unit))
    return a, b


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    second = None
    if '--second' in arguments:
        index = arguments.index('--second')
        second = arguments[index + 1]
        del arguments[index:index + 2]
    if len(arguments) != 1:
        sys.stderr.write(__doc__.split(chr(10))[0] + chr(10))
        return 2
    with open(arguments[0], errors='replace') as stream:
        found = read(stream)
    if second is None:
        report(found)
        return 0
    with open(second, errors='replace') as stream:
        other = read(stream)
    compare(found, other)
    return 0


if __name__ == '__main__':
    sys.exit(main())
