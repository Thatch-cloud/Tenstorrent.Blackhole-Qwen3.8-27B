"""Engine-start time per phase, read from a served container's log (stdlib only).

    python3 scripts/ci/engine_start_phases.py LOG [LOG ...]              one JSON report per log
    python3 scripts/ci/engine_start_phases.py --control LOG --flag LOG   the two side by side: seconds per phase, the difference, the ratio

Why: the engine start of the four-card stack takes 2 min 47 s on an idle rig and 9 min 43 s under CI load, and the production deploy's platform timeout is about
13.5 minutes (docs/tp4-fabric-upload.md section 1). Which phase moves when an upload lever (QWEN_FAST_DEVICE_ZEROS, QWEN_FAST_LAZY_SHARD_W) is switched on is
read from the log of two engine starts, one per arm (references/upload-p0-jobs), never from a memory of one.

A phase is the time between two MILESTONES, each the first log line matching one of its patterns at or after the previous milestone:

  engine_init      the vLLM engine core starts                                  -> mesh_open       vLLM init to the mesh and fabric open
  mesh_open        the (1, 4) mesh is created                                   -> layers_begin    config, state-dict read, tokenizer
  layers_begin     the 64-layer load starts                                     -> layers_end      THE LAYER LOAD (every projection a cache hit in production)
  layers_end       the layer progress bar completes (its next timestamp)        -> kv_begin        LM head, final norm, vision, KV sizing
  kv_begin         vLLM reports the KV pool size                                -> kv_end          THE KV POOL: allocation of 32 tensors, GDN state reset
  kv_end           the worker logs the prefix block size                        -> warm_end        prefix warm, memory ledger P0
  warm_end         ledger phase P0                                              -> pool_begin      (ledger P1)
  pool_begin       ledger phase P1                                              -> pool_end        THE BUFFER POOL and the admission checks before it
  pool_end         ledger phase P2                                              -> eager_warm_end  drafter weights upload and the eager prefill warm
  eager_warm_end   the four-card eager prefill is warmed                        -> startup         CAPTURES (the packed traces)
  startup          the API server's "Application startup complete" (it has no timestamp; the last timestamp before it)

A line carries a timestamp in either of the formats the log holds: loguru's `YYYY-MM-DD HH:MM:SS.mmm` and vLLM's `LEVEL MM-DD HH:MM:SS`; a line without one (the
progress bars, uvicorn's) takes the time of the last line before it that had one (the layer load ends at its last layer's log line, not at the bar). Milestones that never appear are listed under `missing`, and the phases that
need them are omitted: a log of a profile without prefix reuse has no "prefix block size" line, so kv_end falls back to the ledger line.

It also reads what the upload levers themselves log: the device-zero fill's per-tag totals (device and host tensors, bytes per card, seconds), the lazy shard loader's
load counters (hits and misses), and the audits' verdicts.
"""

import argparse
import datetime
import json
import re
import sys

LOGURU = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})')
VLLM = re.compile(r'\b(?:INFO|WARNING|ERROR|DEBUG) (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b')

# (name, patterns): the first line matching any pattern, in log order after the previous milestone. 'layers_end' is special (see milestones()).
MILESTONES = (
    ('engine_init', (r'Initializing a V1 LLM engine',)),
    ('mesh_open', (r'multidevice with \d+ devices and grid',)),
    ('layers_begin', (r'Loading \d+ transformer layers',)),
    ('layers_end', (r'Loading layers: 100%',)),
    ('kv_begin', (r'GPU KV cache size:',)),
    ('kv_end', (r'\[PINDIAG\] prefix: worker block_size=', r'\[MEMLEDGER\] phase=P0 chip0')),
    ('warm_end', (r'\[MEMLEDGER\] phase=P0 chip0',)),
    ('pool_begin', (r'\[MEMLEDGER\] phase=P1 chip0',)),
    ('pool_end', (r'\[MEMLEDGER\] phase=P2 chip0',)),
    ('eager_warm_end', (r'four-card eager prefill warmed before the packed traces',)),
    ('startup', (r'Application startup complete',)),
)
# phase name -> (from milestone, to milestone), in order.
PHASES = (
    ('vllm_init_to_mesh', 'engine_init', 'mesh_open'),
    ('config_and_state_dict', 'mesh_open', 'layers_begin'),
    ('layers', 'layers_begin', 'layers_end'),
    ('lm_head_vision', 'layers_end', 'kv_begin'),
    ('kv_pool', 'kv_begin', 'kv_end'),
    ('prefix_warm', 'kv_end', 'warm_end'),
    ('ledger_to_pool', 'warm_end', 'pool_begin'),
    ('buffer_pool', 'pool_begin', 'pool_end'),
    ('drafter_and_eager_warm', 'pool_end', 'eager_warm_end'),
    ('captures', 'eager_warm_end', 'startup'),
)
ZEROS_SUMMARY = re.compile(r'\[PINDIAG\] tp4 device zeros engaged summary tag=(\S+) device=(\d+) host=(\d+) bytes_per_card=(\d+) seconds=([0-9.]+) latched=(\S+)')
LAZY_LOADS = re.compile(r'\[PINDIAG\] tp4 lazy shard loads calls=(\d+) misses=(\d+) hits=(\d+) latched=(\S+)')


def timestamp(line, year=1970):
    """The line's time as epoch seconds, else None. vLLM's format has no year: `year` is the log's (see parse)."""
    match = LOGURU.search(line)
    if match:
        parts = [int(part) for part in match.groups()]
        try:
            return datetime.datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6] * 1000).timestamp()
        except ValueError:
            return None
    match = VLLM.search(line)
    if match:
        month, day, hour, minute, second = (int(part) for part in match.groups())
        try:
            return datetime.datetime(year, month, day, hour, minute, second).timestamp()
        except ValueError:
            return None
    return None


def parse(lines):
    """-> dict(milestones={name: seconds}, phases={name: seconds}, total, missing=[...]). Seconds are on the log's own clock (an arbitrary epoch).

    Each milestone is the first line matching one of its patterns at or after the line of the previous milestone that was found, so a missing milestone leaves
    the ones after it findable and a later duplicate (a second engine start in the same log) is never taken."""
    lines = list(lines)
    year = 1970
    for line in lines:
        match = LOGURU.search(line)
        if match:
            year = int(match.group(1))
            break
    clock, last, clocks = None, None, []
    for line in lines:
        stamp = timestamp(line, year)
        if stamp is not None:
            if last is not None and stamp < last - 12 * 3600:
                stamp += 86400                                  # past midnight
            clock = last = stamp
        clocks.append(clock)
    found, position = {}, 0
    for name, options in MILESTONES:
        patterns = [re.compile(option) for option in options]
        for index in range(position, len(lines)):
            if clocks[index] is not None and any(pattern.search(lines[index]) for pattern in patterns):
                found[name] = clocks[index]
                position = index
                break
    missing = [name for name, _ in MILESTONES if name not in found]
    phases = {}
    for name, begin, end in PHASES:
        if begin in found and end in found:
            phases[name] = round(found[end] - found[begin], 3)
    total = round(found['startup'] - found['engine_init'], 3) if 'engine_init' in found and 'startup' in found else None
    return dict(milestones={name: round(value, 3) for name, value in found.items()}, phases=phases, total=total, missing=missing)


def levers(text):
    """What the upload levers logged: per-tag device-zero totals, lazy-loader counters, audit verdicts."""
    zeros = {}
    for tag, device, host, bytes_per_card, seconds, latched in ZEROS_SUMMARY.findall(text):
        zeros[tag] = dict(device=int(device), host=int(host), bytes_per_card=int(bytes_per_card), seconds=float(seconds), latched=latched)
    lazy = LAZY_LOADS.findall(text)
    out = dict(device_zeros=zeros)
    if lazy:
        calls, misses, hits, latched = lazy[-1]
        out['lazy_shard'] = dict(calls=int(calls), misses=int(misses), hits=int(hits), latched=latched)
    audits = []
    for line in text.splitlines():
        if line.startswith('[PINDIAG] tp4 device zeros audit') or '[PINDIAG] tp4 device zeros audit' in line:
            verdict = re.search(r'exact=(True|False)', line)
            tag = re.search(r'tag=(\S+)', line)
            if verdict:
                audits.append(dict(lever='device_zeros', tag=tag.group(1) if tag else None, exact=verdict.group(1) == 'True'))
        elif '[PINDIAG] tp4 lazy shard audit' in line:
            verdict = re.search(r'exact=(True|False)', line)
            if verdict:
                audits.append(dict(lever='lazy_shard', exact=verdict.group(1) == 'True'))
    if audits:
        out['audits'] = audits
    return out


def report(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    result = parse(text.splitlines())
    result['levers'] = levers(text)
    return result


def compare(control, flag):
    """Side by side: [(phase, control seconds, flag seconds, difference, ratio)] for the phases both logs have, and the totals."""
    rows = []
    for name, _, _ in PHASES:
        if name in control['phases'] and name in flag['phases']:
            a, b = control['phases'][name], flag['phases'][name]
            rows.append((name, a, b, round(b - a, 3), round(b / a, 3) if a else None))
    if control.get('total') is not None and flag.get('total') is not None:
        a, b = control['total'], flag['total']
        rows.append(('total', a, b, round(b - a, 3), round(b / a, 3) if a else None))
    return rows


def table(rows):
    lines = ['%-26s %10s %10s %10s %7s' % ('phase', 'control s', 'flag s', 'delta s', 'ratio')]
    for name, a, b, delta, ratio in rows:
        lines.append('%-26s %10.1f %10.1f %+10.1f %7s' % (name, a, b, delta, '-' if ratio is None else '%.2f' % ratio))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('logs', nargs='*')
    parser.add_argument('--control')
    parser.add_argument('--flag')
    arguments = parser.parse_args(argv)
    if arguments.control or arguments.flag:
        if not (arguments.control and arguments.flag) or arguments.logs:
            parser.error('--control and --flag go together and take no other log')
        control, flag = report(arguments.control), report(arguments.flag)
        print(json.dumps(dict(control=control, flag=flag, rows=compare(control, flag)), indent=2))
        print(table(compare(control, flag)))
        return 0
    if not arguments.logs:
        parser.error('a log, or --control and --flag')
    for path in arguments.logs:
        print(json.dumps(dict(log=path.rsplit('/', 1)[-1], **report(path)), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
