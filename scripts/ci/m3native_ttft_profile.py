"""Derive the admission profile from the gate's own per-stream records.

The four-user arm already records `ttft_s`, `gaps_ms` and `wall_s` for every stream,
and run 35658854824 showed those numbers carry a precise story that no gate asserted:

  TTFT   13.5 / 26.5 / 39.6 / 52.5 s   deltas 13.0, 13.1, 12.9 - prefill is SERIAL
  gap[0] 39.5 / 26.5 / 13.4 /  0.5 s   each user freezes for the prefills still to come

79.4 s of stall against 257.4 s of user-facing wall, 31 percent, and nothing in CI
would have noticed. This module turns that reading into a computed profile so a change
to prefill scheduling is falsifiable rather than anecdotal.

Two distinct signatures, deliberately separated, because run 35667198294 produced the
second without the first:

  SERIAL PREFILL - the TTFT deltas agree with each other, and each user's first
  inter-token gap matches (remaining prefills) x (the prefill interval). When both hold,
  the stall is explained by admission order alone.

  SYNCHRONISED STALL - two or more users stall on the SAME round index. That cannot be
  admission order, since by then every prefill has finished; it is a global engine
  event. Raising MAX_PREFILL_CHUNK_SIZE introduced exactly this (rounds 27-30, all four
  users, ~25.6 s), and reporting it separately is what made the regression legible.

Reporting is unconditional. Assertion is opt-in through thresholds, because the right
ceiling is a product decision and a gate that fails without one would just be noise.
"""

import os
import statistics


FLAG_MAX_STALL = 'M3NATIVE_TTFT_MAX_STALL_S'
FLAG_MAX_TTFT = 'M3NATIVE_TTFT_MAX_S'
STALL_MS = 1000.0
# Deltas this close count as agreeing. The measured spread across three intervals was
# 0.2 s on a 13.1 s interval; 15 percent leaves room for load without admitting a
# genuinely interleaved run, which would show deltas near the round time instead.
SERIAL_TOLERANCE = 0.15
# Agreeing deltas alone do not mean serial prefill: four users admitted 0.3 s apart also
# agree. A staircase STEP has to be a prefill, which is many rounds long, so the interval
# must also dwarf the steady-state round time. v34: 13.0 s against a 0.253 s round, 51x.
# Near-simultaneous admission sits near 1x and is correctly refused.
SERIAL_MIN_ROUNDS = 5.0


def _threshold(environ, name):
    value = (os.environ if environ is None else environ).get(name)
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be a number of seconds' % name)
    if not seconds > 0:
        raise ValueError('%s must be positive' % name)
    return seconds


def synchronised_stalls(streams, stall_ms=STALL_MS):
    """Round indices where more than one user stalls, with the per-round durations.

    Index 0 is excluded: every user's first gap is the admission stall by construction,
    so counting it here would report the serial-prefill signature twice under a name
    that means something else.
    """
    rounds = {}
    for stream in streams:
        for index, gap in enumerate((stream or {}).get('gaps_ms') or []):
            if index and gap > stall_ms:
                rounds.setdefault(index, []).append(gap / 1000.0)
    return [dict(round=index, users=len(values), seconds=sorted(values))
            for index, values in sorted(rounds.items()) if len(values) > 1]


def profile(streams, environ=None):
    streams = [s for s in streams if s]
    if not streams:
        raise ValueError('At least one completed stream required for a TTFT profile')

    ttfts = sorted(s['ttft_s'] for s in streams if s.get('ttft_s') is not None)
    deltas = [b - a for a, b in zip(ttfts, ttfts[1:])]
    interval = statistics.median(deltas) if deltas else None

    # Steady-state round time, from every gap that is not a stall.
    steady = [g for s in streams for g in (s.get('gaps_ms') or []) if g <= STALL_MS]
    round_s = (statistics.median(steady) / 1000.0) if steady else None

    # Serial prefill needs BOTH: the deltas agree with each other, and the interval is
    # many rounds long. The second half is what separates a prefill staircase from four
    # users admitted a few hundred milliseconds apart.
    serial = bool(deltas) and interval > 0 and all(
        abs(d - interval) <= SERIAL_TOLERANCE * interval for d in deltas) and (
        round_s is None or interval >= SERIAL_MIN_ROUNDS * round_s)

    first_gaps, predicted = [], []
    for position, stream in enumerate(sorted(streams, key=lambda s: s.get('ttft_s') or 0.0)):
        gaps = stream.get('gaps_ms') or []
        first_gaps.append((gaps[0] / 1000.0) if gaps else None)
        # A user admitted at position p still waits for the (n - 1 - p) prefills behind it.
        predicted.append((len(streams) - 1 - position) * interval if interval else None)

    residuals = [abs(a - b) for a, b in zip(first_gaps, predicted)
                 if a is not None and b is not None]
    explained = bool(residuals) and max(residuals) <= max(1.0, 0.1 * (interval or 0.0))

    stall_total = sum(g for s in streams for g in (s.get('gaps_ms') or []) if g > STALL_MS) / 1000.0
    wall_total = sum(s.get('wall_s') or 0.0 for s in streams)

    result = dict(
        users=len(streams),
        ttft_s=[round(v, 2) for v in ttfts],
        ttft_deltas_s=[round(v, 2) for v in deltas],
        ttft_spread_s=round(ttfts[-1] - ttfts[0], 2) if ttfts else None,
        prefill_interval_s=round(interval, 2) if interval else None,
        prefill_serial=serial,
        round_s=round(round_s, 3) if round_s else None,
        first_gap_s=[None if v is None else round(v, 2) for v in first_gaps],
        predicted_stall_s=[None if v is None else round(v, 2) for v in predicted],
        stall_explained_by_admission=explained,
        stall_total_s=round(stall_total, 1),
        wall_total_s=round(wall_total, 1),
        stall_share=round(stall_total / wall_total, 4) if wall_total else None,
        synchronised_stalls=synchronised_stalls(streams),
    )

    failures = []
    ceiling = _threshold(environ, FLAG_MAX_STALL)
    if ceiling is not None and stall_total > ceiling:
        failures.append('total decode stall %.1f s exceeds %s=%.1f' % (stall_total, FLAG_MAX_STALL, ceiling))
    worst = _threshold(environ, FLAG_MAX_TTFT)
    if worst is not None and ttfts and ttfts[-1] > worst:
        failures.append('worst TTFT %.1f s exceeds %s=%.1f' % (ttfts[-1], FLAG_MAX_TTFT, worst))
    result['thresholds_checked'] = ceiling is not None or worst is not None
    result['failures'] = failures
    return result


def render(result):
    lines = ['TTFT profile: %d users' % result['users'],
             '  TTFT            %s  (spread %.1f s)' % (
                 ' '.join('%.1f' % v for v in result['ttft_s']), result['ttft_spread_s'] or 0.0)]
    if result['ttft_deltas_s']:
        lines.append('  deltas          %s  -> prefill %s at %.1f s each' % (
            ' '.join('%.1f' % v for v in result['ttft_deltas_s']),
            'SERIAL' if result['prefill_serial'] else 'not serial',
            result['prefill_interval_s'] or 0.0))
    lines.append('  first gap       %s' % ' '.join(
        '-' if v is None else '%.1f' % v for v in result['first_gap_s']))
    lines.append('  predicted       %s  -> %s' % (
        ' '.join('-' if v is None else '%.1f' % v for v in result['predicted_stall_s']),
        'explained by admission order' if result['stall_explained_by_admission']
        else 'NOT explained by admission order'))
    lines.append('  stall           %.1f s of %.1f s wall (%.1f%%)' % (
        result['stall_total_s'], result['wall_total_s'], 100.0 * (result['stall_share'] or 0.0)))
    for entry in result['synchronised_stalls']:
        lines.append('  SYNCHRONISED    round %d: %d users, %s s' % (
            entry['round'], entry['users'], ' '.join('%.1f' % v for v in entry['seconds'])))
    for failure in result['failures']:
        lines.append('  FAIL            %s' % failure)
    return '\n'.join(lines)
