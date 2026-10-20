"""The ship-5 admission chain as a timing model, and what the three tp4/sched-2 levers take out of it (docs/tp4-sched2.md). Stdlib only.

    python3 -B scripts/ci/sched2_model.py            the tables of the doc
    python3 -B scripts/ci/sched2_model.py --check    exit 1 unless the eager replay reproduces the measured burst (the calibration the tests hold)

EVERY FIGURE IN THIS FILE BEFORE THE `LEVERS` TABLES IS MEASURED: the 93 admissions of the ship-5 timed arm (four cards, Swift-1.5, the container log of the T1B job; read with
scripts/ci/sched2_report.py), the first eight of them being the eight fresh prompts of ~3.5k tokens that the `concurrent8_steady` test sends together into a fresh process. The lever
effects are ESTIMATES: arithmetic over those figures under the assumptions each function names, never a card result.

THE CHAIN. For arrival k the scheduler's hold reads the allocator (`hold`), the prefill runs (`prefill`, a synchronous device call: the host waits in it), the sampler drains the last of
the device's work (`drain`), and the request's engine is bound (`bind` = A the slot adoption and host checks + B the placement and window join + C the rebind). Then the decoders are owed
rounds (`gap`), and the next arrival starts. The first token of arrival k leaves after its own bind (eager) and the decode of the others waits in `gap`.

  `bracket`   the part of `bind` that is the memory ledger's before and after readings: paid at a slot's FIRST rebind only (the ship-5 candidate's lean admission keeps the first of each slot)
  `readback`  the part of `bind` that is the conv-slice readback: paid at a native slot's FIRST adoption only

So the first eight arrivals into a fresh process pay both eight times (every slot is new); an arrival into a warm process pays neither.
"""

import argparse
import sys

# One arrival: hold, prefill, drain, bind, bracket (inside bind), readback (inside bind), gap (the decode rounds after it, before the next hold), measured first-token time (s).
BURST = (
    (116, 662, 217, 497, 249, 0, 18, 1.51),
    (121, 784, 2, 736, 269, 139, 1215, 3.17),
    (118, 682, 2, 622, 226, 129, 522, 5.82),
    (123, 685, 2, 640, 218, 140, 969, 7.79),
    (92, 680, 2, 556, 176, 114, 426, 10.09),
    (91, 666, 2, 557, 176, 109, 763, 11.83),
    (90, 617, 1, 598, 173, 155, 238, 13.90),
    (104, 612, 1, 522, 174, 82, 0, 15.39),
)
FIELDS = ('hold', 'prefill', 'drain', 'bind', 'bracket', 'readback', 'gap', 'ttft')

# A returning turn into a warm process (78 of the 93): p50 of each segment, ms. The first token of such a turn leaves at p50 1.46 s (p90 2.33 s) in the same run.
WARM = dict(hold=84, prefill=690, drain=13, bind=275, a=65, b=11, c=181, rezero=30, project=91, history=15, engine=29, bracket=10, readback=0)
WARM_TTFT_P50_MS, WARM_TTFT_P90_MS = 1460.0, 2330.0

# The ship-5 TTFT of the burst above, last first token (s): the measured one.
MEASURED_LAST_S = 15.39

# HOLD_REUSE (admission_pipeline): one real reading then up to this many reuses (the default QWEN_FAST_ADMISSION_HOLD_REUSE_MAX_REUSES).
HOLD_REUSES = 3


def arrivals():
    return [dict(zip(FIELDS, row)) for row in BURST]


def hold_hit(index, reuses=HOLD_REUSES):
    """Whether arrival `index` (0-based) reuses the reading: the first reads, then `reuses` reuse it, then one reads again. The model assumes the reading clears the terms by the wide
    margin of this run (4.6 GB largest free block against about 1 GB needed) and that the arrivals are closer together than the reading's age limit (10 s)."""
    return index % (reuses + 1) != 0


def replay(levers=(), reuses=HOLD_REUSES, rows=None):
    """The burst's first-token times (s) with `levers` among 'seed', 'hold', 'ledger': arithmetic over the measured segments. The decode gaps are taken as measured: the owed-rounds
    cap (QWEN_FAST_LEVERN_MAX_ROUNDS x the measured round time) binds in this run, so a shorter step does not shorten what the decoders are owed (amplified() is the case where it does not).
    'ledger' takes `bracket` out of the bind; 'hold' takes the hold out of every arrival that reuses a reading; 'seed' takes the (trimmed) bind out of the arrival's OWN first-token time
    only, the bind still runs before the next arrival's hold, so the cycle is the eager one."""
    levers = set(levers)
    unknown = levers - {'seed', 'hold', 'ledger'}
    if unknown:
        raise ValueError('unknown levers %s' % sorted(unknown))
    out, clock = [], 0.0
    for index, row in enumerate(arrivals() if rows is None else rows):
        hold = 0 if 'hold' in levers and hold_hit(index, reuses) else row['hold']
        bind = row['bind'] - (row['bracket'] if 'ledger' in levers else 0)
        first = hold + row['prefill'] + row['drain'] + (0 if 'seed' in levers else bind)
        out.append((clock + first) / 1000.0)
        clock += hold + row['prefill'] + row['drain'] + bind + row['gap']
    return out


def amplified(levers=(), reuses=HOLD_REUSES):
    """UPPER BOUND: the same replay when the owed decode rounds follow the step time one for one (dt (1 - f) / f, at any share f) instead of binding at their cap, so that every
    millisecond taken out of the chain also comes out of the rounds the decoders are owed. Not what this run did (its gaps are capped); the bound on what a run whose cap does not
    bind could see. The gap after arrival k is the measured one scaled by the new step time over the old one."""
    levers = set(levers)
    out, clock = [], 0.0
    for index, row in enumerate(arrivals()):
        hold = 0 if 'hold' in levers and hold_hit(index, reuses) else row['hold']
        bind = row['bind'] - (row['bracket'] if 'ledger' in levers else 0)
        step = hold + row['prefill'] + row['drain'] + bind
        old = row['hold'] + row['prefill'] + row['drain'] + row['bind']
        gap = row['gap'] * step / old
        first = hold + row['prefill'] + row['drain'] + (0 if 'seed' in levers else bind)
        out.append((clock + first) / 1000.0)
        clock += step + gap
    return out


def returning(levers=(), reuses=HOLD_REUSES):
    """A returning turn into a warm process: (the first-token time p50, ms) with the levers. Warm slots pay no bracket and no readback, so 'ledger' changes nothing; 'hold' takes the
    hold out of the share of turns that reuse a reading (reuses of reuses + 1, when the turns arrive within the reading's age: the back-to-back case); 'seed' takes the bind out."""
    levers = set(levers)
    saved = 0.0
    if 'hold' in levers:
        saved += WARM['hold'] * reuses / (reuses + 1)
    if 'seed' in levers:
        saved += WARM['bind']
    return WARM_TTFT_P50_MS - saved


def serial_chain(row):
    """What is serial between one arrival's prefill and the next arrival's: its drain and its bind, and the next arrival's hold."""
    return row['drain'] + row['bind']


def tables():
    out = []
    rows = arrivals()
    total = sum(row['hold'] + row['prefill'] + row['drain'] + row['bind'] + row['gap'] for row in rows[:-1]) + sum(rows[-1][key] for key in ('hold', 'prefill', 'drain', 'bind'))
    parts = [('prefill (the device)', sum(row['prefill'] for row in rows)), ('the decoders\' owed rounds (the gaps)', sum(row['gap'] for row in rows)),
             ('the scheduler\'s hold (allocator reading)', sum(row['hold'] for row in rows)), ('the drain', sum(row['drain'] for row in rows)),
             ('the bind, the rebind itself and its host checks', sum(row['bind'] - row['bracket'] - row['readback'] for row in rows)),
             ('the bind: the ledger bracket (first rebind of each slot)', sum(row['bracket'] for row in rows)),
             ('the bind: the conv-slice readback (first adoption of each slot)', sum(row['readback'] for row in rows))]
    out.append('WHERE THE 8-USER BURST GOES (measured, ship 5, eight fresh ~3.5k prompts into a fresh process; last first token %.2f s)' % MEASURED_LAST_S)
    out.append('%-66s %9s %7s' % ('', 'ms', 'share'))
    for name, value in parts:
        out.append('%-66s %9d %6.1f%%' % (name, value, 100.0 * value / total))
    out.append('')
    out.append('A WARM RETURNING TURN (measured p50, ms): hold %(hold)d + prefill %(prefill)d + drain %(drain)d + bind %(bind)d (A %(a)d, B %(b)d, C %(c)d: re-zero %(rezero)d, projection %(project)d, '
               'K/V history %(history)d, engine %(engine)d); first token p50 %%d' % WARM % int(WARM_TTFT_P50_MS))
    out.append('')
    out.append('LEVERS (ESTIMATES: arithmetic over the measured segments)')
    out.append('%-34s %12s %10s %12s %10s' % ('', 'burst last s', 'vs eager', 'warm turn ms', 'vs eager'))
    base_last, base_warm = replay()[-1], returning()
    for name, levers in (('eager (ship 5)', ()), ('ledger trim', ('ledger',)), ('hold reuse', ('hold',)), ('seed first', ('seed',)), ('hold + ledger', ('hold', 'ledger')),
                         ('all three', ('hold', 'ledger', 'seed'))):
        last, warm = replay(levers)[-1], returning(levers)
        out.append('%-34s %12.2f %9.1f%% %12.0f %9.1f%%' % (name, last, 100.0 * (last - base_last) / base_last, warm, 100.0 * (warm - base_warm) / base_warm))
    out.append('%-34s %12.2f %9.1f%%' % ('all three, owed rounds not capped', amplified(('hold', 'ledger', 'seed'))[-1], 100.0 * (amplified(('hold', 'ledger', 'seed'))[-1] - amplified()[-1]) / amplified()[-1]))
    return out


def check():
    """Problems with the calibration, [] when the eager replay reproduces the measured burst to 2% at every arrival."""
    problems = []
    model = replay()
    for index, (row, got) in enumerate(zip(arrivals(), model)):
        if abs(got - row['ttft']) > 0.02 * row['ttft']:
            problems.append('arrival %d: the replay says %.2f s, the run measured %.2f s' % (index + 1, got, row['ttft']))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args(argv)
    if arguments.check:
        problems = check()
        for problem in problems:
            sys.stderr.write(problem + '\n')
        return 1 if problems else 0
    print('\n'.join(tables()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
