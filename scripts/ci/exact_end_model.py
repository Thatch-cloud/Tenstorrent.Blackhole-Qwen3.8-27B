"""Exact-end resume: what a returning agent turn re-prefills under each resume policy, on the engine's own chunk grid (docs/tp4-exact-end.md).

Stdlib only, no device, no flag: this is arithmetic the doc's tables and the CPU tests share, so that a number in the doc can be recomputed
and the claim "a byte-exact resume runs the cold prefill's own programs" can be checked as a statement about program lists.

THE GRID (all of it read from the image's model code, scripts/ci/fixtures/qwen36_model.py, and the fast path):
  * A prefill of P tokens is a sequence of programs, each (chunk_start, bucket, valid): every full chunk of 2048 rows is bucket 2048 with
    valid 2048 (_prefill_chunked_eager_tp: "valid_len == bucket == chunk_size"), then the tail t = P mod 2048 is ONE masked bucket, the
    smallest of (128, 256, 512, 1024, 2048) that holds it (_mask_bucket_for, _PREFILL_MASK_BUCKETS), valid = t.
  * A program's cost is set by its BUCKET (the padded rows are computed), not by how many rows are valid: a tail of 1,047 rows and a tail of
    2,047 rows both run the 2048-row program.
  * The matmul program config is chosen by the bucket (graft/model_config.py prefill_progcfg(seq_len, k, n)), so two programs of different
    buckets are different arithmetic. That is why a byte-exact resume has to run the cold prefill's own programs on the cold prefill's own bytes.
  * Lever N's final step starts at S_last = floor2048(P) - 2048 when P has a tail (levern_policy.final_start), which is why today's resume
    is floor2048(P_prev) - 2048; exact resume P0 (undrop + window carry) resumes at floor2048(P_prev) instead.

POLICIES (resume position Q of the next prompt, given the previous prompt length):
  today       floor2048(P_prev) - 2048   (sticky sessions; prefix_judge.Oracle(sticky=True) agrees; the production Q=161,792 for P_prev=164,887)
  p0          floor2048(P_prev)          (exact resume P0; Oracle(sticky=True, undrop=True, carry=True)); byte-exact: the cold prefill's own tail
  exact_end   floor128(P_prev)           (the idealised "prefill only the new tokens": a 128-aligned resume; NOT byte-exact, docs/tp4-exact-end.md 2-3)
  fine        the last complete piece of a prefix-stable tail grid (Option C, a different cold baseline: docs/tp4-exact-end.md 5)

COST (modelled, not measured on the production stack): forward_seconds() interpolates the card medians of docs/tp4-exact-resume.md 4.2
(final tail forward plus end-of-prefill, T=128 and T=2048 at 32k / 128k / 254k; one full 2048 forward by context) and adds the Lever N step
setup (0.2 s, calibrated there). Use it to compare policies, not to predict a TTFT.

    python3 -B scripts/ci/exact_end_model.py --start 164887 --delta 169 --turns 8
    python3 -B scripts/ci/exact_end_model.py --simulate 400 --delta 170
"""

import argparse
import collections
import random
import sys

CHUNK = 2048
BLOCK = 64
SUBCHUNK = 128
BUCKETS = (128, 256, 512, 1024, 2048)
STEP_SETUP_S = 0.2

Program = collections.namedtuple('Program', 'chunk_start bucket valid')

# docs/tp4-exact-resume.md 4.2 (measured medians, solo). (context, final tail forward T=128, final tail forward T=2048), in seconds.
FINAL_FORWARD = ((32768, 0.36, 0.53), (131072, 0.49, 0.83), (253952, 0.71, 1.34))
# One full 2048-row forward that is not the last (context, seconds): 0.31 at 0-16k, 0.42 at 16-48k, 0.58 at 48-96k, 0.71 at 96-160k, 1.03 at 160-224k, 1.20 above.
CHUNK_FORWARD = ((8192, 0.31), (32768, 0.42), (72000, 0.58), (128000, 0.71), (192000, 1.03), (240000, 1.20))

# Option C: a prefix-stable tail grid. The cell [floor2048(P), P) is cut at FIXED offsets (a piece never depends on P), so every completed piece's
# state is a state any longer prompt of the same text also passes through.
FINE_PIECES = (1024, 512, 256, 128, 128)


def floor_to(value, multiple):
    return (int(value) // multiple) * multiple


def ceil_to(value, multiple):
    return -(-int(value) // multiple) * multiple


def bucket_for(rows):
    """The smallest fixed bucket that holds `rows` (qwen36_model._mask_bucket_for)."""
    if type(rows) is not int or rows < 1:
        raise ValueError('a positive integer row count is required, got %r' % (rows,))
    for bucket in BUCKETS:
        if rows <= bucket:
            return bucket
    return ceil_to(rows, SUBCHUNK)


def cold_programs(prompt, start=0):
    """The programs a prefill of `prompt` tokens runs for rows [start, prompt) on the engine's grid. start is 0 or a CHUNK multiple below the
    prompt: a resume anywhere else is refused (every layer of the engine refuses it)."""
    if type(prompt) is not int or prompt < 1:
        raise ValueError('a positive integer prompt length is required, got %r' % (prompt,))
    if type(start) is not int or not 0 <= start < prompt or start % CHUNK:
        raise ValueError('a resume starts at 0 or at a %d-token boundary below the prompt, got start=%r prompt=%r' % (CHUNK, start, prompt))
    full = prompt // CHUNK
    programs = [Program(index * CHUNK, CHUNK, CHUNK) for index in range(start // CHUNK, full)]
    tail = prompt - full * CHUNK
    if tail:
        programs.append(Program(full * CHUNK, bucket_for(tail), tail))
    return programs


def split_programs(prompt, resume):
    """The programs of a prefill that resumes at a 128-aligned position `resume` inside a 2048 cell (the probe's B arms, tp4_split_probe): the head
    segment [resume, min(prompt, ceil2048(resume))) as its own masked bucket at chunk_start=resume, then the grid again from the cell boundary."""
    if type(resume) is not int or not 0 < resume < prompt or resume % SUBCHUNK:
        raise ValueError('a split resume is a multiple of %d inside the prompt, got %r for %r' % (SUBCHUNK, resume, prompt))
    if resume % CHUNK == 0:
        return cold_programs(prompt, resume)
    head_end = min(prompt, ceil_to(resume, CHUNK))
    programs = [Program(resume, bucket_for(head_end - resume), head_end - resume)]
    if head_end < prompt:
        programs += cold_programs(prompt, head_end)
    return programs


def fine_cuts(rows):
    """The piece boundaries inside one cell, as (offset, size): (0, 1024), (1024, 512), (1536, 256), (1792, 128), (1920, 128)."""
    cuts, offset = [], 0
    for size in FINE_PIECES:
        cuts.append((offset, size))
        offset += size
    return cuts


def fine_programs(prompt, start=0):
    """Option C's programs for rows [start, prompt): full chunks, then the tail cell cut at the fixed offsets (the last piece masked). start is 0
    or a boundary of that grid (a CHUNK multiple or a completed piece)."""
    if type(prompt) is not int or prompt < 1:
        raise ValueError('a positive integer prompt length is required, got %r' % (prompt,))
    full = prompt // CHUNK
    tail = prompt - full * CHUNK
    programs = [Program(index * CHUNK, CHUNK, CHUNK) for index in range(0, full)]
    for offset, size in fine_cuts(tail):
        if offset >= tail:
            break
        programs.append(Program(full * CHUNK + offset, size, min(size, tail - offset)))
    kept = [program for program in programs if program.chunk_start >= start]
    if not kept or kept[0].chunk_start != start and start != 0:
        raise ValueError('start=%r is not a boundary of the fine grid for %r' % (start, prompt))
    return kept


def fine_resume(prompt_prev, prompt=None):
    """The last boundary of the fine grid the previous prompt completed: the cell start, or the end of its last COMPLETE piece. A prompt that
    has moved into a later cell sees the earlier cell as one full chunk, so only that cell's start is on its grid."""
    full = prompt_prev // CHUNK
    tail = prompt_prev - full * CHUNK
    best = full * CHUNK
    if prompt is not None and prompt // CHUNK > full:
        return best
    for offset, size in fine_cuts(tail):
        if offset + size <= tail:
            best = full * CHUNK + offset + size
    return best


def resume_today(prompt_prev):
    return max(0, floor_to(prompt_prev, CHUNK) - CHUNK)


def resume_p0(prompt_prev):
    return floor_to(prompt_prev, CHUNK)


def resume_exact_end(prompt_prev):
    return floor_to(prompt_prev, SUBCHUNK)


def interpolate(table, x, column):
    """Piecewise-linear value of table[...][column] at x, clamped at the ends (the first column is x)."""
    if x <= table[0][0]:
        return table[0][column]
    for (x0, *low), (x1, *high) in zip(table, table[1:]):
        if x <= x1:
            fraction = (x - x0) / float(x1 - x0)
            return low[column - 1] + (high[column - 1] - low[column - 1]) * fraction
    return table[-1][column]


def final_forward_seconds(bucket, context):
    """The last program of a prefill (its logits, host snapshot and slot write included) at `context` tokens: linear in the bucket between the
    T=128 and T=2048 medians (the T=1024 medians, 0.42 and 0.73 s at 32k and 128k, sit 0.01-0.08 s above the line)."""
    low = interpolate(FINAL_FORWARD, context, 1)
    high = interpolate(FINAL_FORWARD, context, 2)
    return low + (high - low) * (bucket - BUCKETS[0]) / float(BUCKETS[-1] - BUCKETS[0])


def chunk_forward_seconds(context):
    return interpolate(CHUNK_FORWARD, context, 1)


def seconds_of(programs, prompt):
    """Modelled prefill seconds of a program list for `prompt`: every program but the last is a full chunk at its own context (its end), the last
    is the final forward, plus one Lever N step setup (the whole resumed prefill is one final step)."""
    if not programs:
        return 0.0
    total = STEP_SETUP_S
    for program in programs[:-1]:
        total += chunk_forward_seconds(program.chunk_start + program.valid) if program.bucket == CHUNK else final_forward_seconds(program.bucket, prompt)
    total += final_forward_seconds(programs[-1].bucket, prompt)
    return total


POLICIES = collections.OrderedDict([
    ('today', lambda prev, prompt: (resume_today(prev), cold_programs(prompt, resume_today(prev)) if resume_today(prev) else cold_programs(prompt))),
    ('p0', lambda prev, prompt: (resume_p0(prev), cold_programs(prompt, resume_p0(prev)) if resume_p0(prev) else cold_programs(prompt))),
    ('exact_end', lambda prev, prompt: (resume_exact_end(prev), split_programs(prompt, resume_exact_end(prev)) if resume_exact_end(prev) else cold_programs(prompt))),
    ('fine', lambda prev, prompt: (fine_resume(prev, prompt), fine_programs(prompt, fine_resume(prev, prompt)))),
])


def evaluate(prompt_prev, prompt, policy):
    """dict(q, programs, rows, forwards, old, new, seconds) of the returning turn `prompt` that follows a turn of `prompt_prev` tokens."""
    if not 0 < prompt_prev < prompt:
        raise ValueError('a returning prompt extends the previous one: %r after %r' % (prompt, prompt_prev))
    q, programs = POLICIES[policy](prompt_prev, prompt)
    return dict(policy=policy, prompt=prompt, q=q, programs=programs, rows=sum(program.bucket for program in programs), forwards=len(programs),
                old=max(0, prompt_prev - q), new=prompt - max(prompt_prev, q), seconds=seconds_of(programs, prompt))


def grow(start, delta, turns):
    return [start + delta * index for index in range(turns)]


def session_table(start, delta, turns, policies=('today', 'p0', 'exact_end', 'fine')):
    """{policy: [evaluate(...) for each returning turn]} for a session that starts at `start` tokens and grows by `delta` a turn."""
    prompts = grow(start, delta, turns)
    return dict((policy, [evaluate(prev, prompt, policy) for prev, prompt in zip(prompts, prompts[1:])]) for policy in policies)


def simulate(sessions, delta, seed=20261019, turns=40, low=20000, high=250000):
    """Totals over `sessions` random sessions (start uniform in [low, high), `turns` returning turns of `delta` tokens each): {policy: dict(...)}."""
    rng = random.Random(seed)
    totals = dict((policy, dict(turns=0, old=0, rows=0, forwards=0, seconds=0.0)) for policy in POLICIES)
    for _ in range(sessions):
        start = rng.randrange(low, high)
        for policy, results in session_table(start, delta, turns + 1).items():
            for item in results:
                bucket = totals[policy]
                bucket['turns'] += 1
                bucket['old'] += item['old']
                bucket['rows'] += item['rows']
                bucket['forwards'] += item['forwards']
                bucket['seconds'] += item['seconds']
    return totals


def format_session(table):
    lines = []
    for policy, results in table.items():
        lines.append('%s' % policy)
        for item in results:
            lines.append('  P=%d Q=%d old=%d new=%d programs=%s rows=%d forwards=%d modelled=%.2fs' % (
                item['prompt'], item['q'], item['old'], item['new'], ' '.join('(%d,%d,%d)' % tuple(p) for p in item['programs']),
                item['rows'], item['forwards'], item['seconds']))
    return '\n'.join(lines)


def format_totals(totals):
    base = totals['today']
    lines = ['policy      turns  old/turn  rows/turn  forwards/turn  modelled s/turn  vs today (seconds)']
    for policy, item in totals.items():
        turns = float(item['turns'])
        lines.append('%-10s %6d %9.0f %10.0f %14.2f %16.2f %12.0f%%' % (
            policy, item['turns'], item['old'] / turns, item['rows'] / turns, item['forwards'] / turns, item['seconds'] / turns,
            100.0 * (item['seconds'] / base['seconds'] - 1.0)))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--start', type=int, default=164887, help='the first prompt of the session')
    parser.add_argument('--delta', type=int, default=169, help='tokens added per turn')
    parser.add_argument('--turns', type=int, default=8, help='prompts in the session (turns - 1 returning turns)')
    parser.add_argument('--simulate', type=int, default=0, help='random sessions to total instead of printing one session')
    arguments = parser.parse_args(argv)
    if arguments.simulate:
        print(format_totals(simulate(arguments.simulate, arguments.delta)))
    else:
        print(format_session(session_table(arguments.start, arguments.delta, arguments.turns)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
