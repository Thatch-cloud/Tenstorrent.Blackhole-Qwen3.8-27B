"""What an octo-T8 server logs (QWEN_FAST_OCTO, gate only), parsed. Stdlib only, Python 3.7 syntax: the gate reads this on the rig host.

Every line format below is its producer's own (each constant names its source; test_octo_judge renders the producers' real lines and parses them
here), and every field is read by name, so a field a producer adds never hides a line. A missing marker is reported by octo_judge, never guessed
around: a mounted change is not an executed change, so what proves the octo block ran is a line its round wrote AFTER its own verify counter moved.

  serving_octo.admission (attach):
    [OCTO] admitted mode=<live|alternate> rows=8 users=8 min_live=<n> (gate only)
    [OCTO] UNQUALIFIED (gate only) (<i>/<n>): <a piece nothing has qualified yet>
    [OCTO] refused: <reason>                                      one per reason; the attach then raises
  serving_packed_step.PackedStep (per round, written after the round ran; octo block configured only):
    [OCTO] round=<n> shape=<octo|m3|seq> planned=<octo|m3> live=<k> rows=<w> eligible=<0|1> counted=<0|1> octo_rounds=<a> m3_rounds=<b>
           committed=<tokens> step_ms=<f> gap_ms=<f>
        shape     where the round RAN: octo when the octo block's own round counter moved, m3 when a 4-user block's did, else seq.
        planned   the shape the drafts were made for (the policy's last plan).
        eligible  1 when the policy found the octo block able to serve the live set (live >= min_live, every member inside the block's rules).
        counted   1 when eligible and the round ran as planned: the rounds an alternation is judged on.
        committed the tokens the round emitted over every user; step_ms the step's wall time; gap_ms the wall time from the previous step's end.
    [OCTO] programs shape=<octo|m3|seq> round=<n> switch=<0|1> before=<a> after=<b>
        written on the first round of a shape after a switch (and the first round of the process): the mesh's program-cache entry count before
        and after the round. `None` when the count is unreadable. after > before is a program compiled after warmup.

scan() returns the facts; octo_judge turns them into what a gate expects of an arm that ran an octo profile (or of one that did not: every [OCTO]
marker is then a leak).
"""
import re

MARKER = '[OCTO]'
ADMITTED = MARKER + ' admitted '
UNQUALIFIED = MARKER + ' UNQUALIFIED (gate only)'
REFUSED = MARKER + ' refused'
ROUND = MARKER + ' round='
PROGRAMS = MARKER + ' programs '

SHAPES = ('octo', 'm3', 'seq')

ROUND_LINE = ('[OCTO] round={round} shape={shape} planned={planned} live={live} rows={rows} eligible={eligible} counted={counted} '
              'octo_rounds={octo_rounds} m3_rounds={m3_rounds} committed={committed} step_ms={step_ms:.1f} gap_ms={gap_ms:.1f}')
PROGRAMS_LINE = '[OCTO] programs shape={shape} round={round} switch={switch} before={before} after={after}'
ADMITTED_LINE = '[OCTO] admitted mode={mode} rows={rows} users={users} min_live={min_live} (gate only)'

ROUND_PATTERN = re.compile(
    r'\[OCTO\] round=(\d+) shape=(octo|m3|seq) planned=(octo|m3) live=(\d+) rows=(\d+) eligible=([01]) counted=([01]) '
    r'octo_rounds=(\d+) m3_rounds=(\d+) committed=(\d+) step_ms=([0-9.]+) gap_ms=([0-9.]+)')
PROGRAMS_PATTERN = re.compile(r'\[OCTO\] programs shape=(octo|m3|seq) round=(\d+) switch=([01]) before=(\d+|None) after=(\d+|None)')
ADMITTED_PATTERN = re.compile(r'\[OCTO\] admitted mode=(live|alternate) rows=(\d+) users=(\d+) min_live=(\d+)')


def rounds(text):
    """[dict] of every round line, in log order."""
    found = []
    for match in ROUND_PATTERN.finditer(text or ''):
        (number, shape, planned, live, rows, eligible, counted, octo_rounds, m3_rounds, committed, step_ms, gap_ms) = match.groups()
        found.append(dict(round=int(number), shape=shape, planned=planned, live=int(live), rows=int(rows), eligible=int(eligible),
                          counted=int(counted), octo_rounds=int(octo_rounds), m3_rounds=int(m3_rounds), committed=int(committed),
                          step_ms=float(step_ms), gap_ms=float(gap_ms)))
    return found


def programs(text):
    """[dict] of every programs line, in log order; before/after are ints or None."""
    found = []
    for match in PROGRAMS_PATTERN.finditer(text or ''):
        shape, number, switch, before, after = match.groups()
        found.append(dict(shape=shape, round=int(number), switch=int(switch),
                          before=None if before == 'None' else int(before), after=None if after == 'None' else int(after)))
    return found


def admissions(text):
    """[dict] of every admitted line."""
    return [dict(mode=mode, rows=int(rows), users=int(users), min_live=int(min_live))
            for mode, rows, users, min_live in ADMITTED_PATTERN.findall(text or '')]


def marker_lines(text):
    """Every line of the log that carries an [OCTO] marker."""
    return [line for line in (text or '').splitlines() if MARKER in line]


def scan(text):
    """The facts of one container log."""
    return dict(admitted=admissions(text), rounds=rounds(text), programs=programs(text),
                unqualified=sum(1 for line in (text or '').splitlines() if UNQUALIFIED in line),
                refused=sum(1 for line in (text or '').splitlines() if REFUSED in line),
                lines=len(marker_lines(text)))
