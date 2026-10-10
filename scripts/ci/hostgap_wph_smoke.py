"""The smoke rule of the host-gap package WPH (prestage_diff, write_packed_lean, batched_reads_tp, hostgap_instr), in the style of c2_smoke_check.spread_problems. Stdlib only, host side.

    problems(env, container_text) -> [problem]

`env` is the served profile's environment (a dict of strings, or None), `container_text` the server log. The parent smoke check calls this once per arm. What it holds, per lever:

  - WITHOUT the flag: none of the lever's lines (engaged, audit, fell back, its per-round line) may appear, and an audit flag without its lever is refused;
  - WITH the flag: its engaged line; no fell-back line and no audit mismatch line (a fall-back is the lever not running: the arm measured the served path);
  - PRESTAGE_DIFF (QWEN_FAST_PRESTAGE_DIFF): at least one `[PACKED-PRESTAGE-DIFF] ... path=diff` line that wrote fewer destinations than it pre-staged (a diff path that wrote
    everything saved nothing), and, audited, at least one passing audit line (`exact=True`) and every audit line passing;
  - WRITE_PACKED_LEAN: an engaged line for the pre-stage site and for the verify site; audited, a passing audit line;
  - BATCHED_READS: an engaged line for the verify site, and for the collect site when the quads serve (QWEN_FAST_QUAD_DRAFT=1); audited, a passing audit line for each;
  - the instrument (QWEN_FAST_TP4_HOSTGAP_LOG): its engaged line, at least one round line and one span line; with QWEN_FAST_TP4_HOSTGAP_PROBE a probe line once there are
    PROBE_MIN_ROUNDS round lines; and no probe line without the probe flag.
"""

import re

DIFF, DIFF_AUDIT = 'QWEN_FAST_PRESTAGE_DIFF', 'QWEN_FAST_PRESTAGE_DIFF_AUDIT'
LEAN, LEAN_AUDIT = 'QWEN_FAST_WRITE_PACKED_LEAN', 'QWEN_FAST_WRITE_PACKED_LEAN_AUDIT'
READS, READS_AUDIT = 'QWEN_FAST_BATCHED_READS', 'QWEN_FAST_BATCHED_READS_AUDIT'
LOG, PROBE = 'QWEN_FAST_TP4_HOSTGAP_LOG', 'QWEN_FAST_TP4_HOSTGAP_PROBE'
QUAD = 'QWEN_FAST_QUAD_DRAFT'
PROBE_MIN_ROUNDS = 40

LEVERS = (
    # (name, flag, audit flag, line prefix of the lever, engaged, fell back, audit, what)
    ('prestage diff', DIFF, DIFF_AUDIT, '[PINDIAG] tp4 prestage diff', '[PINDIAG] tp4 prestage diff engaged', '[PINDIAG] tp4 prestage diff fell back',
     '[PINDIAG] tp4 prestage diff audit', 'the pre-stage diff'),
    ('write packed lean', LEAN, LEAN_AUDIT, '[PINDIAG] tp4 write packed lean', '[PINDIAG] tp4 write packed lean engaged', '[PINDIAG] tp4 write packed lean fell back',
     '[PINDIAG] tp4 write packed lean audit', 'the lean write_packed'),
    ('batched reads', READS, READS_AUDIT, '[PINDIAG] tp4 batched reads', '[PINDIAG] tp4 batched reads engaged', '[PINDIAG] tp4 batched reads fell back',
     '[PINDIAG] tp4 batched reads audit', 'the batched read-backs'),
)


def _on(env, name):
    value = (env or {}).get(name, '0')
    return value not in ('', '0')


def _lines(text, prefix):
    return [line for line in text.splitlines() if prefix in line]


def problems(env, container_text):
    found = []
    for name, flag, audit_flag, prefix, engaged, fell_back, audit, what in LEVERS:
        lines = _lines(container_text, prefix)
        found += ['%s fell back (the served path ran, it saved nothing): %s' % (what, line.strip()[:200]) for line in lines if fell_back in line][:4]
        mismatched = [line for line in lines if audit + ' mismatch' in line]
        found += ['the audit of %s found a difference: %s' % (what, line.strip()[:200]) for line in mismatched][:4]
        audited = [line for line in lines if audit in line and audit + ' mismatch' not in line]
        found += ['an audit line of %s that is not exact: %s' % (what, line.strip()[:200]) for line in audited if ' exact=True' not in line][:4]
        if not _on(env, flag):
            if lines:
                found.append('lines of %s on a profile without %s' % (what, flag))
            if _on(env, audit_flag):
                found.append('%s is set without %s' % (audit_flag, flag))
            continue
        if not any(engaged in line for line in lines):
            found.append('%s is set and no engaged line (%s) was logged: %s never ran' % (flag, engaged, what))
        if _on(env, audit_flag) and not any(' exact=True' in line for line in audited):
            found.append('%s is set and no passing audit line (%s ... exact=True) was logged: %s was never audited' % (audit_flag, audit, what))
    on = lambda flag: _on(env, flag)
    if on(DIFF):
        rounds = [re.search(r'path=(\w+) .*total=(\d+) written=(\d+)', line) for line in _lines(container_text, '[PACKED-PRESTAGE-DIFF]')]
        useful = [match for match in rounds if match and match.group(1) == 'diff' and int(match.group(3)) < int(match.group(2))]
        if not useful:
            found.append('%s is set and no [PACKED-PRESTAGE-DIFF] line took the diff path and wrote fewer destinations than it pre-staged: the lever saved nothing' % DIFF)
    if on(LEAN):
        for site in ('prestage', 'verify'):
            if not any('write packed lean engaged site=%s' % site in line for line in container_text.splitlines()):
                found.append('%s is set and no engaged line for site=%s was logged: that write never went through the lean writer' % (LEAN, site))
    if on(READS):
        sites = ['verify'] + (['collect'] if (env or {}).get(QUAD) == '1' else [])
        for site in sites:
            if not any('batched reads engaged site=%s' % site in line for line in container_text.splitlines()):
                found.append('%s is set and no engaged line for site=%s was logged: that read-back never went through the batched reader' % (READS, site))
        if on(READS_AUDIT):
            for site in sites:
                if not any('batched reads audit site=%s' % site in line and ' exact=True' in line for line in container_text.splitlines()):
                    found.append('%s is set and no passing audit line for site=%s was logged' % (READS_AUDIT, site))
    rounds = _lines(container_text, '[PACKED-HOSTGAP-ROUND]')
    probes = _lines(container_text, '[PACKED-HOSTGAP-PROBE]')
    if on(LOG):
        if not any('[PINDIAG] tp4 hostgap instrument engaged' in line for line in container_text.splitlines()):
            found.append('%s is set and no instrument engaged line was logged: the host calls were never timed' % LOG)
        if not rounds or not _lines(container_text, '[PACKED-HOSTGAP-SPAN]'):
            found.append('%s is set and the log holds no round line and span line of the instrument' % LOG)
        if on(PROBE) and len(rounds) >= PROBE_MIN_ROUNDS and not probes:
            found.append('%s is set and %d rounds ran without one probe line' % (PROBE, len(rounds)))
    elif rounds or _lines(container_text, '[PACKED-HOSTGAP-SPAN]') or probes:
        found.append('instrument lines on a profile without %s' % LOG)
    if probes and not on(PROBE):
        found.append('probe lines on a profile without %s' % PROBE)
    return found
