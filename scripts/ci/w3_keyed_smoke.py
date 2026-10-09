"""Smoke rules for the keyed staging diff (tp4/w3 1b, QWEN_FAST_TP4_PRESTAGE_KEYED). Host side, stdlib only: the smoke runs these on a gate arm's
server log, beside c2_smoke_check's other per-lever rules (the parent wires `problems` in; this module has no import of it).

`problems(env, container_text)` answers a list of strings, empty when the log is what the profile asked for:

  * a profile WITHOUT the flag logs none of the lever's text: the engaged line, a ` keyed=1` on a verify line (the pre-stage's
    [PACKED-PRESTAGE], the full audit's [PACKED-PRESTAGE-FULLAUDIT], the host-gap [PACKED-HOSTGAP-VERIFY] ` keyed=1`), the keyed audit line;
  * a keyed audit line with keyed_mismatches other than 0 is a problem whatever the profile says (the lever's exactness proof failed);
  * a profile WITH the flag logs the engaged line (per block; its absence means the lever never ran: it needs QWEN_FAST_PRESTAGE=1, which the
    image sets, and a pre-staged block);
  * the share of keyed rounds among the `path=diff` verify lines: the design expects about 75% of 4-live blocks keyed (a user crosses a
    64-token page about every 14.7 rounds at tau 4.35), so with at least MIN_DIFF_ROUNDS diff rounds logged, KEYED_MIN_SHARE (30%) of them
    must carry keyed=1 - below it the lever engaged and almost never applied; fewer diff rounds than that are not judged;
  * an audited arm (QWEN_FAST_PRESTAGE_AUDIT=1 or QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT=1) logs at least one keyed audit line, and its keyed
    full-audit lines (path=diff keyed=1) carry mismatches=0.
"""

import re

FLAG = 'QWEN_FAST_TP4_PRESTAGE_KEYED'
AUDIT_FLAGS = ('QWEN_FAST_PRESTAGE_AUDIT', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT')
ENGAGED = '[PINDIAG] verify prestage keyed engaged'
KEYED_AUDIT = '[PACKED-PRESTAGE-KEYED-AUDIT]'
FULL_AUDIT = '[PACKED-PRESTAGE-FULLAUDIT]'
PRESTAGE_LINE = re.compile(r'\[PACKED-PRESTAGE\] round=\d+ path=(\w+) buffers=\d+ reason=\S+ live=\d+( keyed=1)?')
KEYED_AUDIT_LINE = re.compile(r'\[PACKED-PRESTAGE-KEYED-AUDIT\] round=\d+ checked=(\d+) keyed_mismatches=(\d+)')
FULL_AUDIT_LINE = re.compile(r'\[PACKED-PRESTAGE-FULLAUDIT\] block=\S+ round=\d+ path=(\w+) buffers=\d+ checked=\d+ mismatches=(\d+)'
                             r'(?: at=\S+)?( keyed=1)?')
HOSTGAP_VERIFY_KEYED = re.compile(r'\[PACKED-HOSTGAP-VERIFY\] .* keyed=([01])\s*$', re.MULTILINE)
MIN_DIFF_ROUNDS = 20
KEYED_MIN_SHARE = 0.30


def problems(env, container_text):
    env = env or {}
    text = container_text or ''
    asked = env.get(FLAG) == '1'
    found = []
    rounds = PRESTAGE_LINE.findall(text)
    keyed_rounds = [item for item in rounds if item[1]]
    audits = KEYED_AUDIT_LINE.findall(text)
    full_keyed = [item for item in FULL_AUDIT_LINE.findall(text) if item[2]]
    for checked, mismatches in audits:
        if int(mismatches):
            found.append('the keyed staging diff audit found a difference: %s' % next(
                line.strip()[:200] for line in text.splitlines() if KEYED_AUDIT in line and 'keyed_mismatches=%s' % mismatches in line))
            break
    for path, mismatches, keyed in full_keyed:
        if int(mismatches):
            found.append('the full pre-stage audit found a difference on a keyed round (%s %s)' % (FULL_AUDIT, path))
            break
    if not asked:
        if env.get(FLAG, '0') not in ('0', '1'):
            found.append('%s=%r is not 0 or 1' % (FLAG, env.get(FLAG)))
        if ENGAGED in text or keyed_rounds or audits or full_keyed or HOSTGAP_VERIFY_KEYED.search(text):
            found.append('keyed staging diff lines (%s, keyed=1 or %s) on a profile without %s' % (ENGAGED, KEYED_AUDIT, FLAG))
        return found
    if ENGAGED not in text:
        found.append('%s is set and no engaged line (%s) was logged: the keyed diff never ran (it needs QWEN_FAST_PRESTAGE=1 and a pre-staged block)'
                     % (FLAG, ENGAGED))
    diff = [item for item in rounds if item[0] == 'diff']
    if len(diff) >= MIN_DIFF_ROUNDS and len(keyed_rounds) < KEYED_MIN_SHARE * len(diff):
        found.append('%s is set and only %d of %d diff verifies were keyed (%d%% needed; the design expects about 75%% of 4-live blocks): the key '
                     'almost never matched' % (FLAG, len(keyed_rounds), len(diff), int(KEYED_MIN_SHARE * 100)))
    if any(env.get(name) == '1' for name in AUDIT_FLAGS) and not audits:
        found.append('%s is set on an audited arm and no keyed audit line (%s round=N checked=K keyed_mismatches=0) was logged: nothing was compared'
                     % (FLAG, KEYED_AUDIT))
    return found
