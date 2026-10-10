"""The smoke rule of the ccl options lever (WP5, QWEN_FAST_CCL_OPTIONS): what a container log must show for a profile that sets it.

    problems(env, container_text) -> [str]   (an empty list is a clean log)

The integrator wires this into c2_smoke_check.lever_engagement_problems beside u1_problems (scripts/ci/fusion-wp/WP5.json says how). The rule, in the
order a reader checks it:

  1. A profile without the flag logs no ccl options line at all (a lever that engages by itself is a finding).
  2. A profile with the flag logs at least one engaged line, every one naming the profile's own set (canonical form), a reduce-scatter count above 0
     when the set names reduce-scatter options (rs= is the number of unit-major calls that carried the set: 128 a 64-row forward) and a gather count
     above 0 when the set wraps the gather (ag= the norm gathers that reached the shim: 128 or 129 a forward; a set of reduce-scatter options alone does
     not wrap it, ccl_options_tp.gather_active). The shim wraps every decode norm gather of the block, so a profile that wraps and engages nothing never bound it.
  3. No fall-back line: a call that left the census ran with the model's own values, so the set saved nothing there. The reason is in the line.
  4. No audit mismatch line, ever.
  5. Under QWEN_FAST_CCL_OPTIONS_AUDIT=1 an exact=True audit line for every op the set names (the gather always, the reduce-scatter when it has
     reduce-scatter options) at shape 64x5120 from a REPLAY (round >= 1), four chips, elements above 0, from at least one block owner per
     QWEN_FAST_M3_BLOCKS (the audit is only valid right after its own block's replay).
"""

import re

import ccl_options_tp

SERVED_SHAPE = '64x5120'
CHIPS = 4
_ENGAGED = re.compile(r' set=(\S+) rows=(\d+) rs=(\d+) ag=(\d+) fallbacks=(\d+) audited=(\d+)')
_AUDIT = re.compile(r' shape=(\d+x\d+) owner=(\S+) round=(\d+) op=(rs|ag) calls=(\d+) chips=(\d+) elements=(\d+) exact=True')


def audit_problems(env, lines, selection):
    problems = []
    parsed = []
    for line in lines:
        if ccl_options_tp.AUDIT_MARKER not in line or ccl_options_tp.AUDIT_MISMATCH_MARKER in line:
            continue
        match = _AUDIT.search(line)
        if match is None:
            problems.append('a ccl options audit line of an unknown form: %s' % line.strip()[:200])
            continue
        shape, owner, round_number, op, calls, chips, elements = match.groups()
        if int(chips) != CHIPS or int(elements) < 1 or int(calls) < 1:
            problems.append('a ccl options audit line compared nothing (need chips=%d, calls and elements above 0): %s' % (CHIPS, line.strip()[:200]))
            continue
        parsed.append((op, shape, owner, int(round_number)))
    blocks = env.get('QWEN_FAST_M3_BLOCKS', '1')
    wanted = int(blocks) if blocks.isdigit() and int(blocks) > 0 else 1
    for op in (['rs'] if selection.rs else []) + (['ag'] if ccl_options_tp.gather_active(selection) else []):
        owners = set(owner for found_op, shape, owner, round_number in parsed if found_op == op and shape == SERVED_SHAPE and round_number >= 1)
        if not owners:
            problems.append('%s=1 and no replay audit line (%s shape=%s ... round>=1 op=%s ... exact=True) was logged: no replay was compared'
                            % (ccl_options_tp.AUDIT_FLAG, ccl_options_tp.AUDIT_MARKER, SERVED_SHAPE, op))
        elif len(owners) < wanted:
            problems.append('%s: op=%s replay audits came from %d block owner(s), %d blocks are served'
                            % (ccl_options_tp.AUDIT_FLAG, op, len(owners), wanted))
    return problems


def problems(env, container_text):
    lines = container_text.splitlines()
    found = ['the ccl options audit found a difference: %s' % line.strip()[:200]
             for line in lines if ccl_options_tp.AUDIT_MISMATCH_MARKER in line][:4]
    found += ['a call left the ccl options census (it ran with the model\'s own values): %s' % line.strip()[:200]
              for line in lines if ccl_options_tp.FALLBACK_MARKER in line][:4]
    text = None if env is None else env.get(ccl_options_tp.OPTIONS_FLAG)
    if text is None or text == '0':
        if any(ccl_options_tp.ENGAGED_MARKER in line or ccl_options_tp.AUDIT_MARKER in line for line in lines):
            found.append('ccl options lines on a profile without %s' % ccl_options_tp.OPTIONS_FLAG)
        return found
    try:
        selection = ccl_options_tp.parse_set(text)
    except ValueError as error:
        return found + ['the profile\'s %s is not a named set: %s' % (ccl_options_tp.OPTIONS_FLAG, error)]
    engaged = [line for line in lines if ccl_options_tp.ENGAGED_MARKER in line]
    if not engaged:
        return found + ['%s=%s and no engaged line (%s) was logged: the options were never applied' % (ccl_options_tp.OPTIONS_FLAG, text, ccl_options_tp.ENGAGED_MARKER)]
    for line in engaged:
        match = _ENGAGED.search(line)
        if match is None:
            found.append('an engaged line of an unknown form: %s' % line.strip()[:200])
            continue
        name, rows, rs, ag, _fallbacks, _audited = match.groups()
        if name != selection.name:
            found.append('the engaged line names set %s, the profile\'s is %s' % (name, selection.name))
        if selection.rs and int(rs) < 1:
            found.append('the set names reduce-scatter options and the engaged line counts none: %s' % line.strip()[:200])
        if ccl_options_tp.gather_active(selection) and int(ag) < 1:
            found.append('the engaged line counts no gather (the wrapper never saw a norm): %s' % line.strip()[:200])
    if env.get(ccl_options_tp.AUDIT_FLAG) == '1':
        found += audit_problems(env, lines, selection)
    return found
