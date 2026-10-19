"""The WC smoke rule (QWEN_FAST_TP4_GDN_WC, gdn_wc): read from a gated arm's server log by the parent's c2_smoke_check (`problems(env, container_text)`, the shape of
c2_smoke_check.spread_problems). Stdlib only; the markers are gdn_wc's, held equal by test_tp4_w3_wc.

Three rules. A fall-back line fails the smoke on a profile with or without the flag (the V1 chain ran: the timing is not the lever's); a profile that asks for
the lever and logs no engaged line (with a conv, a gate and a z core) fails; an audit flag with no passing 'exact=True' audit line, or any mismatch line,
fails. Lines on a profile that did not ask for the lever fail too.
"""

import re

FLAG = 'QWEN_FAST_TP4_GDN_WC'
AUDIT_FLAG = 'QWEN_FAST_TP4_GDN_WC_AUDIT'
ENGAGED = '[PINDIAG] tp4 gdn wc engaged'
FELL_BACK = '[PINDIAG] tp4 gdn wc fell back'
AUDIT = '[PINDIAG] tp4 gdn wc audit'
MISMATCH = '[PINDIAG] tp4 gdn wc audit mismatch'
OCTO_DECLINE = 'reason=8 users (WC takes 2 or 4)'    # gdn_wc.problem() for the octo block's 8 users


def problems(env, container_text):
    """[problem] for the WC launch: any mismatch or fall-back line; with the flag, the engaged line (conv_cores, gate_cores and z_cores at least 1) and, with the
    audit flag, a passing audit line (the marker, '<n> exact=True')."""
    lines = container_text.splitlines()
    found = ['the gdn wc audit found a difference: %s' % line.strip()[:200] for line in lines if MISMATCH in line][:4]
    # The octo block (QWEN_FAST_OCTO=live|alternate: 8 users of 8 rows) is never a WC block: WC takes 2 or 4 users, declines it with this one line, and the block
    # runs the served chain. That decline is the design, not a fault; every other fall-back still fails the arm.
    octo = env is not None and env.get('QWEN_FAST_OCTO') in ('live', 'alternate')
    found += ['the gdn wc launch fell back (the V1 chain ran, it saved nothing): %s' % line.strip()[:200] for line in lines
              if FELL_BACK in line and not (octo and OCTO_DECLINE in line)][:4]
    if env is None or env.get(FLAG) != '1':
        if any(ENGAGED in line or AUDIT in line for line in lines):
            found.append('gdn wc lines on a profile without %s' % FLAG)
        return found
    engaged = [line for line in lines if ENGAGED in line]
    if not engaged:
        found.append('%s is set and no engaged line (%s) was logged: the wc launch never ran' % (FLAG, ENGAGED))
    else:
        for field in ('conv_cores', 'gate_cores', 'z_cores'):
            if not any(re.search(r' %s=[1-9]\d*( |$)' % field, line) for line in engaged):
                found.append('%s engaged lines report no %s: %s' % (FLAG, field, engaged[0].strip()[:200]))
    if env.get(AUDIT_FLAG) == '1' and not any(AUDIT in line and 'exact=True' in line and MISMATCH not in line for line in lines):
        found.append('%s=1 is set and no passing audit line (%s <n> exact=True) was logged: nothing was compared' % (AUDIT_FLAG, AUDIT))
    return found
