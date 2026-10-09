"""The smoke rule of W3's MR and 2a levers (tp4_mesh_read), in the style of c2_smoke_check.spread_problems. Stdlib only, host side.

    problems(env, container_text) -> [problem]

`env` is the served profile's environment (a dict of strings, or None), `container_text` the server log. The parent smoke check calls this once
per arm. What it holds, per lever:

  - without the flag: none of the lever's lines (engaged, audit, hostgap `mr=1`) may appear - a log line from a lever the profile did not ask for
    is a profile that engaged it by another route;
  - with the flag: at least one engaged line (MR: `chips=4`; 2a: one per quad block, two when QWEN_FAST_QUAD_DRAFT_BLOCKS=2), and under
    QWEN_FAST_TP4_VERIFY_MESH_READ=1 with the hostgap log on, the verify lines carry `mr=1` (their reads_ms came from the mesh read);
  - with the audit flag: at least one audit line, every audit line `equal=1`;
  - on any arm: no fall-back line (a fall-back is the lever not running: the arm measured the old read) and no `equal=0` line.
"""

import re

MESH_READ_FLAG = 'QWEN_FAST_TP4_VERIFY_MESH_READ'
MESH_READ_AUDIT_FLAG = 'QWEN_FAST_TP4_VERIFY_MESH_READ_AUDIT'
JOINED_FLAG = 'QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED'
JOINED_AUDIT_FLAG = 'QWEN_FAST_TP4_QUAD_OUTPUTS_JOINED_AUDIT'
QUAD_BLOCKS_FLAG = 'QWEN_FAST_QUAD_DRAFT_BLOCKS'
HOSTGAP_LOG_FLAG = 'QWEN_FAST_TP4_HOSTGAP_LOG'

MESH_READ_ENGAGED = '[PINDIAG] tp4 verify mesh read engaged'
MESH_READ_FELL_BACK = '[PINDIAG] tp4 verify mesh read fell back'
MESH_READ_AUDIT = '[PACKED-MESH-READ-AUDIT]'
JOINED_ENGAGED = '[PINDIAG] tp4 quad outputs joined engaged'
JOINED_FELL_BACK = '[PINDIAG] tp4 quad outputs joined fell back'
JOINED_AUDIT = '[QUAD-JOINED-AUDIT]'
HOSTGAP_VERIFY = '[PACKED-HOSTGAP-VERIFY]'


def _on(env, name):
    return (env or {}).get(name) == '1'


def _audit_problems(lines, marker, label):
    found = [line for line in lines if marker in line]
    bad = [line for line in found if ' equal=1' not in line]
    return found, ['%s found a difference: %s' % (label, line.strip()[:200]) for line in bad][:4]


def problems(env, container_text):
    lines = container_text.splitlines()
    found = []
    for fell_back, label in ((MESH_READ_FELL_BACK, 'the verify mesh read'), (JOINED_FELL_BACK, 'the joined quad outputs')):
        found += ['%s fell back (the old read ran, it saved nothing): %s' % (label, line.strip()[:200]) for line in lines if fell_back in line][:4]
    mesh_audit_lines, mesh_audit_bad = _audit_problems(lines, MESH_READ_AUDIT, 'the verify mesh-read audit')
    joined_audit_lines, joined_audit_bad = _audit_problems(lines, JOINED_AUDIT, 'the joined quad-output audit')
    found += mesh_audit_bad + joined_audit_bad

    mesh_on, joined_on = _on(env, MESH_READ_FLAG), _on(env, JOINED_FLAG)
    mesh_engaged = [line for line in lines if MESH_READ_ENGAGED in line]
    joined_engaged = [line for line in lines if JOINED_ENGAGED in line]
    mr_verify = [line for line in lines if HOSTGAP_VERIFY in line and re.search(r' mr=1( |$)', line)]
    verify_lines = [line for line in lines if HOSTGAP_VERIFY in line]

    if not mesh_on:
        if mesh_engaged or mesh_audit_lines or mr_verify:
            found.append('verify mesh-read lines on a profile without %s' % MESH_READ_FLAG)
    else:
        if not any(re.search(r' chips=4( |$)', line) for line in mesh_engaged):
            found.append('%s=1 is set and no engaged line (%s chips=4) was logged: the verify read never came from the mesh' % (
                MESH_READ_FLAG, MESH_READ_ENGAGED))
        if _on(env, HOSTGAP_LOG_FLAG) and verify_lines and not mr_verify:
            found.append('%s=1 and the hostgap log is on, and no %s line carries mr=1: reads_ms is the loop\'s' % (MESH_READ_FLAG, HOSTGAP_VERIFY))
    if _on(env, MESH_READ_AUDIT_FLAG) and mesh_on and not any(' equal=1' in line for line in mesh_audit_lines):
        found.append('%s=1 is set and no passing audit line (%s ... equal=1) was logged: nothing was compared' % (MESH_READ_AUDIT_FLAG, MESH_READ_AUDIT))
    if not mesh_on and _on(env, MESH_READ_AUDIT_FLAG):
        found.append('%s=1 without %s=1' % (MESH_READ_AUDIT_FLAG, MESH_READ_FLAG))

    if not joined_on:
        if joined_engaged or joined_audit_lines:
            found.append('joined quad-output lines on a profile without %s' % JOINED_FLAG)
    else:
        wanted = 2 if (env or {}).get(QUAD_BLOCKS_FLAG) == '2' else 1
        groups = set()
        for line in joined_engaged:
            match = re.search(r' slots=\[([0-9,]*)\]', line)
            groups.add(match.group(1) if match else line)
        if len(groups) < wanted:
            found.append('%s=1 is set and %d engaged line(s) (%s) were logged for %d quad block(s): a quad never joined its outputs' % (
                JOINED_FLAG, len(groups), JOINED_ENGAGED, wanted))
    if _on(env, JOINED_AUDIT_FLAG) and joined_on and not any(' equal=1' in line for line in joined_audit_lines):
        found.append('%s=1 is set and no passing audit line (%s ... equal=1) was logged: nothing was compared' % (JOINED_AUDIT_FLAG, JOINED_AUDIT))
    if not joined_on and _on(env, JOINED_AUDIT_FLAG):
        found.append('%s=1 without %s=1' % (JOINED_AUDIT_FLAG, JOINED_FLAG))
    return found
