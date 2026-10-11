"""The smoke rule of exact resume P0 (docs/tp4-exact-resume.md 5.2; QWEN_PREFIX_STICKY_UNDROP, QWEN_PREFIX_WINDOW_CARRY,
QWEN_PREFIX_WINDOW_CARRY_AUDIT), in the style of tp4_shard_argmax_smoke. Stdlib only, host side.

    problems(env, container_text) -> [problem]

`env` is the served profile's environment (a dict of strings, or None), `container_text` the server log. The parent smoke check calls this
once per arm. A smoke sends no cache_salt, so nothing is ever carried in it: what the smoke can hold is the install, and that nothing ran
that should not have:
  - without the switches: no window carry line of any kind, and neither switch's companions set;
  - with them: the scheduler graft's engaged line ('[PINDIAG] prefix: window carry engaged undrop=<0|1> carry=<0|1> audit=<0|1> ...') naming
    exactly the profile's switches, the carry only beside the undrop, the audit only beside the carry, and sticky sessions with prefix
    reuse on (the graft serves the fast path's hit only);
  - on any arm: no audit line that says exact=False (a carried window that did not read back as it was stored).
The carried admissions, the window exports and the cold-twin window comparison are the prefix gate's (c2_prefix_gate.exact_resume_findings).
"""

import re

UNDROP = 'QWEN_PREFIX_STICKY_UNDROP'
CARRY = 'QWEN_PREFIX_WINDOW_CARRY'
AUDIT = 'QWEN_PREFIX_WINDOW_CARRY_AUDIT'
REUSE = 'QWEN_PREFIX_REUSE'
STICKY = 'QWEN_FAST_STICKY_SESSIONS'
MARKER = '[PINDIAG] prefix: window carry'
ENGAGED = MARKER + ' engaged'
AUDIT_LINE = MARKER + ' audit'
FIELD = re.compile(r'(undrop|carry|audit)=(\d)')


def _on(env, name):
    return (env or {}).get(name) == '1'


def problems(env, container_text):
    lines = [line for line in container_text.splitlines() if MARKER in line]
    found = []
    for line in lines:
        if AUDIT_LINE in line and 'exact=False' in line:
            found.append('a carried drafter window did not read back as it was stored: %s' % line.strip()[:200])
    undrop, carry, audit = _on(env, UNDROP), _on(env, CARRY), _on(env, AUDIT)
    if not (undrop or carry or audit):
        if lines:
            found.append('window carry lines on a profile without %s or %s: %s' % (UNDROP, CARRY, lines[0].strip()[:200]))
        return found
    if carry and not undrop:
        found.append('%s=1 without %s=1' % (CARRY, UNDROP))
    if audit and not carry:
        found.append('%s=1 without %s=1' % (AUDIT, CARRY))
    if not (_on(env, REUSE) and _on(env, STICKY)):
        found.append('the exact-resume switches need %s=1 and %s=1 (the graft serves the fast path\'s sticky hit only)' % (REUSE, STICKY))
    engaged = [line for line in lines if ENGAGED in line]
    if not engaged:
        found.append('the exact-resume switches are set and no engaged line (%s) was logged: the scheduler graft did not take them' % ENGAGED)
        return found
    fields = dict(FIELD.findall(engaged[-1]))
    want = dict(undrop=str(int(undrop)), carry=str(int(carry)), audit=str(int(audit)))
    if fields != want:
        found.append('the engaged line says %s, the profile %s' % (
            ' '.join('%s=%s' % item for item in sorted(fields.items())), ' '.join('%s=%s' % item for item in sorted(want.items()))))
    return found
