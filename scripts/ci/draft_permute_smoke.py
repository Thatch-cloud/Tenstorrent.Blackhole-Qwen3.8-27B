"""The smoke rule of WP7's drafter permutation levers (draft_permute_tp: QWEN_FAST_DRAFT_PERMUTE and its audit), in the style of c2_smoke_check.spread_problems. Stdlib only, host side.

    problems(env, container_text) -> [problem]

`env` is the served profile's environment (a dict of strings, or None), `container_text` the server log. The parent smoke check calls this once per arm. What it holds:

  - without the flag: none of the lever's lines (engaged, audit, fell back) may appear - a log line from a lever the profile did not ask for is a profile that engaged it by another route;
  - with the flag: an engaged line for each of the three sites at each shape the profile serves - K/V (`site=kv`), fold and unfold at `shape=quad` when QWEN_FAST_QUAD_DRAFT=1, and `site=kv shape=octo`
    when QWEN_FAST_OCTO_DRAFT=1 (the octo fold and unfold are not hooked: octo_draft_tp is not WP7's file). The pair shapes are reported when they ran and demanded when no quad serves;
  - with the audit flag: a passing line (`audit exact=True`) for each quad site the engaged lines name (the audit compares in the quad bucket's eager warm pass);
  - on any arm: no fall-back line (a fall-back is the lever not running: the arm measured the old composition) and no audit mismatch line.
"""

import re

FLAG = 'QWEN_FAST_DRAFT_PERMUTE'
AUDIT_FLAG = 'QWEN_FAST_DRAFT_PERMUTE_AUDIT'
QUAD_FLAG = 'QWEN_FAST_QUAD_DRAFT'
OCTO_FLAG = 'QWEN_FAST_OCTO_DRAFT'

ENGAGED = '[PINDIAG] tp4 draft permute engaged'
FELL_BACK = '[PINDIAG] tp4 draft permute fell back'
AUDIT = '[PINDIAG] tp4 draft permute audit'
MISMATCH = '[PINDIAG] tp4 draft permute audit mismatch'
SITES = ('kv', 'fold', 'unfold')


def _on(env, name):
    return (env or {}).get(name) == '1'


def _site(line):
    match = re.search(r' site=(kv|fold|unfold)\b', line)
    shape = re.search(r' shape=([a-z-]+)\b', line)
    return (match.group(1) if match else None), (shape.group(1) if shape else None)


def problems(env, container_text):
    lines = [line for line in container_text.splitlines() if 'tp4 draft permute' in line]
    engaged = [line for line in lines if ENGAGED in line]
    fell_back = [line for line in lines if FELL_BACK in line]
    mismatch = [line for line in lines if MISMATCH in line]
    audited = [line for line in lines if AUDIT in line and MISMATCH not in line]
    found = ['the drafter permutation fell back (the served ops ran, it saved nothing): %s' % line.strip()[:200] for line in fell_back][:4]
    found += ['the drafter permutation audit found a difference: %s' % line.strip()[:200] for line in mismatch][:4]
    found += ['a permutation audit line that is not exact: %s' % line.strip()[:200] for line in audited if ' exact=True' not in line][:4]
    if not _on(env, FLAG):
        if lines:
            found.append('drafter permutation lines on a profile without %s' % FLAG)
        if _on(env, AUDIT_FLAG):
            found.append('%s=1 without %s=1' % (AUDIT_FLAG, FLAG))
        return found
    seen = set(_site(line) for line in engaged)
    wanted = []
    if _on(env, QUAD_FLAG):
        wanted += [(site, 'quad') for site in SITES]
    if _on(env, OCTO_FLAG):
        wanted.append(('kv', 'octo'))
    if not wanted:
        wanted = [('kv', None)]
    for site, shape in wanted:
        if not any(found_site == site and (shape is None or found_shape == shape) for found_site, found_shape in seen):
            found.append('%s=1 is set and no engaged line (%s site=%s%s) was logged: that site never ran as a launch' % (
                FLAG, ENGAGED, site, (' shape=%s' % shape) if shape else ''))
    if _on(env, AUDIT_FLAG):
        passing = set(_site(line) for line in audited if ' exact=True' in line)
        for site, shape in wanted:
            if shape != 'quad':
                continue
            if (site, 'quad') not in passing:
                found.append('%s=1 is set and no passing audit line (%s exact=True site=%s shape=quad) was logged: nothing was compared' % (AUDIT_FLAG, AUDIT, site))
    return found
