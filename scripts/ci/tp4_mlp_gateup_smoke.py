"""The smoke rules of the WP4 MLP levers (tp4_mlp_gateup): what a gate arm's container log must hold for the profile's environment (host side; c2_smoke_check calls
`problems(env, container_text)` once per arm beside fusion_problems, which already holds the generic rules: an engaged line for a lever the profile sets, a fell-back
line fails, an audit flag needs a line with exact=True).

This module adds the route-specific ones the generic table cannot say:

  * no [PINDIAG] tp4 mlp gateup line at all when neither lever is on (a lever that engaged by accident is a timing that says nothing about the control);
  * the engaged line names the route and the canonical configuration the profile asked for (`route=cfg name=g3u3d4`, `route=fused name=p3`), at the 64-row block, and
    over every layer of the model (`layers=64`): an engaged line for another name is another configuration's timing;
  * an audited arm's passing lines name the same route and configuration, and a mismatch line (or any `exact=False`) fails whatever else is in the log.

Stdlib only; py 3.7.
"""

import tp4_mlp_gateup as lever

PREFIX = '[PINDIAG] tp4 mlp gateup'
LAYERS = 64
ROWS = 64


def problems(env, container_text):
    """[str]: why `container_text` does not show the levers `env` asks for (empty = clean)."""
    out = []
    try:
        selection = lever.resolve(env)
    except ValueError as error:
        return ['the MLP lever flags in the profile are not a configuration the lever accepts: %s' % error]
    lines = [line.strip() for line in container_text.splitlines() if PREFIX in line]
    if selection.route is None:
        return ['a %s line was logged but no MLP lever is on: %s' % (PREFIX, line[:160]) for line in lines[:4]]
    key = 'route=%s name=%s ' % (selection.route, selection.name)
    engaged = [line for line in lines if lever.ENGAGED in line]
    out += ['the MLP lever fell back (the served ops ran, it saved nothing): %s' % line[:200] for line in lines if lever.FALLBACK in line][:4]
    wanted = [line for line in engaged if key in line + ' ']
    if not wanted:
        out.append('no engaged line for %s (the profile asks for route=%s name=%s)%s' % (
            PREFIX, selection.route, selection.name,
            '; engaged lines seen: %s' % '; '.join(line[:120] for line in engaged[:3]) if engaged else ''))
    else:
        line = wanted[0] + ' '
        if ' rows=%d ' % ROWS not in line:
            out.append('the engaged line is not the %d-row block: %s' % (ROWS, wanted[0][:160]))
        if ' layers=%d ' % LAYERS not in line:
            out.append('the engaged line does not cover all %d layers: %s' % (LAYERS, wanted[0][:160]))
        if ' l1_multiply=1 ' not in line:
            out.append('the engaged line does not say the multiply was written to L1: %s' % wanted[0][:160])
    audits = [line for line in lines if line.startswith(lever.AUDIT_MARKER) or (lever.AUDIT_MARKER in line and lever.AUDIT_MISMATCH not in line)]
    mismatches = [line for line in lines if lever.AUDIT_MISMATCH in line or 'exact=False' in line]
    out += ['the MLP audit found a difference: %s' % line[:200] for line in mismatches[:4]]
    if selection.audit:
        passing = [line for line in audits if 'exact=True' in line and key in line + ' ']
        if not passing:
            out.append('the audit flag is on and no passing audit line for route=%s name=%s was logged (the lever was never compared with the served composition)'
                       % (selection.route, selection.name))
    elif audits:
        out.append('an audit line was logged with no audit flag on: %s' % audits[0][:160])
    return out
