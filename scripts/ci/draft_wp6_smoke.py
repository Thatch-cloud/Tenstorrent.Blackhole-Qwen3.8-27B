"""What a gated arm must see in the server log from the drafter fusion levers of tp4/fx-wp6 (QWEN_FAST_DRAFT_REDUCE, _TAIL, _GATEUP1, _MM_GRID and their _AUDIT
twins), as functions of the profile's env and the log. Stdlib and the markers' table only, py 3.7 syntax: the gate reads it on the rig host.

A mounted lever is not an executed one: a profile can carry the flag and run every chain on the served ops (the hook never reached, the shape
refused, the launch failed) and its texts will still equal the control's. So, per lever (draft_fusion_tp.TABLE):

  flag on     at least one `engaged` line (the proof the launch ran); NO `fell back` line (a fall-back means the served ops ran, so the timing is not
              the lever's) and NO `audit mismatch` line;
  audit on    (the audit flag; the lever flag must be on too) at least one `audit <n> exact=True` line: an audited arm must have audited;
  flag off    not one engaged, fell-back, audit or mismatch line of that lever (a leak means the lever ran where nobody asked for it).

The REDUCE engaged line carries chains=<n> (draft_reduce_tp.MILESTONES: 1, 5 and 10 chains run). A profile whose chains never reach the count the hooks give
(5 per quad once the MLP branch's chains run; 10 once the attention branch's hook is in too) has half of its chains still on the served ops:
`chains_problems` reports it when the caller says how many to expect (--expect-chains).

main() reads one log and one profile env from the command line:
    python3 draft_wp6_smoke.py --log container.log [--profile NAME [--profiles qwen_c2_profiles.json]] [--env KEY=VALUE ...] [--expect-chains N]
and exits 1 on any problem, printing each.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# (lever flag, audit flag, engaged, fell back, audit, mismatch, what): the same rows as draft_fusion_tp.TABLE (test_draft_wp6_smoke holds them
# equal); repeated here so that the rig host reads this file alone.
TABLE = (('QWEN_FAST_DRAFT_REDUCE', 'QWEN_FAST_DRAFT_REDUCE_AUDIT', '[PINDIAG] tp4 draft reduce engaged',
          '[PINDIAG] tp4 draft reduce fell back', '[PINDIAG] tp4 draft reduce audit', '[PINDIAG] tp4 draft reduce audit mismatch',
          'drafter gather-add reduce'),
         ('QWEN_FAST_DRAFT_TAIL', 'QWEN_FAST_DRAFT_TAIL_AUDIT', '[PINDIAG] tp4 draft tail engaged',
          '[PINDIAG] tp4 draft tail fell back', '[PINDIAG] tp4 draft tail audit', '[PINDIAG] tp4 draft tail audit mismatch',
          'drafter SwiGLU and residual kernels'),
         ('QWEN_FAST_DRAFT_GATEUP1', 'QWEN_FAST_DRAFT_GATEUP1_AUDIT', '[PINDIAG] tp4 draft gateup1 engaged',
          '[PINDIAG] tp4 draft gateup1 fell back', '[PINDIAG] tp4 draft gateup1 audit', '[PINDIAG] tp4 draft gateup1 audit mismatch',
          'drafter fused gate|up matmul'),
         ('QWEN_FAST_DRAFT_MM_GRID', 'QWEN_FAST_DRAFT_MM_GRID_AUDIT', '[PINDIAG] tp4 draft mmgrid engaged',
          '[PINDIAG] tp4 draft mmgrid fell back', '[PINDIAG] tp4 draft mmgrid audit', '[PINDIAG] tp4 draft mmgrid audit mismatch',
          'drafter matmul grids'))
REDUCE_FLAG = TABLE[0][0]
CHAINS = re.compile(r'\[PINDIAG\] tp4 draft reduce engaged chains=(\d+)')
EXACT = re.compile(r'audit (\d+) exact=True')


def _on(env, name):
    return (env or {}).get(name) == '1'


def chains_run(text):
    """The most chains any REDUCE engaged line reports (0 when there is none)."""
    return max([int(found) for found in CHAINS.findall(text or '')] or [0])


def problems(env, container_text):
    """[problem] for the profile `env` (flag name -> value) and the server log `container_text`. Empty when every lever the profile asks for ran
    and was audited as asked, and no lever it did not ask for left a line."""
    found = []
    lines = (container_text or '').splitlines()
    for flag, audit_flag, engaged, fell_back, audit_marker, mismatch, what in TABLE:
        on, audited = _on(env, flag), _on(env, audit_flag)
        fell = [line.strip()[:200] for line in lines if fell_back in line]
        wrong = [line.strip()[:200] for line in lines if mismatch in line]
        proofs = [line for line in lines if engaged in line]
        exact = [line for line in lines if audit_marker in line and mismatch not in line and EXACT.search(line)]
        if audited and not on:
            found.append('%s is set without %s: the audit would compare nothing' % (audit_flag, flag))
        if on:
            found += ['the %s fell back (the served ops ran, it saved nothing): %s' % (what, line) for line in fell[:4]]
            found += ['the %s audit found a difference: %s' % (what, line) for line in wrong[:4]]
            if not proofs:
                found.append('%s is set and no engaged line (%s) was logged: the %s never ran' % (flag, engaged, what))
            if audited and not exact:
                found.append('%s is set and no passing audit line (%s <n> exact=True) was logged: the %s was never audited'
                             % (audit_flag, audit_marker, what))
        else:
            leaked = [line.strip()[:200] for line in lines if any(marker in line for marker in (engaged, fell_back, audit_marker, mismatch))]
            found += ['%s is not set and the log holds a %s line: the lever ran where nobody asked for it: %s' % (flag, what, line)
                      for line in leaked[:2]]
    return found


def chains_problems(env, container_text, expected):
    """[problem] when REDUCE is on and fewer than `expected` chains ever ran (half of them still served: a branch's hook is missing)."""
    if not _on(env, REDUCE_FLAG) or expected is None:
        return []
    ran = chains_run(container_text)
    if ran < expected:
        return ['%s is set and only %d of the expected %d chains ran on the launch (the engaged line reports chains=%d at most)'
                % (REDUCE_FLAG, ran, expected, ran)]
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--log', required=True)
    parser.add_argument('--profile')
    parser.add_argument('--profiles', default=str(Path(__file__).with_name('qwen_c2_profiles.json')))
    parser.add_argument('--env', action='append', default=[], help='KEY=VALUE, added to (and overriding) the profile env')
    parser.add_argument('--expect-chains', type=int)
    options = parser.parse_args(argv)
    env = {}
    if options.profile:
        with open(options.profiles) as handle:
            env.update(json.load(handle)['profiles'][options.profile].get('env', {}))
    for item in options.env:
        key, _, value = item.partition('=')
        env[key] = value
    text = Path(options.log).read_text(errors='replace')
    found = problems(env, text) + chains_problems(env, text, options.expect_chains)
    for line in found:
        print('PROBLEM: %s' % line)
    if not found:
        print('draft_wp6_smoke: clean (%d lines)' % len(text.splitlines()))
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
