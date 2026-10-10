"""What a gated arm must see in the server log from the 13 x 10 matmul-grid lever of tp4/m1-grid13 (QWEN_FAST_GRID13 and QWEN_FAST_GRID13_AUDIT), as a function of the profile's
env and the log. Stdlib only, py 3.7 syntax: the gate reads it on the rig host. A host-side rule (scripts/ci/c2_smoke_check.fusion_problems calls problems()), not an image file.

A mounted lever is not an executed one: a profile can carry the flag and run every matmul on the served grid (the install hook never reached, a census refusal, a narrower
device) and its texts will still equal the control's. So, by the sub-switches the flag names (1 is all three):

  prefill   an `engaged` line for each of the three fused all-gather matmul sites: site=gdn_in, site=attn_in and the MLP's (site=gate_up for the fused SwiGLU, or
            site=mlp_gate_or_up for the separate gate and up calls)
  verify    an `engaged` line for site=verify.out (the output projections; the in-projections only when QWEN_FAST_GRID13_CFG names them: gdn_in= / attn_in=)
  drafter   nothing is required: its hook is the unapplied patch fusion-wp/M1-dflash-device.patch; a site=drafter.commit line, when there is one, is read like the others
  every arm NO `fell back` line (the served ops ran: the timing is not the lever's) and NO `audit mismatch` line
  audit on  (QWEN_FAST_GRID13_AUDIT=1; the lever must be on too) a passing `audit n=<n> exact=True site=<site>` line for every site the arm requires
  lever off not one engaged, fell-back or audit line (a leak means the lever ran where nobody asked for it)

    python3 grid13_smoke.py --log container.log [--profile NAME [--profiles qwen_c2_profiles.json]] [--env KEY=VALUE ...]

exits 1 on any problem, printing each.
"""

import argparse
import json
import re
import sys
from pathlib import Path

FLAG = 'QWEN_FAST_GRID13'
AUDIT_FLAG = 'QWEN_FAST_GRID13_AUDIT'
CFG_FLAG = 'QWEN_FAST_GRID13_CFG'
SUBSWITCHES = ('verify', 'drafter', 'prefill')
PREFIX = '[PINDIAG] tp4 grid13'
ENGAGED = PREFIX + ' engaged'
FELL_BACK = PREFIX + ' fell back'
AUDIT = PREFIX + ' audit'
MISMATCH = PREFIX + ' audit mismatch'
# Held equal to grid13_tp's by test_grid13_tp; repeated so that the rig host reads this file alone.
PREFILL_SITES = (('gdn_in',), ('attn_in',), ('gate_up', 'mlp_gate_or_up'))
EXACT = re.compile(r'audit n=(\d+) exact=True site=(\S+)')


def subswitches(env):
    """The sub-switches the profile's QWEN_FAST_GRID13 value names (empty when the lever is off)."""
    value = (env or {}).get(FLAG)
    if value in (None, '', '0'):
        return ()
    if value == '1':
        return SUBSWITCHES
    return tuple(name for name in value.split(',') if name in SUBSWITCHES)


def required_sites(env):
    """[(label, [accepted site names])] the arm must show an engaged line for."""
    subs = subswitches(env)
    sites = []
    if 'prefill' in subs:
        sites += [('/'.join(names), list(names)) for names in PREFILL_SITES]
    if 'verify' in subs:
        sites.append(('verify.out', ['verify.out']))
        config = (env or {}).get(CFG_FLAG, '')
        for name in ('gdn_in', 'attn_in'):
            if re.search(r'(^|,)%s=' % name, config):
                sites.append(('verify.' + name, ['verify.' + name]))
    return sites


def problems(env, container_text):
    """[problem] for the profile `env` (flag name -> value) and the server log `container_text`."""
    found = []
    lines = [line for line in (container_text or '').splitlines() if PREFIX in line]
    subs = subswitches(env)
    if not subs:
        found += ['%s is not set and the log holds a grid13 line: the lever ran on a profile without it: %s' % (FLAG, line.strip()[:200]) for line in lines[:4]]
        return found
    found += ['the grid13 lever fell back (the served matmul grids ran, it saved nothing): %s' % line.strip()[:200] for line in lines if FELL_BACK in line][:4]
    found += ['audit mismatch in the log: %s' % line.strip()[:200] for line in lines if MISMATCH in line][:4]
    for label, names in required_sites(env):
        if not any(ENGAGED in line and any('site=%s ' % name in line + ' ' for name in names) for line in lines):
            found.append('%s=%s is set and no engaged line (%s site=%s) was logged: that site ran the served grid' % (FLAG, env.get(FLAG), ENGAGED, label))
    if (env or {}).get(AUDIT_FLAG) == '1':
        audited = set(site for _count, site in (EXACT.findall(line)[0] for line in lines if EXACT.search(line)))
        for label, names in required_sites(env):
            if not any(name in audited for name in names):
                found.append('%s=1 is set and no passing audit line (%s n=<n> exact=True site=%s) was logged: that site was never audited' % (AUDIT_FLAG, AUDIT, label))
    elif any(AUDIT in line and 'exact=' in line for line in lines):
        found.append('%s is not set and the log holds audit lines' % AUDIT_FLAG)
    return found


def _profile_env(name, path):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    entry = (document.get('profiles') or {}).get(name)
    if entry is None:
        raise ValueError('profile %s is not in %s' % (name, path))
    return dict(entry.get('env') or {})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--log', required=True)
    parser.add_argument('--profile')
    parser.add_argument('--profiles', default=str(Path(__file__).with_name('qwen_c2_profiles.json')))
    parser.add_argument('--env', action='append', default=[])
    options = parser.parse_args(argv)
    env = _profile_env(options.profile, options.profiles) if options.profile else {}
    for item in options.env:
        key, _, value = item.partition('=')
        env[key] = value
    text = Path(options.log).read_text(encoding='utf-8', errors='replace')
    found = problems(env, text)
    for line in found:
        print(line)
    print('grid13 smoke: %s' % ('%d problem(s)' % len(found) if found else 'ok'))
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
