"""What a gated arm must see in the server log from the K-split of the tail passes (tp4/prefill-3: QWEN_FAST_SDPA_PF_KSPLIT and QWEN_FAST_SDPA_PF_KSPLIT_AUDIT, docs/tp4-prefill3-ksplit.md), as a
function of the profile's env and the log. Stdlib only, py 3.7 syntax: the gate reads it on the rig host. A host-side rule (scripts/ci/c2_smoke_check.fusion_problems calls problems()), not an image file.

A mounted lever is not an executed one: a profile can carry the flag and split no call (the binary without the edits raises, but a prompt shape the lever does not take never reaches it). And the
audit of this lever is NOT the bitwise `exact=True` line of the other levers: the K-split is a numerics change, so its audit line says `within_bound=True` (the error statistic against the waiver's
bound, ksplit_numerics.BOUND). So, by the env:

  lever on   an `engaged` line (the generic fusion table demands it too) whose rows are 512 or 1024 (the rows the split takes), and NO `audit mismatch` line; a `skipped` line is information (every
             2,048-row chunk pass is one), a `fell back` line fails the arm (the call was not a oneq call: the oneq lever did not take it, so the split had nothing to extend)
  audit on   (QWEN_FAST_SDPA_PF_KSPLIT_AUDIT=1; the lever must be on too) a passing `audit n=<n> ... within_bound=True` line, and NO line with `within_bound=False`
  lever off  not one engaged or audit line (a leak means the lever ran where nobody asked for it)

    python3 ksplit_smoke.py --log container.log [--env KEY=VALUE ...]

exits 1 on any problem, printing each.
"""

import argparse
import re
import sys

FLAG = 'QWEN_FAST_SDPA_PF_KSPLIT'
AUDIT_FLAG = 'QWEN_FAST_SDPA_PF_KSPLIT_AUDIT'
BASE = '[PINDIAG] sdpa prefill ksplit'
ENGAGED = BASE + ' engaged'
SKIPPED = BASE + ' skipped'
FELL_BACK = BASE + ' fell back'
AUDIT = BASE + ' audit'
MISMATCH = BASE + ' audit mismatch'
ROWS = (512, 1024)          # held equal to sdpa_pf_ksplit_tp.ROWS by test_sdpa_ksplit_tp; repeated so that the rig host reads this file alone
ENGAGED_ROWS = re.compile(r'engaged .*?rows=(\d+)')


def problems(env, container_text):
    """[problem] for the K-split a profile's `env` asks for (or does not ask for), read from the container log."""
    env = env or {}
    lines = container_text.splitlines()
    on = env.get(FLAG) == '1'
    audit = env.get(AUDIT_FLAG) == '1'
    found = []
    engaged = [line for line in lines if ENGAGED in line]
    audits = [line for line in lines if AUDIT in line and MISMATCH not in line]
    bad = [line.strip()[:240] for line in lines if MISMATCH in line or 'within_bound=False' in line]
    found += ['the K-split audit found a call outside the waiver bound: %s' % line for line in bad[:4]]
    if not on:
        if engaged or audits:
            found.append('the K-split logged %d engaged and %d audit line(s) but %s is not set to 1: it ran without its switch' % (len(engaged), len(audits), FLAG))
        return found
    fell = [line.strip()[:240] for line in lines if FELL_BACK in line]
    found += ['the K-split fell back (the call was not a oneq call, so there was nothing to split): %s' % line for line in fell[:4]]
    if not engaged:
        found.append('%s=1 is set and no engaged line (%s) was logged: the split never ran' % (FLAG, ENGAGED))
    for line in engaged:
        match = ENGAGED_ROWS.search(line)
        if match is None or int(match.group(1)) not in ROWS:
            found.append('an engaged line for rows the split does not take (%s): %s' % ('/'.join(str(rows) for rows in ROWS), line.strip()[:200]))
    if audit and not any('within_bound=True' in line for line in audits):
        found.append('%s=1 is set and no passing audit line (%s ... within_bound=True) was logged: the split was never audited' % (AUDIT_FLAG, AUDIT))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--log', required=True)
    parser.add_argument('--env', action='append', default=[], help='KEY=VALUE of the profile env')
    options = parser.parse_args(argv)
    env = dict(item.split('=', 1) for item in options.env)
    with open(options.log, encoding='utf-8', errors='replace') as handle:
        found = problems(env, handle.read())
    for line in found:
        print('PROBLEM', line)
    print('ksplit_smoke: %s' % ('FAIL' if found else 'ok'))
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
