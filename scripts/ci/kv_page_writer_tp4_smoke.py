"""Smoke rules for the page-parallel K/V writer (QWEN_FAST_KV_PAGE_WRITER, kv_page_writer_tp4.py), read from the container log.

The markers (kv_page_writer_tp4 logs them; the tests hold these equal to the module's):

    [PINDIAG] tp4 kv page writer engaged cores=<n> wt=<w> source=<dram|l1> ordered=<0|1> width=<W> audit=<0|1> design=<16 hex>
    [PINDIAG] tp4 kv page writer fell back reason=<why>
    [PINDIAG] tp4 kv page writer audit <n> exact=True writers=<w> units=<u>
    [PINDIAG] tp4 kv page writer audit mismatch round=<r> <what>

Rules (problems(env, container_text) returns a list; empty is clean):
  * an audit mismatch line, or ANY fell-back line, fails the smoke (the chained writer ran: the timing is not the lever's, and a missing or stale
    ordered_writer_evidence_tp4.json record is one of the reasons - the lever refuses to engage without it);
  * a profile with QWEN_FAST_KV_PAGE_WRITER=1 must log an engaged line with at least one core, and one with ordered=0 (the captured forward ran the
    page launch; ordered=1 is the warm forward's single-chain mode);
  * a profile with QWEN_FAST_KV_PAGE_WRITER_AUDIT=1 must log an audit line carrying 'exact=True' (a round compared the page launch with the served
    chained writer on a shadow cache) and no mismatch line;
  * engaged or audit lines on a profile without the flag are a problem.

c2_smoke_check's fusion table (generated from scripts/ci/fusion-wp/WP2.json by make_fusion_profiles.py) dispatches the same three rules; this module
adds the ordered=0 rule and is what a hand run of a log uses:

    python3 -s scripts/ci/kv_page_writer_tp4_smoke.py <container log> [QWEN_FAST_KV_PAGE_WRITER=1 QWEN_FAST_KV_PAGE_WRITER_AUDIT=1]

Stdlib only.
"""

import re
import sys

FLAG = 'QWEN_FAST_KV_PAGE_WRITER'
AUDIT_FLAG = 'QWEN_FAST_KV_PAGE_WRITER_AUDIT'
ENGAGED = '[PINDIAG] tp4 kv page writer engaged'
FELL_BACK = '[PINDIAG] tp4 kv page writer fell back'
AUDIT = '[PINDIAG] tp4 kv page writer audit'
MISMATCH = '[PINDIAG] tp4 kv page writer audit mismatch'


def problems(env, container_text):
    lines = container_text.splitlines()
    found = ['the K/V page writer audit found a difference: %s' % line.strip()[:200] for line in lines if MISMATCH in line][:4]
    found += ['the K/V page writer fell back (the chained writer ran, it saved nothing): %s' % line.strip()[:200] for line in lines if FELL_BACK in line][:4]
    if env is None or env.get(FLAG) != '1':
        if any(ENGAGED in line or AUDIT in line for line in lines):
            found.append('K/V page writer lines on a profile without %s' % FLAG)
        return found
    engaged = [line for line in lines if ENGAGED in line]
    if not engaged:
        found.append('%s is set and no engaged line (%s) was logged: the page launch never ran' % (FLAG, ENGAGED))
    else:
        if not any(re.search(r' cores=[1-9]\d*( |$)', line) for line in engaged):
            found.append('%s engaged lines report no core: %s' % (FLAG, engaged[0].strip()[:200]))
        if not any(re.search(r' ordered=0( |$)', line) for line in engaged):
            found.append('%s: no engaged line with ordered=0 - only the warm forward ran the page launch, the captured forward did not' % FLAG)
    if env.get(AUDIT_FLAG) == '1' and not any(AUDIT in line and 'exact=True' in line and MISMATCH not in line for line in lines):
        found.append('%s=1 is set and no passing audit line (%s <n> exact=True) was logged: nothing was compared' % (AUDIT_FLAG, AUDIT))
    return found


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    with open(argv[0], encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    env = dict(item.split('=', 1) for item in argv[1:] if '=' in item)
    found = problems(env, text)
    for problem in found:
        print('PROBLEM: ' + problem)
    print('kv page writer smoke: %s' % ('clean' if not found else '%d problem(s)' % len(found)))
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
