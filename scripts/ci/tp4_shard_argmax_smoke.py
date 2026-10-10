"""The smoke rule of S1, the tile-native shard argmax (tp4_shard_argmax), in the style of w3_mr_smoke and c2_smoke_check.spread_problems.
Stdlib only, host side.

    problems(env, container_text) -> [problem]

`env` is the served profile's environment (a dict of strings, or None), `container_text` the server log. The parent smoke check calls this once per
arm. What it holds:

  - without QWEN_FAST_TP4_SHARD_ARGMAX: none of the lever's lines (engaged, audit, mismatch), and neither of its companion flags
    (QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT, QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2) set: a flag that chooses how the lever runs, or audits it, on an arm
    that does not run it measured nothing;
  - with it: at least one engaged line for the packed block (rows=64), every engaged line naming the fold the profile asked for
    (fold=2 under QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2=1, fold=1 otherwise), `words=1` (the joined readback tensor the W3 mesh-read work shares),
    and a task count of two a worker (tasks = 2 x workers: the grid the line reports is the grid the plan used);
  - under the audit flag: at least one audit line that says exact=True for four chips;
  - on any arm: no fall-back line (the served path ran, the arm's timing is not the lever's) and no audit-mismatch line.
"""

import re

SHARD_ARGMAX = 'QWEN_FAST_TP4_SHARD_ARGMAX'
AUDIT = 'QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT'
FOLD2 = 'QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2'

ENGAGED = '[PINDIAG] tp4 shard argmax engaged'
FELL_BACK = '[PINDIAG] tp4 shard argmax fell back'
AUDIT_PASSED = '[PINDIAG] tp4 shard argmax audit'
MISMATCH = '[PINDIAG] tp4 shard argmax audit mismatch'

PACKED_ROWS = 64
CHIPS = 4


def _on(env, name):
    return (env or {}).get(name) == '1'


def problems(env, container_text):
    lines = container_text.splitlines()
    found = []
    engaged = [line for line in lines if ENGAGED in line]
    fell_back = [line for line in lines if FELL_BACK in line]
    mismatch = [line for line in lines if MISMATCH in line]
    audits = [line for line in lines if AUDIT_PASSED in line and MISMATCH not in line]
    found += ['the shard argmax fell back (the served path ran, it saved nothing): %s' % line.strip()[:200] for line in fell_back[:4]]
    found += ['the shard argmax audit found a difference: %s' % line.strip()[:200] for line in mismatch[:4]]

    if not _on(env, SHARD_ARGMAX):
        if engaged or audits:
            found.append('shard argmax lines on a profile without %s' % SHARD_ARGMAX)
        for flag in (AUDIT, FOLD2):
            if _on(env, flag):
                found.append('%s=1 without %s=1' % (flag, SHARD_ARGMAX))
        return found

    want_fold = 2 if _on(env, FOLD2) else 1
    if not engaged:
        found.append('%s=1 is set and no engaged line (%s) was logged: the kernels never ran' % (SHARD_ARGMAX, ENGAGED))
    else:
        if not any(re.search(r' rows=%d( |$)' % PACKED_ROWS, line) for line in engaged):
            found.append('no engaged line for the packed block (rows=%d): the verify trace never took the kernels' % PACKED_ROWS)
        for line in engaged:
            fold = re.search(r' fold=(\d+)( |$)', line)
            if fold is None or int(fold.group(1)) != want_fold:
                found.append('an engaged line names the wrong fold (the profile asks for fold=%d): %s' % (want_fold, line.strip()[:200]))
                break
        for line in engaged:
            if not re.search(r' words=1( |$)', line):
                found.append('an engaged line does not report the joined words output (words=1): %s' % line.strip()[:200])
                break
        for line in engaged:
            workers = re.search(r' workers=(\d+)( |$)', line)
            tasks = re.search(r' tasks=(\d+)( |$)', line)
            if workers is None or tasks is None or int(tasks.group(1)) != 2 * int(workers.group(1)):
                found.append('an engaged line reports a task count that is not two a worker: %s' % line.strip()[:200])
                break
    if _on(env, AUDIT):
        passing = [line for line in audits if ' exact=True' in line and re.search(r' chips=%d( |$)' % CHIPS, line)]
        if not passing:
            found.append('%s=1 is set and no passing audit line (%s N exact=True ... chips=%d) was logged: nothing was compared' % (
                AUDIT, AUDIT_PASSED, CHIPS))
    return found
