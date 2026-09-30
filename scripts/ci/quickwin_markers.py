"""What a server logs under the phase-1 quick-win flags, parsed: the evidence the quickwin gate plans judge.

QWEN_FAST_MEMORY_LEDGER_OFF=1 (memory_ledger.OFF_FLAG): the ledger writes nothing, so an arm that set it must carry no
    '[MEMLEDGER]' line at all, and the flag-off reference (the image sets QWEN_FAST_MEMORY_LEDGER=1) must carry some,
    or the comparison compared nothing.
QWEN_FAST_ENGINE_WARM_SKIP=1 (serving_fast_policy.ENGINE_WARM_MARKER), one line per engine build, and only under the flag:
    [PINDIAG] engine warm forwards run=<n> skipped=<n> request=<id>       (serving_request_factory, a per-request build)
    [PINDIAG] engine warm forwards run=<n> skipped=<n> slot=<k>           (serving_parked_engines, a parked attach or rebuild)
    The first build of a process runs its warm forwards (run >= 1, skipped 0); an arm that shows a skip at all proves the
    skip ran; a flag-off arm must show none of these lines.

Every field is read by name. Stdlib only, Python 3.7 syntax: the gate reads this on the rig host.
"""
import re

LEDGER_MARKER = '[MEMLEDGER]'
WARM_MARKER = '[PINDIAG] engine warm forwards '
WARM_LINE = re.compile(r'\[PINDIAG\] engine warm forwards run=([0-9]+|None) skipped=([0-9]+|None) (request|slot)=(\S+)')


def scan(lines):
    """{'ledger': number of ledger lines, 'warm': [{'run', 'skipped', 'kind', 'id'}, ...] in log order, 'warm_lines':
    number of lines carrying the warm marker (a line that does not parse is counted here and not in 'warm')}."""
    ledger, warm, marked = 0, [], 0
    for line in lines:
        if LEDGER_MARKER in line:
            ledger += 1
        if WARM_MARKER in line:
            marked += 1
            match = WARM_LINE.search(line)
            if match:
                run, skipped, kind, ident = match.groups()
                warm.append(dict(run=None if run == 'None' else int(run),
                                 skipped=None if skipped == 'None' else int(skipped), kind=kind, id=ident))
    return dict(ledger=ledger, warm=warm, warm_lines=marked)


def problems(facts, ledger_off=False, ledger_reference=False, warm_skip=False):
    """(problems, not exercised) of one arm's log. ledger_off: the arm set QWEN_FAST_MEMORY_LEDGER_OFF=1 (no ledger line
    may appear); ledger_reference: the arm is the ledger plan's flag-off reference (some must). warm_skip: the arm set
    QWEN_FAST_ENGINE_WARM_SKIP=1 (its lines must parse, the first build must have warmed, and one later build must have
    skipped); otherwise no warm line may appear."""
    found, missing = [], []
    if ledger_off and facts['ledger']:
        found.append('%d "%s" lines with QWEN_FAST_MEMORY_LEDGER_OFF=1: the ledger ran' % (facts['ledger'], LEDGER_MARKER))
    if ledger_reference and not facts['ledger']:
        missing.append('no "%s" line in the flag-off reference: the ledger the comparison switches off never ran' % LEDGER_MARKER)
    if not warm_skip:
        if facts['warm_lines']:
            found.append('%d "%s" lines without QWEN_FAST_ENGINE_WARM_SKIP=1' % (facts['warm_lines'], WARM_MARKER.strip()))
        return found, missing
    if facts['warm_lines'] != len(facts['warm']):
        found.append('%d warm-forward lines do not parse' % (facts['warm_lines'] - len(facts['warm'])))
    builds = facts['warm']
    if not builds:
        missing.append('no "%s" line: the warm skip never reached an engine build' % WARM_MARKER.strip())
        return found, missing
    if not builds[0]['run'] or builds[0]['skipped']:
        found.append('the first engine build ran %s warm forwards and skipped %s: the process\'s first build must warm '
                     'every bucket' % (builds[0]['run'], builds[0]['skipped']))
    if any(entry['run'] is None or entry['skipped'] is None for entry in builds):
        found.append('a warm-forward line without its counts: the engine carries no warms_run / warms_skipped')
    elif not any(entry['skipped'] for entry in builds[1:]):
        missing.append('no engine build after the first skipped a warm forward: the skip was not exercised')
    return found, missing
