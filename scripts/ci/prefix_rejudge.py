"""Re-judge one prefix-gate arm from the directory the gate saved, with no docker, no network and no card.

    python3 prefix_rejudge.py --arm-dir <results>/prefix/<arm> [--plan PLAN] [--arm ARM] [--profile NAME] [--profiles qwen_c2_profiles.json]
                              [--model REPO@REVISION] [--json out.json]

A card run that fails on the judge's own bookkeeping (a stale rule, not an engine fault) leaves everything the judgement read in its results: server.log (the follower's lines;
server-final.log is docker's copy of the same), records.jsonl (every request, its token ids and log window), pairs.json, events.json and arm.json (the verdict, the stats export, the
error). This module replays the gate's judgement over them: the arm is rebuilt by c2_prefix_gate.plan_arms from the profiles file, the plan and the profile (named by the log's own
`[QWEN-C2] profile <name>: vLLM argv` line unless --profile says), the markers are scanned and resolved from the log again with the CURRENT prefix_markers and prefix_judge, and
c2_prefix_gate.judge_arm renders the same lines the gate printed ('problem: ...', 'not exercised: ...', the pair line), then the verdict. The first reading of a fixed rule on a finished
run costs seconds, not a card window; an unchanged rule reproduces the original verdict line for line (the original is printed beside it), which is the check that the replay is faithful.

What it is not: a second run. The records are the saved ones (the oracle's expectations were decided during the run), the stats export is the one arm.json kept, and the checkpoint pair
the job asked for (C2_MODEL_REPO / C2_MODEL_REVISION) is read back from the log's own checkpoint line unless --model gives it. salt.key in the directory is a throwaway secret of the run
and is never opened.

Exit status: 0 when the re-judged verdict is PASS, 1 otherwise, 2 when the directory cannot be replayed.

Stdlib only, Python 3.7 syntax.
"""
import argparse
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_prefix_gate as gate  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_markers as markers  # noqa: E402

# The container log's own checkpoint line (serving_c2_contract): `[QWEN-C2] checkpoint <repo>@<40-hex revision> key=...`.
CHECKPOINT_LINE = re.compile(r'\[QWEN-C2\] checkpoint (\S+)@([0-9a-f]{40})\b')


class RejudgeError(ValueError):
    """A results directory this module cannot replay (a file missing, a profile or arm the plan does not name)."""


def _json(path, default=None):
    if not os.path.isfile(path):
        if default is not None:
            return default
        raise RejudgeError('%s is missing' % path)
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def read_arm_dir(arm_dir):
    """What the gate saved of one arm, as the judgement reads it: dict(lines, records, pairs, events, phases, stats, error, original). The log is split on newlines alone
    (str.splitlines also splits on form feeds and the like, which would shift every line index the records' log windows hold)."""
    log = os.path.join(arm_dir, 'server.log')
    if not os.path.isfile(log):
        log = os.path.join(arm_dir, 'server-final.log')
    if not os.path.isfile(log):
        raise RejudgeError('%s holds neither server.log nor server-final.log' % arm_dir)
    with open(log, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    records = []
    with open(os.path.join(arm_dir, 'records.jsonl'), encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    events = _json(os.path.join(arm_dir, 'events.json'), default={})
    saved = _json(os.path.join(arm_dir, 'arm.json'), default={})
    result = saved.get('result') or {}
    return dict(lines=lines, records=records, pairs=_json(os.path.join(arm_dir, 'pairs.json'), default=[]), events=events.get('events') or {},
                phases=events.get('phases') or {}, stats=result.get('stats'), error=result.get('error'), original=result)


def checkpoint_of(lines):
    """(repo, revision) of the first checkpoint line of the log, or None (a run that served the base model logs none)."""
    for line in lines:
        match = CHECKPOINT_LINE.search(line)
        if match:
            return match.group(1), match.group(2)
    return None


def profile_of(scanned):
    """The profile the log's last vLLM argv line says it served (a derived arm's name carries its '+kind' suffix, which plan_arms rebuilds), or None."""
    launches = scanned.get('launches') or []
    return launches[-1]['profile'] if launches else None


def find_plan(arm_name, plan=None):
    """The plan that owns the arm: `plan` when given (and it must be one of PLAN_ARMS), else the plan named like the arm when it lists it (a single-arm plan: exact-resume), else the
    first plan in PLAN_ARMS whose arm tuples name it."""
    if plan is not None:
        if plan not in gate.PLAN_ARMS:
            raise RejudgeError('unknown plan %r (%s)' % (plan, ', '.join(sorted(gate.PLAN_ARMS))))
        return plan
    owners = [name for name, entries in sorted(gate.PLAN_ARMS.items()) if any(entry[0] == arm_name for entry in entries)]
    if not owners:
        raise RejudgeError('no plan lists an arm named %r: pass --plan' % arm_name)
    return arm_name if arm_name in owners else owners[0]


def rebuild_arm(profiles, plan, arm_name, profile):
    """The plan_arms entry of `arm_name` for the base profile `profile` (a derived arm's served name is '<base>+<kind>')."""
    base = profile.split('+')[0]
    try:
        arms = gate.plan_arms(plan, base, 'none', profiles)
    except (gate.PlanError, ValueError) as error:
        raise RejudgeError('the plan %s cannot be rebuilt for %s: %s' % (plan, base, error))
    for arm in arms:
        if arm['arm'] == arm_name:
            return arm
    raise RejudgeError('plan %s on %s has no arm %r (it has %s)' % (plan, base, arm_name, ', '.join(arm['arm'] for arm in arms) or 'none'))


def rejudge(arm_dir, profiles, plan=None, arm_name=None, profile=None, model=None):
    """-> (result, arm, saved): judge_arm's verdict dict over the saved directory, the rebuilt arm, and what was read."""
    saved = read_arm_dir(arm_dir)
    scanned = markers.scan(saved['lines'])
    arm_name = arm_name or os.path.basename(os.path.normpath(arm_dir))
    profile = profile or profile_of(scanned)
    if not profile:
        raise RejudgeError('the log holds no "[QWEN-C2] profile <name>: vLLM argv" line: pass --profile')
    arm = rebuild_arm(profiles, find_plan(arm_name, plan), arm_name, profile)
    judge.resolve(saved['records'], scanned)
    driver = types.SimpleNamespace(records=saved['records'], pairs=saved['pairs'], events=saved['events'], phases=saved['phases'], fitted=[])
    stats = saved['stats'] if saved['stats'] is not None else scanned.get('stats')
    model = model if model is not None else checkpoint_of(saved['lines'])
    result = gate.judge_arm(arm, driver, scanned, stats, saved['error'], log_text='\n'.join(saved['lines']), **({'model': model} if model else {}))
    return result, arm, saved


def render(result, arm, saved):
    """The lines to print: the original verdict, every line of the re-judged result (the gate's own rendering), the new verdict."""
    original = saved['original'] or {}
    out = ['[REJUDGE] arm %s: profile %s, scenario %s' % (arm['arm'], arm['served'], arm['scenario'])]
    if original.get('verdict'):
        out.append('[REJUDGE] saved verdict: %s (%d problem(s), %d not exercised)' % (
            original['verdict'], len(original.get('problems') or ()), len(original.get('not_exercised') or ())))
    out += ['[REJUDGE] %s %s' % (arm['arm'], line) for line in result['lines']]
    out.append('[REJUDGE] arm %s: %s (%d problem(s), %d not exercised)' % (
        arm['arm'], result['verdict'], len(result.get('problems') or ()), len(result.get('not_exercised') or ())))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm-dir', required=True, help='<results>/prefix/<arm>: the directory the gate wrote for the arm')
    parser.add_argument('--plan', help='the plan that owns the arm (default: the first plan listing it)')
    parser.add_argument('--arm', help='the arm name (default: the directory name)')
    parser.add_argument('--profile', help='the served profile (default: the one the log launched)')
    parser.add_argument('--profiles', default=os.path.join(HERE, 'qwen_c2_profiles.json'), help='the profiles file the run served from')
    parser.add_argument('--model', help='REPO@REVISION the job asked for (default: the log\'s own checkpoint line)')
    parser.add_argument('--json', help='also write the verdict dict here')
    options = parser.parse_args(argv)
    model = None
    if options.model:
        repo, _, revision = options.model.partition('@')
        model = (repo, revision) if repo and revision else None
        if model is None:
            parser.error('--model wants REPO@REVISION')
    try:
        with open(options.profiles, encoding='utf-8') as handle:
            profiles = json.load(handle)
        result, arm, saved = rejudge(options.arm_dir, profiles, plan=options.plan, arm_name=options.arm, profile=options.profile, model=model)
    except (RejudgeError, OSError, ValueError) as error:
        print('[REJUDGE] cannot replay %s: %s' % (options.arm_dir, error))
        return 2
    for line in render(result, arm, saved):
        print(line)
    if options.json:
        with open(options.json, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=1, sort_keys=True, default=str)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
