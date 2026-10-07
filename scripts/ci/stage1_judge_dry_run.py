"""stage1_judge_dry_run: re-judge the archived artifacts of hardware runs with the COMMITTED rules, and report every rule-only failure.

Why. A job that serves everything and then fails on a judge rule alone costs a rerun of 15 to 183 minutes (v476 failed its S2 marker rules in 14 s, v578 failed smoke-check
rules that do not apply to an audited profile, v580 served 64 of 64 turns and failed on a marker). Before a window, the rules of the commit the window runs from are
applied to the artifacts of earlier runs, so a rule that is wrong in either direction shows on a CPU and not on the cards. It reads artifacts only: no card, no container.

What it judges, per run directory (the downloaded artifact of one run, `c2-serving-<run id>`; `--download` fetches it with `gh run download`):
  smoke  (smoke.log and container.log): c2_smoke_check.check with the profile the container log names (the `[QWEN-C2] profile X` line), or `--profile`.
  prefix (prefix/c2-prefix-summary.json and one directory per arm): the prefix gate's own arm judge (prefix_markers.scan of the server log, prefix_judge.resolve,
         c2_prefix_gate.judge_arm) over the arm's records, pairs and events, with the plan's arm spec rebuilt from the summary's profile, baseline and plan.

What it reports, one JSON object per run on one `STAGE1_JUDGE` line: the recorded outcome (`--recorded`, else the run's own summary for a prefix run), the re-judged outcome
and its problems, and
  rule_only_failure   the run was recorded as a failure and the committed rules judge it a pass: the failure was a rule, and the rule is fixed;
  rule_regression     the run was recorded as a pass and the committed rules judge it a failure: the rules are stricter than when it passed (a reference run must be
                      re-read before a window relies on it);
  still_failing       recorded and re-judged failures agree (a real failure, or a rule not yet fixed).
Every report also names the RECORDED problems next to the re-judged ones (a prefix run's from its own summary; a smoke run's SMOKE_CHECK FAILED lines from the run log, `run.log` in the artifact folder,
which `--download` saves with `gh run view --log`), so "the rule-only failure is cleared" can be audited rule by rule, and the rules' own state (`STAGE1_JUDGE_RULES`: the commit and any uncommitted
change to a judge file): a rule fix counts only when it is COMMITTED. `--pair A,B` (repeatable) names the two arms of an A/B pair and refuses to re-judge one of them alone: a rule fix must apply to both.
It exits 0 unless `--strict` is given: with it, a rule regression or a failure the run recorded as a pass exits 1, so a gate of rule changes can use it.

    python3 stage1_judge_dry_run.py --run v536=DIR[:recorded] --run v578=DIR:FAIL ... [--profile P] [--strict] [--json OUT]
    python3 stage1_judge_dry_run.py --download v578=37238391881:FAIL --out DIR

Stdlib and the repository's own modules only.
"""

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import c2_smoke_check  # noqa: E402

PROFILE_LINE = re.compile(r'\[QWEN-C2\] profile (\S+?):')
PASS, FAIL = 'PASS', 'FAIL'
RECORDED_LINE = 'SMOKE_CHECK FAILED: '
RULE_FILES = ('c2_smoke_check.py', 'c2_prefix_gate.py', 'prefix_judge.py', 'prefix_markers.py', 'qwen_c2_profiles.json')


class DryRunError(ValueError):
    """An artifact the dry run cannot read."""


def read_text(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        return handle.read()


def run_root(path):
    """The directory that holds the artifact's files: the directory itself, or its single c2-serving-<id> child."""
    if os.path.isfile(os.path.join(path, 'smoke.log')) or os.path.isdir(os.path.join(path, 'prefix')):
        return path
    children = [name for name in sorted(os.listdir(path)) if name.startswith('c2-serving-') and os.path.isdir(os.path.join(path, name))]
    if len(children) == 1:
        return os.path.join(path, children[0])
    raise DryRunError('%s holds neither smoke.log nor prefix/ nor one c2-serving-<run> directory' % path)


def kind_of(root):
    if os.path.isfile(os.path.join(root, 'prefix', 'c2-prefix-summary.json')):
        return 'prefix'
    if os.path.isfile(os.path.join(root, 'smoke.log')) and os.path.isfile(os.path.join(root, 'container.log')):
        return 'smoke'
    raise DryRunError('%s has no smoke.log with container.log and no prefix/c2-prefix-summary.json' % root)


def profile_of_container_log(text):
    """The profile the container's contract line names: [QWEN-C2] profile X: ..., the first one."""
    found = PROFILE_LINE.search(text)
    return found.group(1) if found else None


def recorded_problems_of(log_text):
    """The SMOKE_CHECK FAILED lines of a run log (a timestamp prefix and all): -> [text]. """
    return [line.split(RECORDED_LINE, 1)[1].strip() for line in (log_text or '').splitlines() if RECORDED_LINE in line]


def recorded_log_of(root):
    """The run log kept beside the artifact (`run.log`, in the folder or its parent), or None. """
    for folder in (root, os.path.dirname(os.path.abspath(root))):
        path = os.path.join(folder, 'run.log')
        if os.path.isfile(path):
            return read_text(path)
    return None


def rules_state(repo=None):
    """-> dict(commit, dirty): the checkout's HEAD and the judge files with an uncommitted change (None when git cannot be asked). """
    repo = repo or os.path.dirname(os.path.dirname(HERE))
    try:
        commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode().strip()
        names = ['scripts/ci/' + name for name in RULE_FILES]
        porcelain = subprocess.run(['git', '-C', repo, 'status', '--porcelain', '--'] + names, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)
    return dict(commit=commit, dirty=[line[3:].strip() for line in porcelain.splitlines() if line.strip()])


def judge_smoke(root, profile=None, profiles_path=None, recorded=None):
    container = read_text(os.path.join(root, 'container.log'))
    name = profile or profile_of_container_log(container)
    if not name:
        raise DryRunError('%s: no "[QWEN-C2] profile X:" line in container.log and no --profile' % root)
    argv = ['--smoke-log', os.path.join(root, 'smoke.log'), '--container-log', os.path.join(root, 'container.log'), '--profile', name]
    if profiles_path:
        argv += ['--profiles', profiles_path]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = c2_smoke_check.main(argv)
    lines = out.getvalue().splitlines()
    problems = [line.split('SMOKE_CHECK FAILED: ', 1)[1] for line in lines if line.startswith('SMOKE_CHECK FAILED: ')]
    if code == 2:
        raise DryRunError('%s: the smoke check could not read its inputs: %s' % (root, err.getvalue().strip()))
    log = recorded_log_of(root)
    return dict(kind='smoke', profile=name, rejudged=PASS if code == 0 else FAIL, problems=problems,
                recorded_problems=recorded_problems_of(log) if log is not None else None,
                unqualified=any(line.startswith('SMOKE_CHECK UNQUALIFIED') for line in lines))


class ArmDriver(object):
    """What judge_arm reads of the driver that ran an arm: its records, pairs and events, restored from the arm's files."""

    def __init__(self, directory):
        with open(os.path.join(directory, 'records.jsonl'), encoding='utf-8') as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        with open(os.path.join(directory, 'pairs.json'), encoding='utf-8') as handle:
            self.pairs = json.load(handle)
        with open(os.path.join(directory, 'events.json'), encoding='utf-8') as handle:
            document = json.load(handle)
        self.events = document.get('events', [])
        self.phases = document.get('phases', {})
        self.fitted = []


def arm_specs(summary, profiles):
    """{arm name: its plan_arms spec} for every plan the summary ran, rebuilt with the checkout's profiles."""
    import c2_prefix_gate as gate

    specs = {}
    baseline = summary.get('baseline')
    for plan in summary.get('plans') or []:
        for arm in gate.plan_arms(plan, summary['profile'], None if baseline in (None, 'none') else baseline, profiles):
            specs[arm['arm']] = arm
    return specs


def judge_prefix(root, profiles_path=None):
    import c2_prefix_gate as gate
    import prefix_judge
    import prefix_markers

    prefix_dir = os.path.join(root, 'prefix')
    with open(os.path.join(prefix_dir, 'c2-prefix-summary.json'), encoding='utf-8') as handle:
        summary = json.load(handle)
    with open(profiles_path or os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
        profiles = json.load(handle)
    specs = arm_specs(summary, profiles)
    arms, recorded, recorded_problems, problems = {}, {}, {}, []
    for plan, result in (summary.get('results') or {}).items():
        for name, verdict in (result.get('arms') or {}).items():
            recorded[name] = verdict.get('verdict')
            recorded_problems[name] = verdict.get('problems') or []
    for name in sorted(os.listdir(prefix_dir)):
        directory = os.path.join(prefix_dir, name)
        if not os.path.isfile(os.path.join(directory, 'arm.json')):
            continue
        if name not in specs:
            raise DryRunError('%s: arm %s is in no plan of the summary (%s)' % (root, name, ', '.join(summary.get('plans') or [])))
        with open(os.path.join(directory, 'arm.json'), encoding='utf-8') as handle:
            arm_json = json.load(handle)
        log_name = 'server-final.log' if os.path.isfile(os.path.join(directory, 'server-final.log')) else 'server.log'
        lines = read_text(os.path.join(directory, log_name)).splitlines()
        driver = ArmDriver(directory)
        scanned = prefix_markers.scan(lines)
        result = arm_json.get('result') or {}
        stats = result.get('stats') if result.get('stats') is not None else scanned.get('stats')
        prefix_judge.resolve(driver.records, scanned)
        spec = dict(specs[name])
        spec['gate_only'] = ((spec.get('derived') or profiles)['profiles'].get(spec['served']) or {}).get('gate_only') is True
        verdict = gate.judge_arm(spec, driver, scanned, stats, result.get('error'), log_text='\n'.join(lines))
        arms[name] = dict(recorded=recorded.get(name) or result.get('verdict'), rejudged=verdict['verdict'], problems=verdict.get('problems') or [],
                          not_exercised=verdict.get('not_exercised') or [], recorded_problems=recorded_problems.get(name) or result.get('problems') or [])
        problems += ['%s: %s' % (name, text) for text in arms[name]['problems']]
    if not arms:
        raise DryRunError('%s: no arm directory with an arm.json under prefix/' % root)
    rejudged = PASS if all(arm['rejudged'] == 'PASS' for arm in arms.values()) else FAIL
    recorded_all = PASS if all(arm['recorded'] == 'PASS' for arm in arms.values()) else FAIL
    return dict(kind='prefix', profile=summary.get('profile'), plans=summary.get('plans'), arms=arms, rejudged=rejudged, problems=problems,
                recorded_from_summary=recorded_all)


def classify(recorded, rejudged):
    """-> 'rule_only_failure', 'rule_regression', 'still_failing', 'agrees_pass' or 'unrecorded'."""
    if recorded is None:
        return 'unrecorded'
    if recorded == FAIL and rejudged == PASS:
        return 'rule_only_failure'
    if recorded == PASS and rejudged == FAIL:
        return 'rule_regression'
    return 'still_failing' if rejudged == FAIL else 'agrees_pass'


def judge_run(label, path, recorded=None, profile=None, profiles_path=None):
    root = run_root(path)
    kind = kind_of(root)
    result = judge_smoke(root, profile, profiles_path) if kind == 'smoke' else judge_prefix(root, profiles_path)
    if recorded is None and kind == 'prefix':
        recorded = result.get('recorded_from_summary')
    result.update(label=label, recorded=recorded, outcome=classify(recorded, result['rejudged']))
    result.pop('recorded_from_summary', None)
    return result


def download(run_id, out):
    """gh run download into out/<run id>; -> the directory. Needs the gh CLI logged in to the repository."""
    target = os.path.join(out, str(run_id))
    os.makedirs(target, exist_ok=True)
    done = subprocess.run(['gh', 'run', 'download', str(run_id), '-D', target], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if done.returncode != 0:
        raise DryRunError('gh run download %s failed (an artifact older than the retention period is gone): %s' % (run_id, done.stderr.decode('utf-8', 'replace').strip()[:300]))
    # The run log carries the SMOKE_CHECK FAILED lines the artifact does not: best effort (a failed fetch leaves recorded_problems None).
    log = subprocess.run(['gh', 'run', 'view', str(run_id), '--log'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if log.returncode == 0:
        with open(os.path.join(target, 'run.log'), 'wb') as handle:
            handle.write(log.stdout)
    return target


def parse_spec(text, expect_recorded=True):
    """label=target[:RECORDED] -> (label, target, recorded or None). A Windows drive letter in the target is not a separator."""
    label, _, rest = text.partition('=')
    if not label or not rest:
        raise DryRunError('%r is not label=target[:PASS|FAIL]' % text)
    recorded = None
    for word in (PASS, FAIL):
        if rest.endswith(':' + word):
            rest, recorded = rest[:-(len(word) + 1)], word
    return label, rest, recorded


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', default=[], help='label=DIR[:PASS|FAIL] an artifact directory already downloaded')
    parser.add_argument('--download', action='append', default=[], help='label=RUNID[:PASS|FAIL] fetched with gh run download into --out')
    parser.add_argument('--out', default='.', help='where --download puts the artifacts')
    parser.add_argument('--profile', default=None, help='the profile a smoke run served, when its container log does not name it')
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the checkout\'s')
    parser.add_argument('--pair', action='append', default=[], help='LABEL_A,LABEL_B: the two arms of an A/B pair; both must be re-judged together (a rule fix applies to both)')
    parser.add_argument('--strict', action='store_true', help='exit 1 on a rule regression')
    parser.add_argument('--json', default=None, help='write the whole report here')
    options = parser.parse_args(argv)
    specs = []
    try:
        for text in options.run:
            specs.append(parse_spec(text))
        for text in options.download:
            label, run_id, recorded = parse_spec(text)
            specs.append((label, download(run_id, options.out), recorded))
    except DryRunError as error:
        out('refused: %s' % error)
        return 2
    if not specs:
        out('refused: no --run and no --download')
        return 2
    labels = [label for label, _, _ in specs]
    for pair in options.pair:
        names = pair.split(',')
        if len(names) != 2 or not all(names) or names[0] == names[1]:
            out('refused: --pair %r is not LABEL_A,LABEL_B' % pair)
            return 2
        missing = [name for name in names if name not in labels]
        if missing:
            out('refused: --pair %s: %s is not among the runs to re-judge; one arm of an A/B pair is never re-judged alone (a rule fix must apply to both)' % (pair, ','.join(missing)))
            return 2
    rules = rules_state()
    results, failures = [], 0
    for label, path, recorded in specs:
        try:
            result = judge_run(label, path, recorded, options.profile, options.profiles)
        except (DryRunError, OSError, ValueError) as error:
            out('STAGE1_JUDGE %s' % json.dumps(dict(label=label, error=str(error))))
            failures += 1
            continue
        results.append(result)
        out('STAGE1_JUDGE %s' % json.dumps(result, sort_keys=True))
    rule_only = [item['label'] for item in results if item['outcome'] == 'rule_only_failure']
    regression = [item['label'] for item in results if item['outcome'] == 'rule_regression']
    out('STAGE1_JUDGE_RULES %s' % json.dumps(rules, sort_keys=True))
    for pair in options.pair:
        names = pair.split(',')
        outcomes = dict((item['label'], item['outcome']) for item in results)
        committed = (rules['dirty'] == []) if rules['dirty'] is not None else None
        out('STAGE1_JUDGE_PAIR %s' % json.dumps(dict(pair=names, outcomes=[outcomes.get(name) for name in names], rules=rules, committed=committed), sort_keys=True))
        if rules['dirty']:
            out('note: the judge files %s have uncommitted changes: a rule fix counts only when committed and reviewed' % ', '.join(rules['dirty']))
    out('STAGE1_JUDGE_SUMMARY runs=%d rule_only_failure=%s rule_regression=%s unreadable=%d' % (len(specs), ','.join(rule_only) or 'none', ','.join(regression) or 'none', failures))
    if options.json:
        with open(options.json, 'w', encoding='utf-8') as handle:
            json.dump(dict(results=results, rule_only_failure=rule_only, rule_regression=regression, rules=rules), handle, indent=1, sort_keys=True)
    if failures:
        return 2
    return 1 if options.strict and regression else 0


if __name__ == '__main__':
    sys.exit(main())
