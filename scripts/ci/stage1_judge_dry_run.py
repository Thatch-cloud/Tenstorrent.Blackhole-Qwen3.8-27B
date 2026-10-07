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
    return dict(kind='smoke', profile=name, rejudged=PASS if code == 0 else FAIL, problems=problems,
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
    arms, recorded, problems = {}, {}, []
    for plan, result in (summary.get('results') or {}).items():
        for name, verdict in (result.get('arms') or {}).items():
            recorded[name] = verdict.get('verdict')
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
                          not_exercised=verdict.get('not_exercised') or [])
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
        raise DryRunError('gh run download %s failed: %s' % (run_id, done.stderr.decode('utf-8', 'replace').strip()[:300]))
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
    out('STAGE1_JUDGE_SUMMARY runs=%d rule_only_failure=%s rule_regression=%s unreadable=%d' % (len(specs), ','.join(rule_only) or 'none', ','.join(regression) or 'none', failures))
    if options.json:
        with open(options.json, 'w', encoding='utf-8') as handle:
            json.dump(dict(results=results, rule_only_failure=rule_only, rule_regression=regression), handle, indent=1, sort_keys=True)
    if failures:
        return 2
    return 1 if options.strict and regression else 0


if __name__ == '__main__':
    sys.exit(main())
