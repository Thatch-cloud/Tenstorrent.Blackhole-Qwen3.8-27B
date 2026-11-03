"""Run the CPU suites of the vLLM 0.30 job (qwen-vllm030-cpu.yml) from a manifest, every one through required_tests.py.

    python3 -B scripts/ci/vllm030_suites.py [--manifest FILE] [--branch NAME] [--root DIR] [--json FILE] [--summary FILE] [--dry-run]

The manifest (vllm030_suites.json next to this file) lists suites: a directory, a glob of test modules in it, the modules that
must exist, a floor on the number of tests that must run, and extra import paths. Each suite runs in its own interpreter as

    python -B required_tests.py --min N MODULE [MODULE ...]

so a skipped test, a module that does not import, a target that loads nothing, or too few tests all fail the suite (see
required_tests.py). A new test file under a suite directory is found by the glob; a deleted one is caught by `require`.

A suite whose directory does not exist FAILS. The one way to let it be missing is the manifest's per-branch `absent_ok` table
(branch -> suite -> reason): the suite is then reported as ABSENT-ALLOWED with its reason and a warning annotation, never as a
pass. An `absent_ok` entry whose directory exists is stale and fails too, so an allowance cannot outlive the reason for it. A run
in which no suite ran at all fails, whatever the table says.

Exit 0 only when no suite failed and at least one ran. Stdlib only, Python 3.7 syntax.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, 'vllm030_suites.json')
REQUIRED_TESTS = os.path.join(HERE, 'required_tests.py')

PASS = 'PASS'
FAIL = 'FAIL'
ABSENT_ALLOWED = 'ABSENT-ALLOWED'

IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
VERDICT = re.compile(r'^REQUIRED TESTS: (PASS|FAIL)(.*)$')
RAN = re.compile(r'(\d+) ran')


class ManifestError(Exception):
    pass


def _relative(path, what):
    if not isinstance(path, str) or not path or os.path.isabs(path) or '..' in path.replace('\\', '/').split('/'):
        raise ManifestError('%s must be a relative path without "..": %r' % (what, path))


def load_manifest(path):
    """-> the manifest dict, validated; raises ManifestError."""
    try:
        with open(path, encoding='utf-8') as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        raise ManifestError('cannot read %s: %s' % (path, error))
    if not isinstance(manifest, dict) or manifest.get('schema') != 1:
        raise ManifestError('schema must be 1')
    suites = manifest.get('suites')
    if not isinstance(suites, dict) or not suites:
        raise ManifestError('suites must be a non-empty object')
    for name, suite in suites.items():
        if not isinstance(suite, dict):
            raise ManifestError('suite %s must be an object' % name)
        _relative(suite.get('dir'), 'suite %s dir' % name)
        if not isinstance(suite.get('glob', 'test_*.py'), str) or '/' in suite.get('glob', 'test_*.py'):
            raise ManifestError('suite %s glob must be a file-name pattern' % name)
        for entry in suite.get('pythonpath', []):
            _relative(entry, 'suite %s pythonpath entry' % name)
        for module in suite.get('require', []):
            if not isinstance(module, str) or not IDENTIFIER.match(module):
                raise ManifestError('suite %s require: %r is not a module name' % (name, module))
        minimum = suite.get('min_tests', 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
            raise ManifestError('suite %s min_tests must be an integer of at least 1' % name)
    absent = manifest.get('absent_ok', {})
    if not isinstance(absent, dict):
        raise ManifestError('absent_ok must be an object')
    for branch, entries in absent.items():
        if not isinstance(entries, dict) or not entries:
            raise ManifestError('absent_ok[%s] must be a non-empty object of suite -> reason' % branch)
        for name, reason in entries.items():
            if name not in suites:
                raise ManifestError('absent_ok[%s] names unknown suite %s' % (branch, name))
            if not isinstance(reason, str) or not reason.strip():
                raise ManifestError('absent_ok[%s][%s] needs a reason' % (branch, name))
    return manifest


def modules_of(directory, pattern):
    """The module names of the test files in `directory` (not below it) that match `pattern`, sorted."""
    names = []
    for entry in sorted(os.listdir(directory)):
        stem, extension = os.path.splitext(entry)
        if extension == '.py' and fnmatch.fnmatchcase(entry, pattern) and IDENTIFIER.match(stem) \
                and os.path.isfile(os.path.join(directory, entry)):
            names.append(stem)
    return names


def plan_suite(name, suite, root, branch, absent_ok):
    """-> a result dict. status is PASS (still to run: 'argv' is set), FAIL (final) or ABSENT-ALLOWED (final)."""
    directory = os.path.join(root, suite['dir'])
    allowance = (absent_ok.get(branch) or {}).get(name) if branch else None
    result = {'suite': name, 'dir': suite['dir'], 'status': FAIL, 'ran': 0, 'detail': ''}
    if not os.path.isdir(directory):
        if allowance:
            result.update(status=ABSENT_ALLOWED, detail='%s is absent on branch %s; allowed by the manifest: %s'
                          % (suite['dir'], branch, allowance))
        else:
            result['detail'] = '%s does not exist and the manifest allows no absence of %s on branch %r' % (
                suite['dir'], name, branch or '')
        return result
    if allowance:
        result['detail'] = ('stale absent_ok: %s exists on branch %s, so the manifest must not excuse its absence any more '
                            '(remove absent_ok[%s][%s])' % (suite['dir'], branch, branch, name))
        return result
    modules = modules_of(directory, suite.get('glob', 'test_*.py'))
    missing = [module for module in suite.get('require', []) if module not in modules]
    if missing:
        result['detail'] = 'required test modules missing from %s: %s' % (suite['dir'], ', '.join(missing))
        return result
    if not modules:
        result['detail'] = 'no test module in %s matches %s' % (suite['dir'], suite.get('glob', 'test_*.py'))
        return result
    paths = [directory] + [os.path.join(root, entry) for entry in suite.get('pythonpath', [])]
    result.update(status=PASS, modules=modules, paths=paths, root=root, minimum=suite.get('min_tests', 1))
    return result


def run_suite(result, python, environment):
    """Run a planned suite; fill in status, ran and detail."""
    env = dict(environment)
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = os.pathsep.join(result['paths'] + ([existing] if existing else []))
    argv = [python, '-B', REQUIRED_TESTS, '--min', str(result['minimum'])] + result['modules']
    print('::group::%s: %s (%d modules, at least %d tests)' % (result['suite'], result['dir'], len(result['modules']),
                                                                result['minimum']))
    sys.stdout.flush()
    verdict = ''
    process = subprocess.Popen(argv, cwd=result['root'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, errors='replace')
    for line in process.stdout:
        sys.stdout.write(line)
        if VERDICT.match(line.strip()):
            verdict = line.strip()
    process.stdout.close()
    code = process.wait()
    print('::endgroup::')
    match = VERDICT.match(verdict)
    if code == 0 and match and match.group(1) == 'PASS':
        ran = RAN.search(match.group(2))
        result.update(status=PASS, ran=int(ran.group(1)) if ran else 0, detail=verdict)
    else:
        result.update(status=FAIL, detail=verdict or 'required_tests.py exited %d without a verdict line' % code)
    return result


def run(manifest, root, branch, python, environment, dry_run=False):
    """-> (exit code, results)."""
    results = []
    for name in sorted(manifest['suites']):
        planned = plan_suite(name, manifest['suites'][name], root, branch, manifest.get('absent_ok', {}))
        if planned['status'] == PASS and not dry_run:
            planned = run_suite(planned, python, environment)
        results.append(planned)
    failed = [r for r in results if r['status'] == FAIL]
    ran = [r for r in results if r['status'] == PASS]
    return (1 if failed or not ran else 0), results


def render(results, branch, code, dry_run=False):
    lines = ['vllm030 suites on branch %r%s' % (branch or '', ' (dry run: nothing executed)' if dry_run else '')]
    for r in results:
        tail = r['detail']
        if r['status'] == PASS and dry_run:
            tail = '%d modules, at least %d tests: %s' % (len(r['modules']), r['minimum'], ' '.join(r['modules']))
        lines.append('  %-15s %-15s %s' % (r['suite'], r['status'], tail))
    if code and not any(r['status'] == PASS for r in results) and not any(r['status'] == FAIL for r in results):
        lines.append('  no suite ran: a run that executed nothing proves nothing')
    lines.append('VLLM030 SUITES: %s' % ('FAIL' if code else 'PASS'))
    return lines


def annotations(results):
    out = []
    for r in results:
        if r['status'] == FAIL:
            out.append('::error title=%s::%s' % (r['suite'], r['detail']))
        elif r['status'] == ABSENT_ALLOWED:
            out.append('::warning title=%s not run::%s' % (r['suite'], r['detail']))
    return out


def summary_markdown(results, branch, code):
    rows = ['### vLLM 0.30 CPU suites (%s): %s' % (branch or 'no branch', 'FAIL' if code else 'PASS'), '',
            '| suite | status | tests ran | detail |', '|---|---|---|---|']
    for r in results:
        rows.append('| %s | %s | %s | %s |' % (r['suite'], r['status'], r['ran'] if r['status'] == PASS else '-',
                                              r['detail'].replace('|', '/')))
    return '\n'.join(rows) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--root', default=os.getcwd(), help='the repository root the suite directories are relative to')
    parser.add_argument('--branch', default=os.environ.get('CI_BRANCH', ''), help='the branch under test (default $CI_BRANCH)')
    parser.add_argument('--json', dest='json_path', help='write the results here')
    parser.add_argument('--summary', help='append a markdown table here (GITHUB_STEP_SUMMARY)')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true', help='plan only: resolve modules and absences, run nothing')
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as error:
        print('VLLM030 SUITES: FAIL: bad manifest: %s' % error)
        return 2
    root = os.path.abspath(args.root)
    code, results = run(manifest, root, args.branch, args.python, os.environ, dry_run=args.dry_run)
    for line in render(results, args.branch, code, args.dry_run) + annotations(results):
        print(line)
    if args.json_path:
        slim = [{k: v for k, v in r.items() if k not in ('paths', 'root')} for r in results]
        with open(args.json_path, 'w', encoding='utf-8') as handle:
            json.dump({'branch': args.branch, 'python': sys.version.split()[0], 'passed': code == 0, 'pins': manifest.get('pins', {}),
                       'suites': slim}, handle, indent=2, sort_keys=True)
            handle.write('\n')
    if args.summary:
        with open(args.summary, 'a', encoding='utf-8') as handle:
            handle.write(summary_markdown(results, args.branch, code))
    sys.stdout.flush()
    return code


if __name__ == '__main__':
    sys.exit(main())
