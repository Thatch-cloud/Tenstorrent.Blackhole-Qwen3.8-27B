"""Run the CPU suites of the vLLM 0.30 job (qwen-vllm030-cpu.yml) from a manifest, every one through required_tests.py.

    python3 -B scripts/ci/vllm030_suites.py [--manifest FILE] [--manifest-root DIR] [--branch NAME] [--root DIR] [--json FILE]
                                            [--summary FILE] [--dry-run]

The manifest (vllm030_suites.json next to this file) lists suites: a directory, a glob of test modules in it, the modules that
must exist, a floor on the number of tests that must run, a time limit, and extra import paths. Each suite runs in its own
interpreter as

    required_tests.py --min N MODULE [MODULE ...]

so a skipped test, a module that does not import, a target that loads nothing, or too few tests all fail the suite (see
required_tests.py). A new test file directly under a suite directory is found by the glob; a deleted one is caught by `require`. A
test file that the suite would NOT run (one in a subdirectory, or one the glob misses) fails the suite, unless the manifest's
`exclude` table names it with a reason: a test that is never run is the dead-test case the allowlist rule exists for.

The child interpreter does not put scripts/ci on its import path (a suite module that shares a name with one of the ~2,000 modules
there would load the wrong one and still pass), its PYTHONPATH is the suite's directory and the manifest's `pythonpath` entries and
nothing the caller had set, and a suite module that was loaded from outside its own directory fails the suite.
A suite that runs past its `timeout_s` is sent SIGUSR1 (faulthandler dumps every thread's stack into the log), then killed, and
reported as a FAIL "timeout": the other suites still run and the JSON and the summary still say which one hung.

--root is the tree under test; --manifest-root is the tree that carries the manifest (default: --root). They differ when the job
checks the CI scaffolding out at the commit the workflow runs from, and the tree under test out at another ref.

A suite whose directory does not exist FAILS. The one way to let it be missing is the manifest's per-branch `absent_ok` table
(branch -> suite -> reason): the suite is then reported as ABSENT-ALLOWED with its reason and a warning annotation, never as a
pass. An `absent_ok` entry whose suite directory exists in the manifest's tree is stale and fails the run, WHATEVER branch the
entry names, so an allowance cannot outlive the reason for it (a branch merged into another one carries its entry along). A run
in which no suite ran at all fails, whatever the table says.

Exit 0 only when no suite failed and at least one ran. Stdlib only, Python 3.7 syntax.
"""

import argparse
import fnmatch
import json
import os
import posixpath
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, 'vllm030_suites.json')
REQUIRED_TESTS = os.path.join(HERE, 'required_tests.py')

PASS = 'PASS'
FAIL = 'FAIL'
ABSENT_ALLOWED = 'ABSENT-ALLOWED'

STALE_ROW = '(absent_ok)'
DEFAULT_TIMEOUT_S = 900
KILL_GRACE_S = 3

IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
VERDICT = re.compile(r'^REQUIRED TESTS: (PASS|FAIL)(.*)$')
RAN = re.compile(r'(\d+) ran')

# The child interpreter. `python -c` puts the working directory first on sys.path and nothing else of ours (a script run by path would
# put scripts/ci, ~2,000 modules, first): the suite's own PYTHONPATH entries then lead. required_tests.py is loaded by file path.
# argv: <required_tests.py> <min tests> <module> ...; VLLM030_SUITE_DIR is the suite directory.
CHILD = """\
import sys
sys.path[:] = [entry for entry in sys.path if entry]
import faulthandler, importlib.util, os, signal
faulthandler.enable()
if hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
spec = importlib.util.spec_from_file_location('required_tests', sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
minimum, modules = sys.argv[2], sys.argv[3:]
code = runner.main(['--min', minimum] + modules)
suite_dir = os.path.realpath(os.environ['VLLM030_SUITE_DIR'])
for name in modules:
    loaded = sys.modules.get(name)
    where = os.path.realpath(getattr(loaded, '__file__', None) or '')
    if os.path.dirname(where) != suite_dir:
        print('REQUIRED TESTS: FAIL: module %s was loaded from %r, not from the suite directory' % (name, where))
        code = 1
sys.stdout.flush()
sys.exit(code)
"""


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
        limit = suite.get('timeout_s', DEFAULT_TIMEOUT_S)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ManifestError('suite %s timeout_s must be an integer of at least 1' % name)
        exclude = suite.get('exclude', {})
        if not isinstance(exclude, dict):
            raise ManifestError('suite %s exclude must be an object of test file -> reason' % name)
        for relative, reason in exclude.items():
            _relative(relative, 'suite %s exclude entry' % name)
            if not relative.endswith('.py') or '\\' in relative:
                raise ManifestError('suite %s exclude entry %r must be a .py file path with "/" separators' % (name, relative))
            if not isinstance(reason, str) or not reason.strip():
                raise ManifestError('suite %s exclude entry %s needs a reason' % (name, relative))
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


def modules_of(directory, pattern, exclude=()):
    """The module names of the test files in `directory` (not below it) that match `pattern`, sorted, less the `exclude`d files."""
    names = []
    for entry in sorted(os.listdir(directory)):
        stem, extension = os.path.splitext(entry)
        if extension == '.py' and fnmatch.fnmatchcase(entry, pattern) and IDENTIFIER.match(stem) \
                and os.path.isfile(os.path.join(directory, entry)) and entry not in exclude:
            names.append(stem)
    return names


def unrun_tests(directory, modules, pattern, exclude=()):
    """The test files below `directory` (as "/" paths relative to it) that the suite would not run: any test_*.py, *_test.py or file
    matching `pattern` that is not one of `modules` and not in `exclude`. __pycache__ and dot directories are not searched."""
    covered = set(module + '.py' for module in modules)
    found = []
    for top, subdirectories, files in os.walk(directory):
        subdirectories[:] = sorted(entry for entry in subdirectories if entry != '__pycache__' and not entry.startswith('.'))
        for name in sorted(files):
            if not name.endswith('.py') or not any(fnmatch.fnmatchcase(name, wanted) for wanted in ('test_*.py', '*_test.py', pattern)):
                continue
            relative = posixpath.join(*os.path.relpath(os.path.join(top, name), directory).split(os.sep))
            if relative not in covered and relative not in exclude:
                found.append(relative)
    return found


def plan_suite(name, suite, root, branch, absent_ok):
    """-> a result dict. status is PASS (still to run: 'modules' and the rest are set), FAIL (final) or ABSENT-ALLOWED (final)."""
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
    pattern = suite.get('glob', 'test_*.py')
    exclude = suite.get('exclude', {})
    gone = sorted(entry for entry in exclude if not os.path.isfile(os.path.join(directory, *entry.split('/'))))
    if gone:
        result['detail'] = 'stale exclude: %s no longer exists in %s (remove it from the suite\'s exclude table)' % (
            ', '.join(gone), suite['dir'])
        return result
    modules = modules_of(directory, pattern, exclude)
    missing = [module for module in suite.get('require', []) if module not in modules]
    if missing:
        result['detail'] = 'required test modules missing from %s: %s' % (suite['dir'], ', '.join(missing))
        return result
    if not modules:
        result['detail'] = 'no test module in %s matches %s' % (suite['dir'], pattern)
        return result
    unrun = unrun_tests(directory, modules, pattern, exclude)
    if unrun:
        result['detail'] = ('test files in %s that this suite would never run: %s (put them where the glob %s finds them, or '
                            'name them with a reason in the suite\'s exclude table)' % (suite['dir'], ', '.join(unrun), pattern))
        return result
    paths = [directory] + [os.path.join(root, entry) for entry in suite.get('pythonpath', [])]
    result.update(status=PASS, modules=modules, paths=paths, root=root, directory=directory,
                  minimum=suite.get('min_tests', 1), timeout=suite.get('timeout_s', DEFAULT_TIMEOUT_S))
    return result


def _pump(stream, verdicts):
    """Copy the child's output to ours line by line, keeping every REQUIRED TESTS verdict line (the last one counts)."""
    for line in stream:
        sys.stdout.write(line)
        sys.stdout.flush()
        if VERDICT.match(line.strip()):
            verdicts.append(line.strip())


def _kill(process):
    """Kill the child and what it started. The child leads its own session (start_new_session), so its group is only its own."""
    try:
        if hasattr(os, 'killpg'):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except OSError:
        pass


def _stop(process):
    """Ask a hung child for its thread stacks (SIGUSR1, see CHILD), give it a moment to write them, then kill it."""
    if hasattr(signal, 'SIGUSR1'):
        try:
            process.send_signal(signal.SIGUSR1)
            time.sleep(KILL_GRACE_S)
        except OSError:
            pass
    _kill(process)


def run_suite(result, python, environment):
    """Run a planned suite; fill in status, ran, seconds and detail."""
    env = dict(environment)
    env['PYTHONPATH'] = os.pathsep.join(result['paths'])  # the caller's own PYTHONPATH (scripts/ci, say) is not passed on
    env['VLLM030_SUITE_DIR'] = result['directory']
    env.setdefault('PYTHONFAULTHANDLER', '1')
    argv = [python, '-B', '-c', CHILD, REQUIRED_TESTS, str(result['minimum'])] + result['modules']
    print('::group::%s: %s (%d modules, at least %d tests, %d s limit)' % (
        result['suite'], result['dir'], len(result['modules']), result['minimum'], result['timeout']))
    sys.stdout.flush()
    verdicts = []
    started = time.monotonic()
    process = subprocess.Popen(argv, cwd=result['root'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, errors='replace', start_new_session=hasattr(os, 'killpg'))
    pump = threading.Thread(target=_pump, args=(process.stdout, verdicts), daemon=True)
    pump.start()
    timed_out = False
    try:
        code = process.wait(timeout=result['timeout'])
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop(process)
        code = process.wait()
    except BaseException:
        _kill(process)
        raise
    pump.join(30)
    if not pump.is_alive():
        process.stdout.close()
    result['seconds'] = round(time.monotonic() - started, 1)
    print('::endgroup::')
    verdict = verdicts[-1] if verdicts else ''
    match = VERDICT.match(verdict)
    if timed_out:
        result.update(status=FAIL, detail='timeout: no result within %d s, the child was stopped (its thread stacks are in the log)'
                      % result['timeout'])
    elif code == 0 and match and match.group(1) == 'PASS':
        ran = RAN.search(match.group(2))
        result.update(status=PASS, ran=int(ran.group(1)) if ran else 0, detail=verdict)
    else:
        result.update(status=FAIL, detail=verdict or 'the child exited %d without a verdict line' % code)
    return result


def stale_excuses(manifest, manifest_root):
    """-> [(branch, suite, directory)] for every absent_ok entry, of ANY branch, whose suite directory exists in `manifest_root`."""
    stale = []
    for branch in sorted(manifest.get('absent_ok', {})):
        for name in sorted(manifest['absent_ok'][branch]):
            directory = manifest['suites'][name]['dir']
            if os.path.isdir(os.path.join(manifest_root, directory)):
                stale.append((branch, name, directory))
    return stale


def run(manifest, root, branch, python, environment, dry_run=False, manifest_root=None):
    """-> (exit code, results). `manifest_root` (default `root`) is the tree the manifest lives in, where stale excuses are looked for."""
    results = []
    for name in sorted(manifest['suites']):
        planned = plan_suite(name, manifest['suites'][name], root, branch, manifest.get('absent_ok', {}))
        if planned['status'] == PASS and not dry_run:
            planned = run_suite(planned, python, environment)
        results.append(planned)
    stale = stale_excuses(manifest, os.path.abspath(manifest_root or root))
    if stale:
        results.append({'suite': STALE_ROW, 'dir': '', 'status': FAIL, 'ran': 0, 'detail': (
            'stale absent_ok: ' + '; '.join('%s excuses %s, but %s exists (remove absent_ok[%s][%s])' % (b, n, d, b, n)
                                           for b, n, d in stale))})
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
    parser.add_argument('--root', default=os.getcwd(), help='the tree under test: the suite directories are relative to it')
    parser.add_argument('--manifest-root', help='the tree the manifest lives in, where stale absent_ok entries are looked for '
                                                '(default --root; differs when the CI scaffolding is checked out apart)')
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
    code, results = run(manifest, root, args.branch, args.python, os.environ, dry_run=args.dry_run,
                        manifest_root=args.manifest_root)
    for line in render(results, args.branch, code, args.dry_run) + annotations(results):
        print(line)
    if args.json_path:
        slim = [{k: v for k, v in r.items() if k not in ('paths', 'root', 'directory')} for r in results]
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
