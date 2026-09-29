"""Run the named unittest targets and pass only if every test in them RAN: a skip fails here.

For a check that has to exercise something to mean anything - the sticky-session scheduler graft and
oracle on the serving image's own vLLM (the c2 serving workflow's probe step) - where the tests skip
themselves when vLLM is not importable (test_qwen_prefix_scheduler_vllm), so an import problem in the
image would otherwise read as a green step.

    python3 -B required_tests.py [--min N] TARGET [TARGET ...]

TARGET is anything unittest.TestLoader.loadTestsFromName takes (module, module.Class,
module.Class.test). Exit 0 only when at least N tests ran (default 1), none failed, errored or was
skipped, and every target loaded at least one test; the last line says which. Stdlib only, Python 3.7
syntax.
"""

import argparse
import sys
import unittest


def run(targets, minimum=1, stream=None):
    """-> (exit code, verdict line)."""
    stream = sys.stdout if stream is None else stream
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for target in targets:
        tests = loader.loadTestsFromName(target)
        if not tests.countTestCases():
            return 1, 'REQUIRED TESTS: FAIL: %s loads no test' % target
        suite.addTests(tests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    problems = []
    if result.skipped:
        problems.append('%d skipped (%s)' % (len(result.skipped), '; '.join(
            '%s: %s' % (test.id(), reason) for test, reason in result.skipped[:8])))
    if not result.wasSuccessful():
        problems.append('%d failed, %d errors, %d unexpected successes' % (
            len(result.failures), len(result.errors), len(result.unexpectedSuccesses)))
    if result.testsRun < minimum:
        problems.append('%d ran, at least %d required' % (result.testsRun, minimum))
    if problems:
        return 1, 'REQUIRED TESTS: FAIL: ' + '; '.join(problems)
    return 0, 'REQUIRED TESTS: PASS: %d ran, none skipped' % result.testsRun


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--min', type=int, default=1, dest='minimum', help='the fewest tests that must run')
    parser.add_argument('targets', nargs='+')
    args = parser.parse_args(argv)
    code, verdict = run(args.targets, args.minimum)
    print(verdict)
    sys.stdout.flush()
    return code


if __name__ == '__main__':
    sys.exit(main())
