"""The report's exit code must carry the verdict.

Run 35413668471 printed 'M1 GATE NOT PASSED (0 lengths checked)' and still exited 0,
so the job went green on a gate that compared nothing. A gate that cannot fail the
build is not a gate, and these tests are what stop that regressing.
"""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from os import path

SCRIPT = path.join(path.dirname(path.abspath(__file__)), 'lever_n_m1_gate_report.py')


def run(report):
    """Invoke the reporter as CI does and return (exit code, stdout)."""
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    with handle:
        if report is not None:
            json.dump(report, handle)
    done = subprocess.Popen([sys.executable, '-B', SCRIPT, '--report', handle.name],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = done.communicate()[0].decode('utf-8', 'replace')
    return done.returncode, out


def comparison(identical):
    return dict(name='approx_400', both_present=True, identical=identical,
                prompt_tokens=460, baseline_len=10, resumable_len=10,
                baseline_seconds=1.0, resumable_seconds=1.0)


class ExitCodeTests(unittest.TestCase):
    def test_a_passing_gate_exits_zero(self):
        code, out = run(dict(gate_passed=True, lengths_checked=1,
                             comparisons=[comparison(True)]))
        self.assertEqual(code, 0)
        self.assertIn('M1 GATE PASSED', out)

    def test_a_diverging_gate_fails_the_build(self):
        code, out = run(dict(gate_passed=False, lengths_checked=1,
                             comparisons=[comparison(False)]))
        self.assertEqual(code, 1)
        self.assertIn('M1 GATE NOT PASSED', out)

    def test_the_exact_shape_of_run_35413668471_fails_the_build(self):
        """Both arms ready, every request 500, nothing compared - must not be green."""
        error = 'HTTPError: HTTP Error 500: Internal Server Error'
        code, out = run(dict(context=16384, chunk_size=2048, targets=[400, 3000, 5000],
                             gate_passed=False, lengths_checked=0,
                             baseline=dict(ready=True, completions=[], error=error),
                             resumable=dict(ready=True, completions=[], error=error),
                             comparisons=[dict(name='approx_400', both_present=False)]))
        self.assertEqual(code, 1)
        self.assertIn('NOT PASSED', out)
        self.assertIn('500', out)

    def test_zero_lengths_checked_is_never_a_pass(self):
        """gate_passed already requires a nonempty comparison set; hold that line."""
        code, _ = run(dict(gate_passed=False, lengths_checked=0, comparisons=[]))
        self.assertEqual(code, 1)

    def test_a_missing_report_fails_the_build(self):
        code, out = run(None)
        self.assertEqual(code, 1)
        self.assertIn('No structured report', out)


if __name__ == '__main__':
    unittest.main()
