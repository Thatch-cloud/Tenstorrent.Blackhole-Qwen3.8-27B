"""The M3native report's exit code must carry the verdict.

Mirrors test_lever_n_m1_gate_report.py / test_lever_n_m2_gate_report.py's discipline:
a report that prints 'NOT PASSED' and still exits 0 is not a gate. The extra risk
here is specific to M3native - a report that is green because the four streams
happened to match their references, without ever confirming the [PINDIAG] native_m3
marker fired or that the retired MLP/GDN-output wrappers made zero calls, would pass
while testing a two-call fallback instead of the native path this graft exists to
prove.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from os import path

SCRIPT = path.join(path.dirname(path.abspath(__file__)), 'lever_n_m3native_gate_report.py')


def run(report):
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    with handle:
        if report is not None:
            json.dump(report, handle)
    done = subprocess.Popen([sys.executable, '-B', SCRIPT, '--report', handle.name],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = done.communicate()[0].decode('utf-8', 'replace')
    return done.returncode, out


def comparison(user, base=None, present=True, identical=True, reference_len=64, actual_len=256, error=None):
    entry = dict(user=user, prompt_base=base if base is not None else 1000 + user,
                reference_present=present)
    if present:
        entry.update(reference_path='ref-%d.json' % user, reference_len=reference_len,
                     actual_len=actual_len, identical_prefix=identical)
        if error:
            entry['error'] = error
    return entry


def passing_report(**overrides):
    report = dict(
        users=4, context=33024, prompt_tokens=32768, prompt_base=1000, prompt_user_offset=1,
        ready=True, references_loaded=[1000, 1001, 1002, 1003], users_checked=4,
        comparisons=[comparison(index) for index in range(4)],
        packed_phase=dict(rounds=12, trace_ms_min=850.1, trace_ms_mean=910.4, trace_ms_max=980.2),
        native_m3_marker_present=True, retired_binder_rounds_observed=12,
        retired_binder_calls_nonzero=[], gate_passed=True)
    report.update(overrides)
    return report


class ExitCodeTests(unittest.TestCase):
    def test_a_passing_gate_exits_zero(self):
        code, out = run(passing_report())
        self.assertEqual(code, 0)
        self.assertIn('M3NATIVE GATE PASSED', out)
        self.assertIn('[PACKED-PHASE] trace_ms rounds=12 min=850.1 mean=910.4 max=980.2', out)

    def test_a_missing_reference_fails_the_build(self):
        report = passing_report(gate_passed=False, users_checked=3,
                                comparisons=[comparison(index) for index in range(3)]
                                + [comparison(3, present=False)])
        code, out = run(report)
        self.assertEqual(code, 1)
        self.assertIn('NO REFERENCE', out)
        self.assertIn('NOT PASSED', out)

    def test_a_diverged_stream_fails_the_build(self):
        report = passing_report(gate_passed=False,
                                comparisons=[comparison(0, identical=False)]
                                + [comparison(index) for index in range(1, 4)])
        code, out = run(report)
        self.assertEqual(code, 1)
        self.assertIn('DIVERGED', out)

    def test_a_missing_native_m3_marker_fails_the_build(self):
        """Bytes matching for some other reason is not the same as the graft engaging."""
        report = passing_report(gate_passed=False, native_m3_marker_present=False)
        code, out = run(report)
        self.assertEqual(code, 1)
        self.assertIn('native_m3 marker present: False', out)

    def test_leaked_retired_binder_calls_fail_the_build(self):
        """The exact failure mode the overlay switch's zero-call assertion targets: a
        silent fallback to the two-call MLP/GDN-output wrappers under the graft."""
        leaked = [dict(mlp=0, gdn_output=0), dict(mlp=2, gdn_output=0)]
        report = passing_report(gate_passed=False, retired_binder_calls_nonzero=leaked)
        code, out = run(report)
        self.assertEqual(code, 1)
        self.assertIn('retired-binder calls LEAKED', out)
        self.assertIn("'mlp': 2", out)

    def test_no_packed_phase_rounds_observed_is_reported(self):
        report = passing_report(gate_passed=False, packed_phase=None)
        code, out = run(report)
        self.assertEqual(code, 1)
        self.assertIn('[PACKED-PHASE] no rounds observed', out)

    def test_the_exact_shape_of_a_server_that_never_became_ready_fails_the_build(self):
        report = dict(users=4, context=33024, prompt_tokens=32768, prompt_base=1000,
                     prompt_user_offset=1, ready=False,
                     fatal='TimeoutError: readiness exceeded 900s', gate_passed=False)
        code, out = run(report)
        self.assertEqual(code, 1)
        self.assertIn('FATAL', out)
        self.assertIn('NOT PASSED', out)

    def test_a_missing_report_fails_the_build(self):
        code, out = run(None)
        self.assertEqual(code, 1)
        self.assertIn('No structured report', out)

    def test_users_checked_short_of_users_is_visible_even_if_every_present_one_matched(self):
        """A report someone hand-edited to drop a divergent user's comparison entirely,
        rather than mark it, must still be readable as short."""
        report = passing_report(users_checked=3)
        code, out = run(report)
        self.assertIn('users_checked=3/4', out)


if __name__ == '__main__':
    unittest.main()
