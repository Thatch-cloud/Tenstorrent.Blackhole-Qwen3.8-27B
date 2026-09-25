"""The M2 report's exit code must carry the verdict.

Mirrors test_lever_n_m1_gate_report.py's discipline (run 35413668471 printed 'NOT
PASSED' and still exited 0): a gate that cannot fail the build is not a gate. The
extra risk here is specific to M2 - a report that is green because both arms
happened to look fine, without ever confirming alternation fired in the m2 arm or
stayed absent in the baseline, would pass while testing nothing.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from os import path

SCRIPT = path.join(path.dirname(path.abspath(__file__)), 'lever_n_m2_gate_report.py')


def run(report):
    handle = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
    with handle:
        if report is not None:
            json.dump(report, handle)
    done = subprocess.Popen([sys.executable, '-B', SCRIPT, '--report', handle.name],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = done.communicate()[0].decode('utf-8', 'replace')
    return done.returncode, out


def arm(ready=True, steady=0.05, overlap=0.4, tokens=96, alternation=0,
        chunk_steps=3, errors=None):
    return dict(ready=ready, steady_gap_median=steady, overlap_gap_max=overlap,
                decode_tokens_received=tokens, decode_thread_alive=False,
                decode_errors=errors or [], alternation_fired=alternation,
                prefill_chunk_steps_seen=chunk_steps)


def passing_controls():
    return dict(m2_ready=True, baseline_ready=True, m2_decode_completed=True,
                baseline_decode_completed=True, m2_prefill_chunked=True,
                baseline_prefill_chunked=True, m2_alternation_fired=True,
                baseline_alternation_absent=True, m2_gap_bounded=True,
                baseline_gap_unbounded=True)


class ExitCodeTests(unittest.TestCase):
    def test_a_passing_gate_exits_zero(self):
        code, out = run(dict(
            m2=arm(overlap=0.4, alternation=3),
            baseline=arm(overlap=6.2, alternation=0),
            m2_bounded=True, baseline_bounded=False, gap_budget_seconds=3.0,
            controls=passing_controls(), gate_passed=True))
        self.assertEqual(code, 0)
        self.assertIn('M2 GATE PASSED', out)

    def test_a_stalling_m2_arm_fails_the_build(self):
        """The whole point of the gate: if M2's own arm still stalls, it must not pass."""
        controls = passing_controls()
        controls['m2_gap_bounded'] = False
        code, out = run(dict(
            m2=arm(overlap=7.0, alternation=3),
            baseline=arm(overlap=6.5, alternation=0),
            m2_bounded=False, baseline_bounded=False, gap_budget_seconds=3.0,
            controls=controls, gate_passed=False))
        self.assertEqual(code, 1)
        self.assertIn('NOT PASSED', out)
        self.assertIn('m2_gap_bounded', out)

    def test_a_baseline_that_does_not_stall_fails_the_build(self):
        """If the baseline arm ALSO stays bounded, the comparison proves nothing -
        either the graft did not actually remove M2, or the test scenario never
        stressed the scheduler in the first place."""
        controls = passing_controls()
        controls['baseline_gap_unbounded'] = False
        code, out = run(dict(
            m2=arm(overlap=0.4, alternation=3),
            baseline=arm(overlap=0.5, alternation=0),
            m2_bounded=True, baseline_bounded=True, gap_budget_seconds=3.0,
            controls=controls, gate_passed=False))
        self.assertEqual(code, 1)
        self.assertIn('baseline_gap_unbounded', out)
        self.assertIn('FAILED', out)

    def test_alternation_never_firing_in_the_m2_arm_fails_the_build(self):
        """Bounded timing with no [M2] alternation log line means the timing was
        bounded for some other reason (e.g. P finished before it could stall D),
        not because alternation worked - the same "equality proves nothing"
        trap M1's baseline_took_one_shot control exists for."""
        controls = passing_controls()
        controls['m2_alternation_fired'] = False
        code, out = run(dict(
            m2=arm(overlap=0.4, alternation=0),
            baseline=arm(overlap=6.0, alternation=0),
            m2_bounded=True, baseline_bounded=False, gap_budget_seconds=3.0,
            controls=controls, gate_passed=False))
        self.assertEqual(code, 1)
        self.assertIn('m2_alternation_fired', out)
        self.assertIn('FAILED', out)

    def test_alternation_present_in_the_baseline_fails_the_build(self):
        """If the "unpatched" baseline still logs [M2] alternation, the graft did
        not actually leave scheduler.py/lane_scheduler.py at stock - the arms are
        not actually different and the comparison is void."""
        controls = passing_controls()
        controls['baseline_alternation_absent'] = False
        code, out = run(dict(
            m2=arm(overlap=0.4, alternation=3),
            baseline=arm(overlap=0.5, alternation=2),
            m2_bounded=True, baseline_bounded=True, gap_budget_seconds=3.0,
            controls=controls, gate_passed=False))
        self.assertEqual(code, 1)
        self.assertIn('baseline_alternation_absent', out)

    def test_a_one_shot_prefill_that_never_chunked_fails_the_build(self):
        """P must actually need multiple chunk steps or the gate tests nothing."""
        controls = passing_controls()
        controls['baseline_prefill_chunked'] = False
        code, out = run(dict(
            m2=arm(overlap=0.4, alternation=3, chunk_steps=3),
            baseline=arm(overlap=0.5, alternation=0, chunk_steps=1),
            m2_bounded=True, baseline_bounded=True, gap_budget_seconds=3.0,
            controls=controls, gate_passed=False))
        self.assertEqual(code, 1)
        self.assertIn('baseline_prefill_chunked', out)

    def test_the_exact_shape_of_an_all_500_run_fails_the_build(self):
        """Both arms ready, every request failed, nothing measured - must not be green."""
        error = 'HTTPError: HTTP Error 500: Internal Server Error'
        code, out = run(dict(
            m2=dict(ready=True, error=error, decode_tokens_received=0,
                    decode_thread_alive=False, decode_errors=[error]),
            baseline=dict(ready=True, error=error, decode_tokens_received=0,
                          decode_thread_alive=False, decode_errors=[error]),
            controls=dict(m2_ready=True, baseline_ready=True, m2_decode_completed=False,
                          baseline_decode_completed=False, m2_prefill_chunked=False,
                          baseline_prefill_chunked=False, m2_alternation_fired=False,
                          baseline_alternation_absent=True, m2_gap_bounded=False,
                          baseline_gap_unbounded=False),
            gate_passed=False))
        self.assertEqual(code, 1)
        self.assertIn('NOT PASSED', out)
        self.assertIn('500', out)

    def test_a_missing_report_fails_the_build(self):
        code, out = run(None)
        self.assertEqual(code, 1)
        self.assertIn('No structured report', out)

    def test_controls_are_reported_when_the_gate_passes(self):
        code, out = run(dict(
            m2=arm(overlap=0.4, alternation=3),
            baseline=arm(overlap=6.2, alternation=0),
            m2_bounded=True, baseline_bounded=False, gap_budget_seconds=3.0,
            controls=passing_controls(), gate_passed=True))
        self.assertEqual(code, 0)
        self.assertIn('control', out)
        self.assertNotIn('FAILED', out)


if __name__ == '__main__':
    unittest.main()
