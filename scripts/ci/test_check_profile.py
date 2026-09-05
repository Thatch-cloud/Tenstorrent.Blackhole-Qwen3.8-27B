"""Require numerical success and actual two-chip timing evidence."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("profile_check", Path(__file__).with_name("check-profile.py"))
profile_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_check)


class ProfileTests(unittest.TestCase):
    def run_case(self, rows, log="", failures=0):
        with tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
            root = Path(directory)
            (root / "tests.xml").write_text(f'<testsuites><testsuite tests="1" failures="{failures}" /></testsuites>')
            (root / "console.log").write_text(log)
            (root / "ops_perf_results.csv").write_text("DEVICE ID,OP CODE,DEVICE KERNEL DURATION [ns]\n" + rows)
            profile_check.check(root)

    def test_two_chip_timings_pass(self):
        self.run_case("0,matmul,123\n1,matmul,125\n")

    def test_host_only_or_empty_report_fails(self):
        for rows in ("", "0,matmul,nan\n1,matmul,\n", "0,matmul,123\n"):
            with self.assertRaises(AssertionError):
                self.run_case(rows)

    def test_dropped_markers_fail(self):
        with self.assertRaises(AssertionError):
            self.run_case("0,matmul,123\n1,matmul,125\n", log="markers were dropped")

    def test_failed_numerics_fail(self):
        with self.assertRaises(AssertionError):
            self.run_case("0,matmul,123\n1,matmul,125\n", failures=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
