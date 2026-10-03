"""Host-only admission tests; no device or simulator execution."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gdn_shared_qk_gate as gate


class AdmissionTests(unittest.TestCase):
    def fixture(self, root):
        directory, runtime = root / "ci", root / "runtime"
        sources = {}
        for name in gate.LOCAL_SOURCES:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            sources["/experiment-scripts/ci/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
        native = [gate.KERNEL_ROOT + "/" + name for name in gate.HASHES]
        native += ["tt_metal/hw/inc/api/compute/" + name for name in gate.API_SOURCES]
        for name in native:
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            sources["/opt/tt-metal/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
        def matrix(operands):
            return [dict(mode=mode, operand=operand, chip=chip, exact=True)
                for mode in ("eager", "replay_1", "replay_2", "replay_0")
                for operand in range(operands) for chip in (0, 1)]
        report = dict(passed=True, closed_cleanly=True, norm_unchanged=True,
            state_math_unchanged=True, shared_qk_preparation=True, backend="simulator",
            rows=16, checks=matrix(3), immutable_checks=matrix(6),
            hardware_qualified=False, timing_qualified=False, sources=sources,
            sources_after=dict(sources))
        return directory, runtime, report

    def exercise(self, mutation=None, source_change=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, runtime, report = self.fixture(root)
            if mutation:
                mutation(report)
            path = root / "report.json"
            payload = json.dumps(report).encode()
            path.write_bytes(payload)
            if source_change:
                (directory / "gdn_shared_qk_pipeline.py").write_bytes(b"changed")
            with patch.object(gate, "REPORT_SHA256", hashlib.sha256(payload).hexdigest()):
                return gate.qualify(path, directory, runtime)

    def test_complete_gate(self):
        self.assertFalse(self.exercise()["hardware_qualified"])

    def test_missing_comparison(self):
        with self.assertRaises(ValueError):
            self.exercise(lambda report: report["checks"].pop())

    def test_changed_pipeline(self):
        with self.assertRaises(ValueError):
            self.exercise(source_change=True)

    def test_changed_sources_after(self):
        with self.assertRaises(ValueError):
            self.exercise(lambda report: report["sources_after"].clear())

    def test_false_state_math_claim(self):
        with self.assertRaises(ValueError):
            self.exercise(lambda report: report.update(state_math_unchanged=False))

    def test_unretained_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                gate.qualify(path, temporary, temporary)
