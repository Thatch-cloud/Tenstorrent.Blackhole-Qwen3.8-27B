"""Host-only tests for the fabric-relay tuning framework."""

import unittest

import abba
import core_inventory
import fabric_bench
import gates
import load_timeline
from report_gen import render
from validate_spec import parse_simple_yaml
from pathlib import Path

SPEC = parse_simple_yaml((Path(__file__).with_name("spec") / "phases.yaml").read_text(encoding="utf-8"))


class AbbaTests(unittest.TestCase):
    def test_order_alternates_and_starts_with_control(self):
        order = abba.abba_order(9)
        self.assertEqual(order[0], "control")
        self.assertEqual(len(order), 9)

    def test_even_blocks_rejected(self):
        with self.assertRaises(ValueError):
            abba.abba_order(8)

    def test_candidate_losing_one_block_not_promotable(self):
        pairs = [(1.0, 0.9)] * 8 + [(1.0, 1.5)]
        result = abba.evaluate(pairs)
        self.assertEqual(result["losses"], 1)
        self.assertFalse(result["promotable"])
        self.assertFalse(result["all_blocks_not_worse"])

    def test_uniform_wins_promotable(self):
        self.assertTrue(abba.evaluate([(1.0, 0.9)] * 9)["promotable"])


class TimelineTests(unittest.TestCase):
    def events(self, upload_card1=100.0):
        starts = {"read": 0.0, "deserialize": 10.0, "convert": 20.0, "upload_card0": 30.0, "upload_card1": 40.0}
        return [{"stage": stage, "card": 0, "started_s": start, "ended_s": start + 10.0}
                for stage, start in starts.items()]

    def test_share_and_stages(self):
        result = load_timeline.timeline(self.events())
        self.assertEqual(set(result["per_stage_s"]), set(load_timeline.REQUIRED_STAGES))
        self.assertEqual(result["per_stage_s"]["upload_card1"], 10.0)
        self.assertAlmostEqual(result["card1_upload_share"], 10.0 / result["total_s"])

    def test_missing_stage_rejected(self):
        with self.assertRaises(ValueError):
            load_timeline.timeline(self.events()[:4])


class GateTests(unittest.TestCase):
    def phase(self, phase_id):
        return next(p for p in SPEC["phases"] if p["id"] == phase_id)

    def test_missing_gate_evidence_fails_closed(self):
        report = gates.evaluate_phase(self.phase("P1"), {})
        self.assertFalse(report["all_passed"])

    def test_digest_equality_pass(self):
        report = gates.evaluate_phase(self.phase("P1"), {
            "digests": {"mlp": True, "gdn": True},
            "output_digest_equal": True,
            "state_digest_equal": True,
            "control_metric": 365.0, "candidate_metric": 120.0,
        })
        self.assertTrue(report["all_passed"])

    def test_latency_budget_bound(self):
        digests = {"output_digest_equal": True, "state_digest_equal": True}
        report = gates.evaluate_phase(self.phase("P2.b"), {**digests, "added_ms_per_block": 0.7})
        self.assertTrue(report["all_passed"])
        report = gates.evaluate_phase(self.phase("P2.b"), {**digests, "added_ms_per_block": 1.4})
        self.assertFalse(report["all_passed"])

    def test_baseline_stage_gate(self):
        evidence = {"per_stage_s": {s: 1.0 for s in load_timeline.REQUIRED_STAGES}, "total_s": 5.0}
        self.assertTrue(gates.evaluate_phase(self.phase("P0.3"), evidence)["all_passed"])


class InventoryTests(unittest.TestCase):
    def test_reclaimable_counts_card1_reserved(self):
        cluster = {"0": {"tensix": ["a", "b", "c"], "dispatch_reserved": ["a"]},
                   "1": {"tensix": ["a", "b", "c", "d"], "dispatch_reserved": ["a", "b"]}}
        result = core_inventory.inventory(cluster)
        self.assertEqual(result["reclaimable_card1"], 2)
        self.assertEqual(result["1"]["compute_available"], 2)

    def test_reserved_outside_tensix_rejected(self):
        with self.assertRaises(ValueError):
            core_inventory.inventory({"1": {"tensix": [], "dispatch_reserved": ["x"]}})


class BenchTests(unittest.TestCase):
    def test_summary_needs_nine_repeats(self):
        with self.assertRaises(ValueError):
            fabric_bench.summarize([1.0] * 8)
        summary = fabric_bench.summarize([float(i) for i in range(1, 10)])
        self.assertEqual(summary["repeats"], 9)
        self.assertEqual(summary["p50_GiB_s"], 5.0)


class ReportTests(unittest.TestCase):
    def test_failed_gate_rendering_marks_not_promoted(self):
        phase = next(p for p in SPEC["phases"] if p["id"] == "P1")
        record = render(SPEC, "P1", {})
        self.assertIn("not promoted", record.lower())
        for gate in phase["gates"]:
            self.assertIn(gate["id"], record)


if __name__ == "__main__":
    unittest.main()
