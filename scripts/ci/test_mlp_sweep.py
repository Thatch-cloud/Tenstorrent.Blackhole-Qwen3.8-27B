import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("sweep", Path(__file__).with_name("mlp-sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class SweepTests(unittest.TestCase):
    def projection_report(self):
        return dict(passed=True, seeds=[123, 456, 789], candidates={"fused-91": dict(
            exact_control=True, eligible_for_mlp_gate=True, kernel=dict(pairs_per_worker=3, workers=91),
            blocks=[dict(control_ms=.2, fused_ms=.18) for _ in range(3)])},
            checks=[dict(mode=mode + "fused-91", seed=seed, chip=chip, exact=True)
                    for mode in ("eager-", "trace-") for seed in (123, 456, 789) for chip in (0, 1)])

    def test_projection_prerequisite_selects_only_eligible_mapping(self):
        report = self.projection_report()
        self.assertEqual(sweep.projection_candidates(report)[1]["pairs_per_worker"], 3)
        report["candidates"]["fused-91"]["eligible_for_mlp_gate"] = False
        self.assertEqual(len(sweep.projection_candidates(report)), 1)

    def test_projection_prerequisite_checks_all_shards_and_modes(self):
        for missing in range(12):
            report = self.projection_report()
            del report["checks"][missing]
            with self.assertRaises(ValueError):
                sweep.projection_candidates(report)

    def test_projection_prerequisite_rechecks_timings(self):
        for elapsed in (.196, .2, float("nan"), 0):
            report = self.projection_report()
            report["candidates"]["fused-91"]["blocks"][2]["fused_ms"] = elapsed
            with self.assertRaises(ValueError):
                sweep.projection_candidates(report)

    def test_projection_prerequisite_rejects_failed_run(self):
        report = self.projection_report()
        report["passed"] = False
        with self.assertRaises(ValueError):
            sweep.projection_candidates(report)

    def test_fusion_candidates_are_separate(self):
        candidates = sweep.candidates(fusion=True)
        self.assertEqual(candidates[0]["name"], "control")
        self.assertEqual([candidate["nblock"] for candidate in candidates[1:]], [8, 16, 32])
        with self.assertRaises(ValueError):
            sweep.candidates(packing=True, fusion=True)

    def test_candidates_and_frozen_geometry(self):
        candidates = sweep.candidates()
        self.assertEqual(len(candidates), 15)
        self.assertEqual(len({candidate["name"] for candidate in candidates}), 15)
        control = sweep.geometry(5120, 8704, 44, 8)
        self.assertEqual(control["compute_with_storage_grid_size"], (11, 4))
        self.assertEqual(control["per_core_N"], 7)
        self.assertEqual(control["out_subblock_w"], 1)
        for candidate in candidates:
            for inner, output, cores in ((5120, 8704, candidate["gate"]), (8704, 5120, candidate["down"])):
                geometry = sweep.geometry(inner, output, cores, candidate["block"])
                self.assertLessEqual(geometry["compute_with_storage_grid_size"][1], 10)
                self.assertLessEqual(geometry["out_subblock_w"], 4)
                self.assertEqual(geometry["per_core_N"] % geometry["out_subblock_w"], 0)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            sweep.geometry(5120, 8704, 121, 8)
        with self.assertRaises(ValueError):
            sweep.geometry(5120, 8704, 44, 3)

    def test_explicit_packing_covers_output_and_preserves_subblock_limit(self):
        self.assertEqual(len(sweep.candidates(packing=True)), 7)
        for tiles in (8, 12, 16):
            config = sweep.geometry(5120, 8704, 44, 8, tiles)
            self.assertEqual(config["per_core_N"], tiles)
            self.assertEqual(config["out_subblock_w"], 4)
        with self.assertRaises(ValueError):
            sweep.geometry(5120, 8704, 44, 8, 6)

    def test_paired_latency_sign(self):
        result = sweep.paired_summary([dict(control_ms=1, candidate_ms=.9), dict(control_ms=2, candidate_ms=1.8)])
        self.assertAlmostEqual(result["mean_latency_change"], -.1)


if __name__ == "__main__":
    unittest.main()
