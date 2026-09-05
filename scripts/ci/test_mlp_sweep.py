import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("sweep", Path(__file__).with_name("mlp-sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class SweepTests(unittest.TestCase):
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
