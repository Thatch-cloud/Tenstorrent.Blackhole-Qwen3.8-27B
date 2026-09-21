"""CPU-only unit tests for the pure parts of matmul64_sweep.py (shape arithmetic, grid
enumeration, config building, report assembly). No ttnn import anywhere in this file -
matmul64_sweep.py defers every ttnn/torch/model import to the device-dependent
functions (load_model_dims, compute_kernel_config, model_progcfg, dram_sharded_progcfg,
make_weight, make_activation, time_matmul, run_shape, main), none of which are exercised
here. Run with: py -3.10 -B -m unittest test_matmul64_sweep
"""

import math
import unittest

import matmul64_sweep as sweep


# A plausible Qwen3.x-27B dim set (dim=5120, hidden_dim=17408 match mlp-sweep.py's own
# asserted values; head counts are round numbers chosen only to exercise the TP
# arithmetic - the real numbers come from load_model_dims() at runtime, never from
# this file).
FAKE_DIMS = dict(
    dim=5120, hidden_dim=17408, n_heads=40, n_kv_heads=8, head_dim=128,
    gdn_nk=16, gdn_nv=32, gdn_dk=128, gdn_dv=128,
)


class TilesAndDivisorsTests(unittest.TestCase):
    def test_tiles_exact_multiple(self):
        self.assertEqual(sweep.tiles(5120), 160)
        self.assertEqual(sweep.tiles(32), 1)

    def test_tiles_rejects_non_multiple(self):
        with self.assertRaises(ValueError):
            sweep.tiles(33)
        with self.assertRaises(ValueError):
            sweep.tiles(0)

    def test_divisors(self):
        self.assertEqual(sweep.divisors(12), [1, 2, 3, 4, 6, 12])
        self.assertEqual(sweep.divisors(1), [1])

    def test_divisors_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            sweep.divisors(0)


class TpDimArithmeticTests(unittest.TestCase):
    def test_attn_tp_dims_tp2(self):
        # n_local_heads=20, kv_dim_per_device = max(1,8//2)*128 = 4*128=512
        # qkv_fused_dim_tp = 20*128*2 + 2*512 = 5120 + 1024 = 6144
        # out_dim_tp = 40*128//2 = 2560
        dims = sweep.attn_tp_dims(40, 8, 128, tp=2)
        self.assertEqual(dims, dict(attn_qkv_fused_dim_tp=6144, attn_out_dim_tp=2560))

    def test_attn_tp_dims_kv_replication_floor(self):
        # n_kv_heads=1 at tp=4 -> n_local_kv_heads floors at 1, not 0
        dims = sweep.attn_tp_dims(8, 1, 128, tp=4)
        self.assertEqual(dims["attn_qkv_fused_dim_tp"], (8 // 4) * 128 * 2 + 2 * 1 * 128)

    def test_attn_tp_dims_rejects_indivisible_heads(self):
        with self.assertRaises(ValueError):
            sweep.attn_tp_dims(41, 8, 128, tp=2)

    def test_gdn_tp_dims_tp2(self):
        # key_dim=16*128=2048, value_dim=32*128=4096, qkv_dim=2*2048+4096=8192
        # z_dim=4096, nv_tp=16, qkvz_dim_tp=(8192+4096)//2=6144
        # qkvzab_dim_tp = 6144 + 2*16 = 6176; value_dim_tp = 4096//2=2048
        dims = sweep.gdn_tp_dims(16, 32, 128, 128, tp=2)
        self.assertEqual(dims, dict(gdn_qkvzab_dim_tp=6176, gdn_value_dim_tp=2048))

    def test_gdn_tp_dims_rejects_indivisible_heads(self):
        with self.assertRaises(ValueError):
            sweep.gdn_tp_dims(15, 32, 128, 128, tp=2)


class ShapeTableTests(unittest.TestCase):
    def test_seven_shapes_in_model_config_order(self):
        table = sweep.shape_table(FAKE_DIMS, tp=2)
        names = [shape["name"] for shape in table]
        self.assertEqual(names, ["mlp_w1", "mlp_w3", "mlp_w2", "attn_qkv_fused",
                                 "gdn_qkvz", "attn_wo", "gdn_out"])

    def test_mlp_shapes_use_hidden_over_tp(self):
        table = {shape["name"]: shape for shape in sweep.shape_table(FAKE_DIMS, tp=2)}
        hidden_tp = FAKE_DIMS["hidden_dim"] // 2
        self.assertEqual((table["mlp_w1"]["K"], table["mlp_w1"]["N"]), (FAKE_DIMS["dim"], hidden_tp))
        self.assertEqual((table["mlp_w3"]["K"], table["mlp_w3"]["N"]), (FAKE_DIMS["dim"], hidden_tp))
        self.assertEqual((table["mlp_w2"]["K"], table["mlp_w2"]["N"]), (hidden_tp, FAKE_DIMS["dim"]))
        self.assertTrue(table["mlp_w1"]["silu"])
        self.assertFalse(table["mlp_w3"]["silu"])

    def test_only_attn_qkv_fused_omits_grid_w(self):
        table = {shape["name"]: shape for shape in sweep.shape_table(FAKE_DIMS, tp=2)}
        for name, shape in table.items():
            expected = name != "attn_qkv_fused"
            self.assertEqual(shape["grid_w_pinned"], expected, name)

    def test_num_cores_matches_model_config(self):
        table = {shape["name"]: shape for shape in sweep.shape_table(FAKE_DIMS, tp=2)}
        self.assertEqual(table["mlp_w1"]["num_cores"], 44)
        self.assertEqual(table["mlp_w3"]["num_cores"], 44)
        self.assertEqual(table["mlp_w2"]["num_cores"], 33)
        self.assertEqual(table["attn_qkv_fused"]["num_cores"], 64)
        self.assertEqual(table["gdn_qkvz"]["num_cores"], 44)
        self.assertEqual(table["attn_wo"]["num_cores"], 33)
        self.assertEqual(table["gdn_out"]["num_cores"], 33)

    def test_missing_dims_raise(self):
        incomplete = dict(FAKE_DIMS)
        del incomplete["head_dim"]
        with self.assertRaises(ValueError):
            sweep.shape_table(incomplete)


class DramFloorTests(unittest.TestCase):
    def test_weight_bytes_includes_bfp8_overhead(self):
        # one exact 32x32 tile: 1024 elements * 1.0625 bytes/element = 1088 bytes
        self.assertAlmostEqual(sweep.weight_bytes(32, 32), 1088.0)

    def test_dram_floor_scales_with_bytes_and_bandwidth(self):
        floor_400 = sweep.dram_floor_us(5120, 8704, bandwidth_gbps=400.0)
        floor_800 = sweep.dram_floor_us(5120, 8704, bandwidth_gbps=800.0)
        self.assertAlmostEqual(floor_400 / 2, floor_800)
        expected = (5120 * 8704 * sweep.BFP8_BYTES_PER_ELEMENT) / 400e9 * 1e6
        self.assertAlmostEqual(floor_400, expected)

    def test_dram_floor_default_reads_current_global_at_call_time(self):
        original = sweep.DRAM_BANDWIDTH_GBPS
        try:
            sweep.DRAM_BANDWIDTH_GBPS = 200.0
            floor_200 = sweep.dram_floor_us(5120, 8704)
            sweep.DRAM_BANDWIDTH_GBPS = 400.0
            floor_400 = sweep.dram_floor_us(5120, 8704)
        finally:
            sweep.DRAM_BANDWIDTH_GBPS = original
        self.assertAlmostEqual(floor_200, floor_400 * 2)


class GridEnumerationTests(unittest.TestCase):
    def test_legal_grids_stay_within_worker_bounds(self):
        for x, y in sweep.legal_grids(160):
            self.assertLessEqual(x, sweep.WORKER_GRID_X)
            self.assertLessEqual(y, sweep.WORKER_GRID_Y)

    def test_legal_grids_cores_never_exceed_n_tiles(self):
        # Ceiling-based per_core_N (see legal_grids' docstring) only needs cores<=n_tiles
        # so every core gets at least one output tile column; it no longer requires
        # exact division (272/44, the model's own mlp_w1/w3 grid, is not exact either).
        n_tiles = 272
        for x, y in sweep.legal_grids(n_tiles):
            self.assertLessEqual(x * y, n_tiles)

    def test_legal_grids_excludes_grids_wider_than_n_tiles(self):
        n_tiles = 100  # the full 13x10=130-core worker grid exceeds this
        grids = sweep.legal_grids(n_tiles)
        self.assertNotIn((13, 10), grids)
        self.assertIn((10, 10), grids)  # 100 cores == n_tiles is still legal

    def test_legal_grids_includes_prime_tile_count(self):
        # gdn_qkvz's real N (6176) is 193 tiles, and 193 is prime - under the old
        # exact-division rule only the trivial 1x1 grid was legal for a shape like
        # this. Every grid up to the worker bound should now be legal (193 > 130).
        n_tiles = 193
        self.assertEqual(len(sweep.legal_grids(n_tiles)),
                         sum(1 for x in range(1, 14) for y in range(1, 11)))

    def test_select_grids_quick_includes_legal_named_grids(self):
        n_tiles = 193  # prime; every named grid (max 130 cores) is now legal
        grids = sweep.select_grids(n_tiles, quick=True)
        for named in sweep.NAMED_GRIDS:
            self.assertIn(named, grids)

    def test_select_grids_quick_respects_budget(self):
        n_tiles = 2 ** 10  # many legal divisors
        grids = sweep.select_grids(n_tiles, quick=True)
        self.assertLessEqual(len(grids), 24)

    def test_select_grids_full_is_superset_of_quick(self):
        n_tiles = 160
        quick = set(sweep.select_grids(n_tiles, quick=True))
        full = set(sweep.select_grids(n_tiles, quick=False))
        self.assertTrue(quick <= full)

    def test_select_grids_includes_extra_when_legal(self):
        n_tiles = 160
        extra_grid = (10, 4)  # 40 cores <= 160 tiles
        grids = sweep.select_grids(n_tiles, quick=True, extra=(extra_grid,))
        self.assertIn(extra_grid, grids)

    def test_select_grids_ignores_illegal_extra(self):
        n_tiles = 100
        illegal = (13, 10)  # 130 cores > 100 tiles
        grids = sweep.select_grids(n_tiles, quick=True, extra=(illegal,))
        self.assertNotIn(illegal, grids)


class BlockAndSubblockTests(unittest.TestCase):
    def test_block_choices_quick_are_divisors(self):
        k_tiles = 160
        for block in sweep.block_choices(k_tiles, quick=True):
            self.assertEqual(k_tiles % block, 0)

    def test_block_choices_quick_subset_of_full(self):
        k_tiles = 272
        self.assertTrue(set(sweep.block_choices(k_tiles, quick=True))
                        <= set(sweep.block_choices(k_tiles, quick=False)))

    def test_block_choices_always_includes_full_k_block(self):
        k_tiles = 3  # none of 2/4/8/16 divide 3; 1 and k_tiles itself always qualify
        self.assertEqual(sweep.block_choices(k_tiles, quick=True), [1, 3])

    def test_out_subblocks_respect_cap(self):
        for h, w in sweep.out_subblocks(2, 8, cap=4):
            self.assertLessEqual(h * w, 4)
            self.assertEqual(2 % h, 0)
            self.assertEqual(8 % w, 0)

    def test_out_subblocks_sorted_largest_first(self):
        pairs = sweep.out_subblocks(2, 8, cap=4)
        products = [h * w for h, w in pairs]
        self.assertEqual(products, sorted(products, reverse=True))

    def test_out_subblocks_raises_when_impossible(self):
        # per_core_m and per_core_n both 1 -> only (1,1) fits, never empty; force
        # impossible by requiring cap below the smallest achievable product.
        with self.assertRaises(ValueError):
            sweep.out_subblocks(2, 3, cap=0)


class BuildConfigsTests(unittest.TestCase):
    def test_per_core_m_matches_ceil_m_over_tile(self):
        for M, expected in ((1, 1), (32, 1), (33, 2), (64, 2)):
            configs = sweep.build_configs(5120, 5120, M, quick=True)
            self.assertTrue(configs)
            for cfg in configs:
                self.assertEqual(cfg["per_core_M"], expected)

    def test_configs_are_unique_grid_block_pairs(self):
        configs = sweep.build_configs(5120, 8704, 64, quick=True)
        keys = [(cfg["grid"], cfg["in0_block_w"]) for cfg in configs]
        self.assertEqual(len(keys), len(set(keys)))

    def test_quick_caps_near_24(self):
        configs = sweep.build_configs(5120, 8704, 64, quick=True)
        self.assertLessEqual(len(configs), 24)

    def test_full_covers_every_legal_grid(self):
        K, N, M = 5120, 5120, 64
        configs = sweep.build_configs(K, N, M, quick=False)
        grids_used = {cfg["grid"] for cfg in configs}
        self.assertEqual(grids_used, set(sweep.legal_grids(sweep.tiles(N))))

    def test_per_core_n_is_ceiling_and_covers_n_tiles(self):
        # 8704 -> 272 tiles is the model's REAL mlp_w1/w3 shape; per_core_N must be
        # the ceiling (not the floor) so cores*per_core_N always covers every tile,
        # padding the last core when the split is not exact (272/44 is not an integer).
        N = 8704
        n_tiles = sweep.tiles(N)
        for cfg in sweep.build_configs(5120, N, 64, quick=False):
            cores = cfg["grid"][0] * cfg["grid"][1]
            self.assertEqual(cfg["per_core_N"], math.ceil(n_tiles / cores))
            self.assertGreaterEqual(cfg["per_core_N"] * cores, n_tiles)

    def test_model_grid_is_included_even_when_not_exact(self):
        # The model's REAL mlp_w1/w3 grid: num_cores=44 over hidden_tp=8704 (272
        # tiles). 272/44 is not an integer, but ceiling-based per_core_N (7) still
        # makes this a legal, included candidate - the exact scenario the old
        # exact-division rule wrongly excluded (see legal_grids' docstring).
        K, N, M = 5120, 8704, 64
        model_grid = (11, 4)  # 44 cores
        configs = sweep.build_configs(K, N, M, model_grid=model_grid, quick=True)
        matches = [cfg for cfg in configs if cfg["grid"] == model_grid]
        self.assertTrue(matches)
        self.assertEqual(matches[0]["per_core_N"], math.ceil(sweep.tiles(N) / 44))

    def test_grids_wider_than_n_tiles_are_never_built(self):
        # N with only 5 tiles: no grid with more than 5 cores should ever appear.
        K, N, M = 5120, 160, 64
        for cfg in sweep.build_configs(K, N, M, quick=False):
            self.assertLessEqual(cfg["grid"][0] * cfg["grid"][1], sweep.tiles(N))

    def test_config_label_format(self):
        cfg = dict(grid=(11, 4), in0_block_w=8)
        self.assertEqual(sweep.config_label(cfg), "grid_11x4_blk8")


class SummarizeAndFormatTests(unittest.TestCase):
    def shapes(self):
        return {"mlp_w1": dict(name="mlp_w1", K=5120, N=8704, num_cores=44,
                               grid_w_pinned=True, silu=True)}

    def test_sorts_by_m64_min_us_ascending(self):
        runs = [
            dict(shape="mlp_w1", M=64, activation="L1", arm="grid_8x8_blk8",
                is_model_current=False, config={"grid": (8, 8)}, mean_us=500.0, min_us=480.0),
            dict(shape="mlp_w1", M=64, activation="L1", arm="model_current",
                is_model_current=True, config={"grid": (11, 4)}, mean_us=1080.0, min_us=1050.0),
            dict(shape="mlp_w1", M=32, activation="L1", arm="grid_8x8_blk8",
                is_model_current=False, config={"grid": (8, 8)}, mean_us=300.0, min_us=290.0),
            dict(shape="mlp_w1", M=32, activation="L1", arm="model_current",
                is_model_current=True, config={"grid": (11, 4)}, mean_us=550.0, min_us=540.0),
        ]
        summary = sweep.summarize(runs, self.shapes())
        rows = summary["mlp_w1"]
        self.assertEqual([row["arm"] for row in rows], ["grid_8x8_blk8", "model_current"])
        self.assertTrue(rows[1]["is_model_current"])

    def test_ratio_vs_dram_floor_only_on_successful_m64(self):
        runs = [
            dict(shape="mlp_w1", M=64, activation="L1", arm="model_current",
                is_model_current=True, config={}, mean_us=1080.0, min_us=1050.0),
            dict(shape="mlp_w1", M=32, activation="L1", arm="model_current",
                is_model_current=True, config={}, mean_us=550.0, min_us=540.0),
        ]
        summary = sweep.summarize(runs, self.shapes())
        row = summary["mlp_w1"][0]
        expected_floor = sweep.dram_floor_us(5120, 8704)
        self.assertAlmostEqual(row["dram_floor_us"], expected_floor)
        self.assertAlmostEqual(row["ratio_vs_dram_floor"], 1050.0 / expected_floor)

    def test_errored_candidate_sorts_last(self):
        runs = [
            dict(shape="mlp_w1", M=64, activation="L1", arm="bad_grid",
                is_model_current=False, config={}, error="RuntimeError: bad grid"),
            dict(shape="mlp_w1", M=64, activation="L1", arm="model_current",
                is_model_current=True, config={}, mean_us=1080.0, min_us=1050.0),
        ]
        summary = sweep.summarize(runs, self.shapes())
        self.assertEqual([row["arm"] for row in summary["mlp_w1"]], ["model_current", "bad_grid"])
        self.assertNotIn("ratio_vs_dram_floor", summary["mlp_w1"][1])

    def test_format_table_marks_error_and_current(self):
        runs = [
            dict(shape="mlp_w1", M=64, activation="L1", arm="model_current",
                is_model_current=True, config={}, mean_us=1080.0, min_us=1050.0),
            dict(shape="mlp_w1", M=32, activation="L1", arm="model_current",
                is_model_current=True, config={}, mean_us=550.0, min_us=540.0),
            dict(shape="mlp_w1", M=64, activation="L1", arm="bad_grid",
                is_model_current=False, config={}, error="boom"),
        ]
        summary = sweep.summarize(runs, self.shapes())
        table = sweep.format_table(summary, self.shapes())
        self.assertIn("mlp_w1", table)
        self.assertIn("yes", table)
        self.assertIn("ERR", table)
        self.assertIn("1050.0", table)


if __name__ == "__main__":
    unittest.main()
