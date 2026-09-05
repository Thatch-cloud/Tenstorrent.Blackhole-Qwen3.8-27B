import unittest

from fused_1d import BF16_PRODUCT, fused_compute, mapping


class Fused1DTests(unittest.TestCase):
    def test_pair_mapping_matches_39_worker_control(self):
        workers = mapping()
        self.assertEqual(len(workers), 39)
        self.assertEqual(workers[-1], (5, 3, 266, 6))
        self.assertEqual([pair for _, _, begin, count in workers for pair in range(begin, begin + count)], list(range(272)))

    def test_only_final_pack_is_replaced(self):
        start = "                            if (last_out) {"
        end = "                            } else {\n                                tile_regs_commit();"
        source = "prefix" + start + "old pack" + end + "native partial loop\n}"
        result = fused_compute(source)
        self.assertIn("prefix", result)
        self.assertIn(end + "native partial loop", result)
        self.assertLess(result.index("native partial loop"), result.index("mul_binary_tile_init"))
        self.assertIn("apply_activation_from_pack<KernelActivation::SILU>(1)", result)
        self.assertLess(result.index("pack_block(start_dst_index, rounded_cb, 2)"), result.index(BF16_PRODUCT))
        self.assertIn("DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul", result)
        self.assertIn("(APPROX, ckernel::BinaryOp::MUL, 8, false)", result)

    def test_changed_source_fails_closed(self):
        with self.assertRaises(ValueError):
            fused_compute("no matching native kernel")

    def test_reviewed_grids_cover_each_pair_once(self):
        for pairs_per_worker, count in ((3, 91), (4, 68), (5, 55), (7, 39)):
            workers = mapping(pairs_per_worker)
            self.assertEqual(len(workers), count)
            self.assertEqual([pair for _, _, begin, valid in workers for pair in range(begin, begin + valid)], list(range(272)))
            self.assertTrue(all(core_x < 11 and core_y < 10 for core_x, core_y, _, _ in workers))
        for invalid in (0, 1, 2, 6, 8):
            with self.assertRaises(ValueError):
                mapping(invalid)

    def test_generated_epilogue_matches_worker_capacity(self):
        source = "                            if (last_out) {old pack" + "                            } else {\n                                tile_regs_commit();\n}"
        for pairs in (3, 4, 5, 7):
            result = fused_compute(source, pairs_per_worker=pairs)
            self.assertIn(f"cb_wait_front(rounded_cb, {2 * pairs});", result)
            self.assertIn(f"pair < {pairs};", result)

    def test_intermediate_variant_preserves_both_rounded_tiles(self):
        source = "                            if (last_out) {old pack" + "                            } else {\n                                tile_regs_commit();\n}"
        result = fused_compute(source, intermediates=True)
        self.assertNotIn("mul_binary_tile_init();", result)
        self.assertNotIn("mul_binary_tile(0, 1, 0);", result)
        self.assertNotIn(BF16_PRODUCT, result)
        self.assertIn("pack_tile(1, out_dfb_id);", result)
        self.assertIn("cb_reserve_back(out_dfb_id, 2);", result)
        self.assertIn("cb_push_back(out_dfb_id, 2);", result)


if __name__ == "__main__":
    unittest.main()
