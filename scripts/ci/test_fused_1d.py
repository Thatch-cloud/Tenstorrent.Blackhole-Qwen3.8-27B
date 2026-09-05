import unittest

from fused_1d import fused_compute, mapping


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
        self.assertLess(result.index("pack_block(start_dst_index, rounded_cb, 2)"), result.index("mul_binary_tile(0, 1, 0)"))

    def test_changed_source_fails_closed(self):
        with self.assertRaises(ValueError):
            fused_compute("no matching native kernel")

    def test_intermediate_variant_preserves_both_rounded_tiles(self):
        source = "                            if (last_out) {old pack" + "                            } else {\n                                tile_regs_commit();\n}"
        result = fused_compute(source, intermediates=True)
        self.assertNotIn("mul_binary_tile_init();", result)
        self.assertNotIn("mul_binary_tile(0, 1, 0);", result)
        self.assertIn("pack_tile(1, out_dfb_id);", result)
        self.assertIn("cb_reserve_back(out_dfb_id, 2);", result)
        self.assertIn("cb_push_back(out_dfb_id, 2);", result)


if __name__ == "__main__":
    unittest.main()
