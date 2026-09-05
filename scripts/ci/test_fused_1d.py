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
        source = "prefix" + start + "old pack" + end + "native partial loop"
        result = fused_compute(source)
        self.assertTrue(result.startswith("prefix"))
        self.assertTrue(result.endswith(end + "native partial loop"))
        self.assertIn("apply_activation_from_pack<KernelActivation::SILU>(1)", result)
        self.assertLess(result.index("pack_block(start_dst_index, rounded_cb, 2)"), result.index("mul_tiles(rounded_cb"))

    def test_changed_source_fails_closed(self):
        with self.assertRaises(ValueError):
            fused_compute("no matching native kernel")


if __name__ == "__main__":
    unittest.main()
