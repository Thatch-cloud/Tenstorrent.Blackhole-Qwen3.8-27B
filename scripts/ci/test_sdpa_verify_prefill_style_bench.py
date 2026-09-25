"""CPU-only unit tests for sdpa_verify_prefill_style_bench.py.

Two layers: pure arithmetic/mask/report tests with no ttnn import at all (capacity_for,
pages_for, decode_fake_head_bundles - which imports the real, pure attention_head_fold.py -
build_tree_mask, format_table), and a FakeTTNN-driven layer that exercises the device-touching
arm builders (prefill_style_arm, decode_fake_head_arm and their tensor-construction helpers)
without any real device, checking the exact shapes/dtypes/configs handed to
ttnn.transformer.scaled_dot_product_attention and
ttnn.transformer.paged_scaled_dot_product_attention_decode, and that every intermediate is
deallocated. time_op/safe_time's timing loop and run()/main() (real ttnn.open_device) are not
exercised here, matching test_matmul64_sweep.py's split between pure and device-dependent code.

Run with: py -3.10 -B -m unittest test_sdpa_verify_prefill_style_bench
"""

import unittest
from types import SimpleNamespace

import torch

import sdpa_verify_prefill_style_bench as bench


# ---------------------------------------------------------------------------------
# Pure parts: no ttnn import anywhere in this section.
# ---------------------------------------------------------------------------------

class CapacityAndPagesTests(unittest.TestCase):
    def test_capacity_for_default_context(self):
        self.assertEqual(bench.capacity_for(32768), 33024)

    def test_capacity_for_rounds_up_to_256(self):
        # 4096 + 256 = 4352, already a multiple of 256.
        self.assertEqual(bench.capacity_for(4096), 4352)
        # An odd context still lands on a 256-aligned capacity.
        self.assertEqual(bench.capacity_for(4097) % 256, 0)
        self.assertGreaterEqual(bench.capacity_for(4097), 4097 + 256)

    def test_capacity_for_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bench.capacity_for(0)
        with self.assertRaises(ValueError):
            bench.capacity_for(-1)

    def test_pages_for(self):
        self.assertEqual(bench.pages_for(33024, 64), 516)

    def test_pages_for_rejects_non_whole_pages(self):
        with self.assertRaises(ValueError):
            bench.pages_for(100, 64)


class DecodeFakeHeadBundlesTests(unittest.TestCase):
    """decode_fake_head_bundles calls the real, pinned attention_head_fold.parallel_groups -
    these tests pin the exact plan the task brief specified for the default 32768-token
    context, so a change to the replay geometry (or to attention_head_fold.py, which the team
    lead's message says is NOT pinned/hash-checked unlike attention_replay.py itself) is
    caught here rather than silently drifting the benchmark's shapes away from the real path."""

    def test_default_context_matches_task_brief(self):
        # "B=3 lanes x 48 fake heads + B=1 x 48" for one user's 16-row verify block at 32768.
        plan = bench.decode_fake_head_bundles()
        self.assertEqual(plan, ((3, 48), (1, 48)))

    def test_every_bundle_uses_the_fixed_fold_geometry(self):
        for batch, heads in bench.decode_fake_head_bundles():
            self.assertEqual(heads, bench.ROWS_PER_FOLD_GROUP * bench.NH_LOCAL)
            self.assertGreaterEqual(batch, 1)
            self.assertLessEqual(batch, 3)

    def test_bundle_row_count_covers_all_draft_rows(self):
        # Total (batch * rows-per-group) across bundles must equal the 16 draft rows this
        # replays, since parallel_groups tiles the block from row 0 with no gaps.
        plan = bench.decode_fake_head_bundles()
        total_groups = sum(batch for batch, _heads in plan)
        self.assertEqual(total_groups * bench.ROWS_PER_FOLD_GROUP, bench.DEFAULT_DRAFT_ROWS)

    def test_smaller_capacity_still_returns_a_valid_plan(self):
        plan = bench.decode_fake_head_bundles(capacity=bench.capacity_for(4096))
        self.assertTrue(plan)
        for batch, heads in plan:
            self.assertEqual(heads, 48)


class BuildTreeMaskTests(unittest.TestCase):
    def test_shape_and_dtype(self):
        mask = bench.build_tree_mask(4, 512, draft_rows=16, padded_rows=32)
        self.assertEqual(tuple(mask.shape), (4, 1, 32, 512))
        self.assertEqual(mask.dtype, torch.bfloat16)

    def test_prefix_fully_visible_for_live_rows(self):
        mask = bench.build_tree_mask(1, 512, draft_rows=16, padded_rows=32)
        prefix = 512 - 32
        self.assertTrue((mask[0, 0, :16, :prefix] == 0).all())

    def test_draft_block_is_causal(self):
        mask = bench.build_tree_mask(1, 512, draft_rows=16, padded_rows=32)
        prefix = 512 - 32
        block = mask[0, 0, :16, prefix:prefix + 16]
        for row in range(16):
            visible = block[row] == 0
            self.assertEqual(int(visible.sum()), row + 1)
            self.assertTrue(bool(visible[:row + 1].all()))
            if row + 1 < 16:
                self.assertTrue(bool(torch.isneginf(block[row, row + 1:]).all()))

    def test_padded_rows_see_exactly_one_key(self):
        mask = bench.build_tree_mask(1, 512, draft_rows=16, padded_rows=32)
        padded = mask[0, 0, 16:, :]
        visible_per_row = (padded == 0).sum(dim=-1)
        self.assertTrue((visible_per_row == 1).all())

    def test_only_zero_and_neg_inf_values(self):
        mask = bench.build_tree_mask(2, 512, draft_rows=16, padded_rows=32)
        self.assertTrue(((mask == 0) | torch.isneginf(mask)).all())

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            bench.build_tree_mask(0, 512)
        with self.assertRaises(ValueError):
            bench.build_tree_mask(1, 16, draft_rows=32, padded_rows=16)


class FormatTableTests(unittest.TestCase):
    def test_formats_success_and_error_rows(self):
        rows = [
            dict(label="prefill_style_masked_sdpa", users=4, calls=1, mean_us=123.4, min_us=110.0),
            dict(label="decode_fake_head", users=4, calls=8, error="boom"),
        ]
        table = bench.format_table(rows)
        self.assertIn("prefill_style_masked_sdpa", table)
        self.assertIn("123.4", table)
        self.assertIn("ERR", table)


# ---------------------------------------------------------------------------------
# FakeTTNN-driven: the device-touching arm builders, without any real device.
# ---------------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, shape, dtype, memory_config, name):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._memory_config = memory_config
        self.name = name

    def memory_config(self):
        return self._memory_config


class FakeTTNN:
    """The subset of the ttnn surface sdpa_verify_prefill_style_bench.py touches: dtype/layout
    markers, from_torch, the two SDPA ops (recording every call's operand shapes/dtypes and the
    program config used), deallocate/synchronize_device bookkeeping, and the compute-kernel/
    program-config constructors as simple namespaces (the benchmark only reads fields back off
    them, e.g. via device.compute_with_storage_grid_size(), never off the kernel config)."""

    bfloat16, bfloat8_b, int32 = "bf16", "bf8", "int32"
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = "tile", "row_major"
    DRAM_MEMORY_CONFIG = "dram"

    class MathFidelity:
        HiFi2 = "hifi2"

    def __init__(self):
        self.deallocated = []
        self.sdpa_calls = []
        self.decode_calls = []
        self._live = 0
        self.transformer = SimpleNamespace(
            scaled_dot_product_attention=self._sdpa,
            paged_scaled_dot_product_attention_decode=self._decode_sdpa,
        )

    def from_torch(self, host, *, device, dtype, layout, memory_config):
        self._live += 1
        return FakeTensor(tuple(host.shape), dtype, memory_config, "t%d" % self._live)

    def deallocate(self, tensor):
        if any(tensor is seen for seen in self.deallocated):
            raise AssertionError("Double free of %r" % (tensor,))
        self.deallocated.append(tensor)

    def synchronize_device(self, device):
        pass

    @staticmethod
    def WormholeComputeKernelConfig(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def SDPAProgramConfig(**kwargs):
        return SimpleNamespace(**kwargs)

    def _sdpa(self, query, key, value, *, attn_mask, is_causal, scale, program_config,
             compute_kernel_config, memory_config):
        self.sdpa_calls.append(dict(query=query.shape, key=key.shape, value=value.shape,
                                    mask=attn_mask.shape, is_causal=is_causal, scale=scale,
                                    program_config=program_config))
        self._live += 1
        return FakeTensor(query.shape, query.dtype, memory_config, "sdpa_out%d" % self._live)

    def _decode_sdpa(self, query, keys, values, *, page_table_tensor, is_causal, attn_mask, scale,
                     program_config, memory_config):
        self.decode_calls.append(dict(query=query.shape, keys=keys.shape, values=values.shape,
                                      page_table=page_table_tensor.shape, mask=attn_mask.shape,
                                      is_causal=is_causal, scale=scale, program_config=program_config))
        self._live += 1
        return FakeTensor(query.shape, query.dtype, memory_config, "decode_out%d" % self._live)


class FakeDevice:
    def __init__(self, x=8, y=8):
        self._grid = SimpleNamespace(x=x, y=y)

    def compute_with_storage_grid_size(self):
        return self._grid


class PrefillStyleArmTests(unittest.TestCase):
    def test_single_call_over_all_users_with_expected_shapes(self):
        ttnn = FakeTTNN()
        device = FakeDevice()
        result = bench.prefill_style_arm(ttnn, torch, device, users=4, kv_heads=2, capacity=33024,
                                         warmup=1, iters=2, k_chunk_size=256)
        self.assertEqual(result["calls"], 1)
        # time_op invokes run_once once per warmup iteration plus once per timed iteration.
        self.assertEqual(len(ttnn.sdpa_calls), 1 + 2)
        call = ttnn.sdpa_calls[0]
        self.assertEqual(call["query"], (4, 12, 32, 256))
        self.assertEqual(call["key"], (4, 2, 33024, 256))
        self.assertEqual(call["value"], (4, 2, 33024, 256))
        self.assertEqual(call["mask"], (4, 1, 32, 33024))
        self.assertFalse(call["is_causal"])
        self.assertAlmostEqual(call["scale"], 256 ** -0.5)

    def test_every_intermediate_is_deallocated(self):
        ttnn = FakeTTNN()
        device = FakeDevice()
        bench.prefill_style_arm(ttnn, torch, device, users=1, kv_heads=2, capacity=4352,
                                warmup=1, iters=1, k_chunk_size=256)
        # query, key, value, mask plus every warmup/timed op output.
        self.assertEqual(len(ttnn.deallocated), len(set(id(t) for t in ttnn.deallocated)))
        self.assertGreaterEqual(len(ttnn.deallocated), 4)

    def test_uses_the_corrected_two_kv_head_default(self):
        self.assertEqual(bench.DEFAULT_KV_HEADS, 2)


class DecodeFakeHeadArmTests(unittest.TestCase):
    def test_default_geometry_issues_two_calls_per_user(self):
        ttnn = FakeTTNN()
        device = FakeDevice()
        result = bench.decode_fake_head_arm(ttnn, torch, device, users=4, kv_heads=2,
                                            capacity=33024, page_size=64, warmup=1, iters=1)
        self.assertEqual(result["calls"], 8)  # 4 users x 2 bundles, exactly the task brief's count
        # warmup=1, iters=1: each bundle's op is invoked twice (once warm, once timed).
        self.assertEqual(len(ttnn.decode_calls), 8 * 2)

    def test_call_shapes_match_the_pinned_replay_geometry(self):
        ttnn = FakeTTNN()
        device = FakeDevice()
        bench.decode_fake_head_arm(ttnn, torch, device, users=1, kv_heads=2, capacity=33024,
                                   page_size=64, warmup=0, iters=1)
        first, second = ttnn.decode_calls
        self.assertEqual(first["query"], (1, 3, 48, 256))
        self.assertEqual(first["page_table"], (3, 516))
        self.assertEqual(first["mask"], (3, 1, 48, 33024))
        self.assertEqual(second["query"], (1, 1, 48, 256))
        self.assertEqual(second["page_table"], (1, 516))
        self.assertFalse(first["is_causal"])

    def test_kv_pool_shared_across_calls(self):
        ttnn = FakeTTNN()
        device = FakeDevice()
        bench.decode_fake_head_arm(ttnn, torch, device, users=2, kv_heads=2, capacity=33024,
                                   page_size=64, warmup=1, iters=1)
        keys_shapes = {call["keys"] for call in ttnn.decode_calls}
        self.assertEqual(keys_shapes, {(516, 2, 64, 256)})

    def test_every_intermediate_is_deallocated(self):
        ttnn = FakeTTNN()
        device = FakeDevice()
        bench.decode_fake_head_arm(ttnn, torch, device, users=1, kv_heads=2, capacity=33024,
                                   page_size=64, warmup=1, iters=1)
        self.assertEqual(len(ttnn.deallocated), len(set(id(t) for t in ttnn.deallocated)))


if __name__ == "__main__":
    unittest.main()
