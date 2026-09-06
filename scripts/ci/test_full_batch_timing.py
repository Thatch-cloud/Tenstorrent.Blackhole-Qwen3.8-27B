import unittest

from full_batch_timing import paired_control_flags, summarize


class TimingTests(unittest.TestCase):
    def test_dma_control_keeps_grouping_and_norm(self):
        flags = dict(compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                     hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
                     batch_conv=True, packed_checkpoints=True, ordered_cache=True, norm_batch=True, grouped_attention=True)
        self.assertEqual(paired_control_flags(**flags, attention_dma=True), dict(flags, attention_dma=False))

    def test_grouped_control_keeps_norm_and_cache(self):
        flags = dict(compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                     hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
                     batch_conv=True, packed_checkpoints=True, ordered_cache=True, norm_batch=True)
        self.assertEqual(paired_control_flags(**flags, grouped_attention=True), dict(flags, grouped_attention=False))

    def test_norm_control_keeps_ordered_cache_and_all_other_flags(self):
        flags = dict(compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                     hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
                     batch_conv=True, packed_checkpoints=True, ordered_cache=True)
        self.assertEqual(paired_control_flags(**flags, norm_batch=True), dict(flags, norm_batch=False))

    def test_batched_convolution_control_preserves_previous_device_loop(self):
        flags = dict(compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                     hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True)
        self.assertEqual(paired_control_flags(**flags, batch_conv=True), flags)
        self.assertEqual(paired_control_flags(**flags, batch_conv=False),
                         dict(flags, device_loop_gdn=False, compact_prologue=False))
        self.assertEqual(paired_control_flags(**flags, batch_conv=True, packed_checkpoints=True),
                         dict(flags, batch_conv=True))

    def test_paired_means_not_mixed_sample_ratio(self):
        samples = [dict(arm=arm, milliseconds=value) for arm, value in
                   zip(("serial", "batch", "batch", "serial"), (10, 3, 5, 14))]
        self.assertEqual(summarize(samples), dict(serial_ms=12, batch_ms=4, speedup=3))

    def test_ordered_cache_control_preserves_all_gdn_flags(self):
        flags = dict(compact_gdn=True, reuse_gdn_input=True, skip_row_clones=True,
                     hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
                     batch_conv=True, packed_checkpoints=True)
        self.assertEqual(paired_control_flags(**flags, ordered_cache=True), flags)

    def test_rejects_incomplete_or_reordered_blocks(self):
        for arms in (("serial",), ("serial", "serial", "batch", "batch")):
            with self.assertRaises(ValueError):
                summarize([dict(arm=arm, milliseconds=1) for arm in arms])

    def test_rejects_nonpositive_latency(self):
        samples = [dict(arm=arm, milliseconds=0) for arm in ("serial", "batch", "batch", "serial")]
        with self.assertRaises(ValueError):
            summarize(samples)


if __name__ == "__main__":
    unittest.main()
