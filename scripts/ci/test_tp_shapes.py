"""tp_shapes: the two-card row is today's literals, the four-card row is what the model's config divides to, the
graft's own widths agree, and the width is chosen fail-closed."""

import unittest

import tp_shapes
from test_tp4_model_widths import tp_args


class GeometryTests(unittest.TestCase):
    def test_the_pair_row_is_the_literals_the_fast_path_carries_today(self):
        found = tp_shapes.geometry(2)
        self.assertEqual((found.residual, found.vocab, found.mlp), (2560, 124160, 8704))
        # attention_head_fold / extent replay / pooled replay: 12 rows per token, 2 KV heads per chip, group 6
        self.assertEqual((found.attn_heads, found.attn_kv_heads, found.attn_group, found.attn_fold_rows), (12, 2, 6, 12))
        self.assertEqual(found.attn_out, 3072)
        # gdn_user_batch.HEADS, gdn_multitoken_conv (8240 / 8256, 5120), gdn_commit_dma.cpp (384 / 160 pages)
        self.assertEqual((found.gdn_nk, found.gdn_nv), (8, 24))
        self.assertEqual((found.gdn_qkv, found.gdn_z, found.gdn_qkvzab, found.gdn_qkvzab_padded), (5120, 3072, 8240, 8256))
        self.assertEqual((found.gdn_key, found.gdn_value), (1024, 3072))
        self.assertEqual((found.gdn_state_pages, found.gdn_conv_pages), (384, 160))
        # pair_row_exact.QUERY_HEADS, KEY_HEADS = 16, 4; serving_buffer_pool QUERY_SHAPE (1, 1, 32, 2048)
        self.assertEqual((found.draft_heads, found.draft_kv_heads, found.draft_query), (16, 4, 2048))
        self.assertEqual((found.draft_embedding, found.draft_taps), (2560, 2560))

    def test_the_four_card_row_halves_every_width_and_keeps_the_group(self):
        found = tp_shapes.geometry(4)
        self.assertEqual((found.residual, found.vocab, found.mlp), (1280, 62080, 4352))
        self.assertEqual((found.attn_heads, found.attn_kv_heads, found.attn_group, found.attn_fold_rows), (6, 1, 6, 6))
        self.assertEqual(found.attn_out, 1536)
        self.assertEqual((found.gdn_nk, found.gdn_nv), (4, 12))
        self.assertEqual((found.gdn_qkv, found.gdn_z, found.gdn_qkvzab, found.gdn_qkvzab_padded), (2560, 1536, 4120, 4128))
        self.assertEqual((found.gdn_key, found.gdn_value), (512, 1536))
        self.assertEqual((found.gdn_state_pages, found.gdn_conv_pages), (192, 80))
        self.assertEqual((found.draft_heads, found.draft_kv_heads, found.draft_query), (8, 2, 1024))
        self.assertEqual((found.draft_embedding, found.draft_taps), (1280, 1280))

    def test_the_graft_computes_the_same_widths(self):
        for tp, batch in ((2, 4), (4, 8)):
            args, _ = tp_args(tp, batch)
            found = tp_shapes.geometry(tp)
            self.assertEqual(
                (args.n_local_heads, args.n_local_kv_heads, args.gdn_nk_tp, args.gdn_nv_tp, args.gdn_qkv_dim_tp,
                 args.gdn_z_dim_tp, args.gdn_qkvzab_dim_tp, args.gdn_value_dim_tp, args.gdn_key_dim_tp,
                 args.attn_out_dim_tp),
                (found.attn_heads, found.attn_kv_heads, found.gdn_nk, found.gdn_nv, found.gdn_qkv, found.gdn_z,
                 found.gdn_qkvzab, found.gdn_value, found.gdn_key, found.attn_out))

    def test_the_sampler_fits_only_at_four_cards(self):
        self.assertFalse(tp_shapes.sampler_fits(2))
        self.assertTrue(tp_shapes.sampler_fits(4))

    def test_the_packed_recurrence_reader_arguments(self):
        # gdn_seq_block.compile_args carries [4, 4, 24, 3, 160, 0, 32, 64, 96, 0] at the pair
        self.assertEqual(tp_shapes.k5_reader_arguments(2), [4, 4, 24, 3, 160, 0, 32, 64, 96, 0])
        self.assertEqual(tp_shapes.k5_reader_arguments(4), [4, 4, 12, 3, 80, 0, 16, 32, 48, 0])

    def test_a_width_the_table_does_not_hold_is_refused(self):
        for tp in (0, 1, 3, 8, '4', 4.0, None, True):
            with self.assertRaises(ValueError):
                tp_shapes.geometry(tp)


class SelectionTests(unittest.TestCase):
    def test_unset_is_the_pair_and_needs_the_pair_mesh(self):
        self.assertEqual(tp_shapes.select({}, 2, (1, 2)), 2)
        with self.assertRaises(ValueError):
            tp_shapes.select({}, 4, (1, 4))

    def test_four_needs_four_devices_on_a_one_by_four_mesh(self):
        environ = {'QWEN_FAST_TP': '4'}
        self.assertEqual(tp_shapes.select(environ, 4, (1, 4)), 4)
        for devices, shape in ((2, (1, 2)), (4, (2, 2)), (4, (4, 1)), (8, (1, 8)), (4, (1, 2))):
            with self.assertRaises(ValueError):
                tp_shapes.select(environ, devices, shape)

    def test_naming_two_explicitly_still_needs_the_pair(self):
        self.assertEqual(tp_shapes.select({'QWEN_FAST_TP': '2'}, 2, (1, 2)), 2)
        with self.assertRaises(ValueError):
            tp_shapes.select({'QWEN_FAST_TP': '2'}, 4, (1, 4))

    def test_garbage_is_refused_rather_than_read_as_the_pair(self):
        for value in ('', ' 4', '4 ', '04', '3', '1', '8', '-4', 'four', '4.0', '0x4'):
            with self.assertRaises(ValueError, msg=repr(value)):
                tp_shapes.requested_tp({'QWEN_FAST_TP': value})


if __name__ == '__main__':
    unittest.main()
