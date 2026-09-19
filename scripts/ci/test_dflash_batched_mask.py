import unittest

import torch

from dflash_attention_mask import draft_attention_mask as shipped_mask
from dflash_batched_mask import batched_attention_mask, segments, validate_batched_mask


class BatchedMaskTests(unittest.TestCase):
    contexts = (1, 31, 32, 33, 512, 2047, 2048)

    def test_one_user_reproduces_the_shipped_t16_mask_exactly(self):
        """The gate on the whole generalisation: N=1 must change nothing.

        dflash_attention_mask builds a 32-row block, keeps 16 live rows and points
        the padded 16 at the anchor column. If the packed builder disagrees anywhere
        then it is a different attention, not a batched one.
        """
        for context in self.contexts:
            packed = batched_attention_mask([context], block_rows=16)
            expected = shipped_mask(context, block_rows=16)
            self.assertEqual(tuple(packed.shape), tuple(expected.shape), context)
            self.assertTrue(torch.equal(packed, expected),
                            'packed mask differs from the shipped T16 mask at context %d' % context)

    def test_two_users_are_block_diagonal_and_each_matches_the_single_user_mask(self):
        """Equivalence, not merely absence of leakage.

        Restricted to one user's rows and its own key segment, the packed mask must
        BE that user's single-user mask. That is what makes packing two users into
        the block arithmetically the same as running them separately.
        """
        for left in (33, 512, 2048):
            for right in (1, 700, 2048):
                packed = batched_attention_mask([left, right], block_rows=16)
                spans, key_rows = validate_batched_mask(packed, [left, right], block_rows=16)
                self.assertEqual(packed.shape[-1], key_rows)
                self.assertEqual(key_rows % 32, 0)
                for span, context in zip(spans, (left, right)):
                    alone = shipped_mask(context, block_rows=16)[0, 0, :16, :]
                    mine = packed[0, 0, span['rows'], span['keys']]
                    self.assertEqual(tuple(mine.shape), tuple(alone.shape), (left, right))
                    self.assertTrue(torch.equal(mine, alone),
                                    'user %d differs from its single-user mask' % span['user'])
                    outside = (packed[0, 0, span['rows']] == 0).clone()
                    outside[:, span['keys']] = False
                    self.assertFalse(bool(outside.any()), 'cross-user visibility')

    def test_key_axis_is_the_sum_of_aligned_spans(self):
        spans, key_rows = segments([2048, 2048], 16)
        self.assertEqual([span['span'] for span in spans], [2080, 2080])
        self.assertEqual(key_rows, 4160)
        self.assertEqual([span['offset'] for span in spans], [0, 2080])

    def test_padded_rows_keep_exactly_one_visible_key(self):
        packed = batched_attention_mask([512], block_rows=8)
        visible = (packed[0, 0] == 0)
        self.assertTrue(bool(visible[8:].sum(-1).eq(1).all()))

    def test_four_users_pack_into_the_block_at_eight_rows_each(self):
        packed = batched_attention_mask([300, 301, 302, 303], block_rows=8)
        spans, _ = validate_batched_mask(packed, [300, 301, 302, 303], block_rows=8)
        self.assertEqual(len(spans), 4)
        self.assertEqual(packed.shape[2], 32)

    def test_more_rows_than_the_block_is_refused(self):
        with self.assertRaises(ValueError):
            batched_attention_mask([10, 20, 30], block_rows=16)
        with self.assertRaises(ValueError):
            batched_attention_mask([], block_rows=16)

    def test_validator_catches_a_leak(self):
        spans, _ = segments([512, 512], 16)
        packed = batched_attention_mask([512, 512], block_rows=16)
        # a user-0 row made to see a key in user 1's segment
        packed[0, 0, 0, spans[1]['offset']] = 0.
        with self.assertRaises(ValueError):
            validate_batched_mask(packed, [512, 512], block_rows=16)


if __name__ == '__main__':
    unittest.main()


def reference_sdpa(query, key, value, mask):
    """GQA attention in float32, the shape the draft path uses: 16 q heads, 4 kv."""
    groups = query.shape[1] // key.shape[1]
    keys = key.repeat_interleave(groups, dim=1).float()
    values = value.repeat_interleave(groups, dim=1).float()
    scores = query.float() @ keys.transpose(-1, -2) / (query.shape[-1] ** 0.5)
    scores = scores + mask.float()
    return torch.softmax(scores, dim=-1) @ values


class PackedAttentionEquivalenceTests(unittest.TestCase):
    """Packing must be arithmetic-neutral, not merely non-leaking.

    If two users packed into one 32-row block produce different numbers than the
    same two users run separately, the batched verifier changes the committed
    tokens and the whole speculative contract is void. This is the gate that says
    it does not - on the host, in float32, with no device.
    """

    def setUp(self):
        torch.manual_seed(20260919)

    def pack(self, contexts, block_rows=16):
        spans, key_rows = segments(contexts, block_rows)
        query = torch.randn(1, 16, 32, 128, dtype=torch.float32)
        key = torch.zeros(1, 4, key_rows, 128)
        value = torch.zeros(1, 4, key_rows, 128)
        pieces = []
        for span, context in zip(spans, contexts):
            own_key = torch.randn(1, 4, span['span'], 128)
            own_value = torch.randn(1, 4, span['span'], 128)
            key[:, :, span['keys']] = own_key
            value[:, :, span['keys']] = own_value
            pieces.append((span, context, own_key, own_value))
        return spans, query, key, value, pieces

    def test_two_users_packed_equal_two_users_separate(self):
        for contexts in ([512, 512], [2048, 33], [1, 2048], [700, 1300]):
            spans, query, key, value, pieces = self.pack(contexts)
            packed = batched_attention_mask(contexts, block_rows=16)
            together = reference_sdpa(query, key, value, packed)
            for span, context, own_key, own_value in pieces:
                alone = reference_sdpa(query[:, :, span['rows']], own_key, own_value,
                                       shipped_mask(context, block_rows=16)[:, :, :16, :span['span']])
                mine = together[:, :, span['rows']]
                self.assertTrue(torch.allclose(mine, alone, rtol=1e-5, atol=1e-6),
                                'user %d output changed when packed with %s: max abs %g'
                                % (span['user'], contexts, float((mine - alone).abs().max())))

    def test_a_users_output_is_independent_of_the_other_users_keys(self):
        """Overwrite the neighbour's keys entirely; this user must not move."""
        contexts = [1024, 1024]
        spans, query, key, value, pieces = self.pack(contexts)
        packed = batched_attention_mask(contexts, block_rows=16)
        before = reference_sdpa(query, key, value, packed)[:, :, spans[0]['rows']]
        key[:, :, spans[1]['keys']] = torch.randn(1, 4, spans[1]['span'], 128) * 100
        value[:, :, spans[1]['keys']] = torch.randn(1, 4, spans[1]['span'], 128) * 100
        after = reference_sdpa(query, key, value, packed)[:, :, spans[0]['rows']]
        self.assertTrue(torch.equal(before, after), 'user 0 moved when user 1 changed')


class PackedRopeTests(unittest.TestCase):
    """Each user's slice of the packed tables must be its standalone table."""

    def test_query_and_key_slices_match_the_single_user_tables(self):
        from draft_head_preparation import rope_tables
        from dflash_batched_mask import packed_rope_tables

        users = [dict(position=5000, history_rows=2048), dict(position=777, history_rows=700)]
        contexts = [user['history_rows'] for user in users]
        spans, key_rows = segments(contexts, 16)
        tables = packed_rope_tables(users, block_rows=16)
        for user, span in zip(users, spans):
            for index, table in enumerate(rope_tables(user['position'], 16)):
                self.assertTrue(torch.equal(tables['q'][index][:, :, span['rows']], table),
                                'query rope for user %d' % span['user'])
            for index, table in enumerate(rope_tables(user['position'] - user['history_rows'], span['span'])):
                self.assertTrue(torch.equal(tables['k'][index][:, :, span['keys']], table),
                                'key rope for user %d' % span['user'])

    def test_one_user_matches_todays_device_tables(self):
        """N=1 must reproduce exactly what DFlashDevice.propose builds today."""
        from draft_head_preparation import rope_tables
        from dflash_batched_mask import packed_rope_tables

        position, history_rows = 4096, 2048
        spans, key_rows = segments([history_rows], 16)
        tables = packed_rope_tables([dict(position=position, history_rows=history_rows)], block_rows=16)
        for index, table in enumerate(rope_tables(position, 32)):
            self.assertTrue(torch.equal(tables['q'][index], table), 'query rope index %d' % index)
        for index, table in enumerate(rope_tables(position - history_rows, key_rows)):
            self.assertTrue(torch.equal(tables['k'][index], table), 'key rope index %d' % index)

    def test_a_user_whose_position_precedes_its_history_is_refused(self):
        from dflash_batched_mask import packed_rope_tables

        with self.assertRaises(ValueError):
            packed_rope_tables([dict(position=10, history_rows=2048)], block_rows=16)


class CachedHistoryPackingTests(unittest.TestCase):
    """The production path: cache_history=True, native_proposal_attention, T16.

    serving_request_factory builds the device with block_rows=16,
    cache_history=True, native_proposal_attention=True, so the branch that must
    pack is the one that does

        heads[name] = concat([cached_history[name], live[name]], dim=2)

    with cached_history shaped (1, 4, context, 128) and context in
    (256, 512, 1024, 2048). Packed, each user contributes its cached rows and its
    own 16 rows out of the shared live block.
    """

    def setUp(self):
        torch.manual_seed(20260920)

    def assemble(self, plan, cached, live):
        """Concatenate the packed key axis exactly as the plan describes."""
        pieces = []
        for piece in plan:
            if piece['kind'] == 'cached':
                pieces.append(cached[piece['user']])
            elif piece['kind'] == 'live':
                pieces.append(live[:, :, piece['source']])
            else:
                pieces.append(torch.zeros(1, 4, piece['rows'], 128))
            self.assertEqual(pieces[-1].shape[2], piece['rows'], piece)
        return torch.cat(pieces, dim=2)

    def test_packed_cached_history_matches_each_user_alone(self):
        from dflash_batched_mask import key_value_plan

        for contexts in ([2048, 2048], [512, 2048], [256, 1024], [1024, 1024]):
            plan, spans, key_rows = key_value_plan(contexts, block_rows=16)
            cached_key = [torch.randn(1, 4, context, 128) for context in contexts]
            cached_value = [torch.randn(1, 4, context, 128) for context in contexts]
            live_key, live_value = torch.randn(1, 4, 32, 128), torch.randn(1, 4, 32, 128)
            query = torch.randn(1, 16, 32, 128)

            packed = reference_sdpa(query, self.assemble(plan, cached_key, live_key),
                                    self.assemble(plan, cached_value, live_value),
                                    batched_attention_mask(contexts, block_rows=16))

            for span, context in zip(spans, contexts):
                rows = span['rows']
                pad = torch.zeros(1, 4, span['span'] - context - 16, 128)
                alone = reference_sdpa(
                    query[:, :, rows],
                    torch.cat([cached_key[span['user']], live_key[:, :, rows], pad], dim=2),
                    torch.cat([cached_value[span['user']], live_value[:, :, rows], pad], dim=2),
                    shipped_mask(context, block_rows=16)[:, :, :16, :])
                self.assertTrue(torch.allclose(packed[:, :, rows], alone, rtol=1e-5, atol=1e-6),
                                'user %d of %s moved when packed: max abs %g'
                                % (span['user'], contexts, float((packed[:, :, rows] - alone).abs().max())))

    def test_plan_covers_the_axis_and_splits_the_live_block_by_user(self):
        from dflash_batched_mask import key_value_plan

        plan, spans, key_rows = key_value_plan([2048, 2048], block_rows=16)
        self.assertEqual(key_rows, 4160)
        self.assertEqual([(piece['kind'], piece['rows']) for piece in plan],
                         [('cached', 2048), ('live', 16), ('pad', 16),
                          ('cached', 2048), ('live', 16), ('pad', 16)])
        live = [piece['source'] for piece in plan if piece['kind'] == 'live']
        self.assertEqual([(part.start, part.stop) for part in live], [(0, 16), (16, 32)])

    def test_live_key_rope_is_each_users_own_block_positions(self):
        from draft_head_preparation import rope_tables
        from dflash_batched_mask import live_key_rope

        users = [dict(position=9000, history_rows=2048), dict(position=1200, history_rows=1024)]
        packed = live_key_rope(users, block_rows=16)
        for index, user in enumerate(users):
            expected = rope_tables(user['position'], 16)
            for half, table in enumerate(expected):
                got = packed[half][:, :, index * 16:(index + 1) * 16]
                self.assertTrue(torch.equal(got, table), 'live key rope for user %d' % index)

    def test_live_key_rope_at_one_user_matches_todays_slice(self):
        from draft_head_preparation import rope_tables
        from dflash_batched_mask import live_key_rope, packed_rope_tables

        position, history_rows = 4096, 2048
        user = [dict(position=position, history_rows=history_rows)]
        spans, key_rows = segments([history_rows], 16)
        today = tuple(table[:, :, history_rows:key_rows]
                      for table in packed_rope_tables(user, block_rows=16)['k'])
        packed = live_key_rope(user, block_rows=16)
        for index, table in enumerate(today):
            self.assertEqual(tuple(table.shape), (1, 1, 32, 128))
            self.assertTrue(torch.equal(packed[index], table), 'live key rope index %d' % index)
