import unittest

import torch

from target_packed_pages import packed_rows, segments, validate_users


def table(blocks, value):
    return torch.full((1, blocks), value, dtype=torch.int32)


class TargetPackedPagesTests(unittest.TestCase):
    def test_one_user_reproduces_what_model_batch_builds_today(self):
        """ModelBatch builds arange(start, start + rows) and pages.repeat(rows, 1)."""
        pages = torch.arange(40, dtype=torch.int32).reshape(1, 40)
        packed = packed_rows([dict(start=9000, rows=16, pages=pages)])
        self.assertTrue(torch.equal(packed['positions'],
                                    torch.arange(9000, 9016, dtype=torch.int32)))
        self.assertTrue(torch.equal(packed['pages'], pages.repeat(16, 1)))
        self.assertEqual(packed['segments'], ((0, 16),))

    def test_two_users_get_their_own_frontiers_and_their_own_blocks(self):
        left, right = table(40, 7), table(40, 11)
        packed = packed_rows([dict(start=100, rows=16, pages=left),
                              dict(start=5000, rows=16, pages=right)])
        self.assertEqual(packed['rows'], 32)
        self.assertEqual(packed['segments'], ((0, 16), (16, 32)))
        self.assertTrue(torch.equal(packed['positions'][:16], torch.arange(100, 116, dtype=torch.int32)))
        self.assertTrue(torch.equal(packed['positions'][16:], torch.arange(5000, 5016, dtype=torch.int32)))
        self.assertTrue(bool((packed['pages'][:16] == 7).all()), 'user 0 reads its own blocks')
        self.assertTrue(bool((packed['pages'][16:] == 11).all()), 'user 1 reads its own blocks')

    def test_uneven_splits_are_allowed_while_the_total_is_a_legal_width(self):
        packed = packed_rows([dict(start=0, rows=8, pages=table(4, 1)),
                              dict(start=64, rows=8, pages=table(4, 2))])
        self.assertEqual(packed['rows'], 16)
        self.assertEqual(packed['segments'], ((0, 8), (8, 16)))

    def test_illegal_totals_and_ragged_tables_are_refused(self):
        with self.assertRaises(ValueError):
            packed_rows([dict(start=0, rows=5, pages=table(4, 1))])
        with self.assertRaises(ValueError):
            packed_rows([dict(start=0, rows=16, pages=table(4, 1)),
                         dict(start=0, rows=16, pages=table(8, 2))])
        with self.assertRaises(ValueError):
            packed_rows([])
        with self.assertRaises(ValueError):
            packed_rows([dict(start=0, rows=16, pages=torch.zeros(2, 4, dtype=torch.int32))])
        with self.assertRaises(ValueError):
            validate_users([dict(start=-1, rows=16, pages=table(4, 1))])

    def test_segments_match_the_row_counts(self):
        spans, total = segments([dict(start=0, rows=4, pages=table(4, 1)),
                                 dict(start=0, rows=4, pages=table(4, 1))])
        self.assertEqual((spans, total), (((0, 4), (4, 8)), 8))


class ModelBatchPackTests(unittest.TestCase):
    """A pack is only accepted once every user carries its own GDN state.

    The row axis is TIME for the 48 GDN layers, so each user needs its own
    per-layer checkpoint and carried recurrent state. Without them user B would
    continue user A's recurrence - wrong for every row of B, and A left advanced
    by the whole block - so an incomplete pack is refused rather than run.
    """

    def user(self, start, value, prefix=12, layers=48):
        return dict(start=start, rows=16, pages=table(4, value), prefix=prefix,
                    checkpoints=['ck%d' % index for index in range(layers)],
                    slots=['slot%d' % index for index in range(layers)])

    def test_a_complete_pack_is_accepted_and_carries_the_per_user_state(self):
        from model_batch import validate_pack

        packed = validate_pack([self.user(0, 1, prefix=12), self.user(500, 2, prefix=9)])
        self.assertEqual(packed['rows'], 32)
        self.assertEqual(packed['segments'], ((0, 16), (16, 32)))
        self.assertEqual(packed['prefixes'], (12, 9))
        self.assertEqual(len(packed['checkpoints']), 2)
        self.assertEqual(len(packed['slots'][0]), 48)
        self.assertTrue(bool((packed['pages'][:16] == 1).all()))
        self.assertTrue(bool((packed['pages'][16:] == 2).all()))

    def test_a_pack_without_per_layer_gdn_state_is_refused(self):
        from model_batch import validate_pack

        for broken in ([dict(start=0, rows=16, pages=table(4, 1)),
                        dict(start=0, rows=16, pages=table(4, 2))],
                       [self.user(0, 1), self.user(0, 2, layers=47)],
                       [self.user(0, 1), dict(self.user(0, 2), prefix=17)]):
            with self.assertRaises(ValueError) as caught:
                validate_pack(broken)
            self.assertIn('GDN', str(caught.exception))

    def test_no_pack_stays_none(self):
        from model_batch import validate_pack

        self.assertIsNone(validate_pack(None))


if __name__ == '__main__':
    unittest.main()
