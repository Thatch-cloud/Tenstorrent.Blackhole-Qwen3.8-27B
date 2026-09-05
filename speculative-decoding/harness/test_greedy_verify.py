import unittest

from greedy_verify import Decision, select_prefix


class GreedyVerifyTests(unittest.TestCase):
    def test_every_forced_rejection_prefix(self):
        proposals = tuple(range(1, 16))
        for accepted in range(16):
            target = list(proposals) + [31]
            target[accepted] = 30
            result = select_prefix(proposals, target, vocab_size=32)
            self.assertEqual(result, Decision(proposals[:accepted] + (30,), accepted,
                                               accepted + 1, 30, False))

    def test_target_only_and_all_accepted_bonus(self):
        self.assertEqual(select_prefix([], [7], vocab_size=10), Decision((7,), 0, 1, 7, False))
        self.assertEqual(select_prefix([1, 2], [1, 2, 3], vocab_size=10),
                         Decision((1, 2, 3), 2, 3, 3, False))

    def test_accepted_eos_discards_later_rows(self):
        self.assertEqual(select_prefix([1, 2, 3], [1, 2, 3, 4], vocab_size=10, eos_ids=[2]),
                         Decision((1, 2), 2, 3, None, True))

    def test_correction_eos_is_not_consumed(self):
        self.assertEqual(select_prefix([1, 2], [1, 9, 4], vocab_size=10, eos_ids=[9]),
                         Decision((1, 9), 1, 2, None, True))

    def test_unaccepted_draft_eos_does_not_stop(self):
        self.assertEqual(select_prefix([9], [2, 3], vocab_size=10, eos_ids=[9]),
                         Decision((2,), 0, 1, 2, False))

    def test_invalid_shapes_and_tokens_rejected(self):
        for proposals, predictions in (([], []), ([1], [1]), ([1], [1, 2, 3]),
                                       ([True], [1, 2]), ([1], [1, -1]),
                                       ([1], [1, 10]), ([1] * 16, [1] * 17)):
            with self.assertRaises(ValueError):
                select_prefix(proposals, predictions, vocab_size=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
