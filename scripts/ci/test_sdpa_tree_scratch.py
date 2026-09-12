import unittest

from sdpa_tree_scratch import scratch_slots


class TreeScratchTests(unittest.TestCase):
    def test_every_possible_binary_tree_send_round_fits(self):
        for allocated in range(1, 65):
            slots = scratch_slots(allocated)
            for active in range(1, allocated + 1):
                for worker in range(1, active):
                    send_round = (worker & -worker).bit_length() - 1
                    self.assertLess(send_round, slots)
            self.assertLessEqual(slots, allocated - 1)
        self.assertEqual(scratch_slots(55), 6)

    def test_invalid_geometry_rejected(self):
        for workers in (0, 65, True, 1.0):
            with self.assertRaises(ValueError):
                scratch_slots(workers)
