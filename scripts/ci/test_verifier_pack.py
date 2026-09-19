from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from verifier_pack import GDN_LAYERS, allocate_slots, build_pack, participant


def engine(start, value, blocks=8):
    return SimpleNamespace(position=start, pages=torch.full((1, blocks), value, dtype=torch.int32))


def snapshots(tag, layers=GDN_LAYERS):
    return tuple(['%s%d' % (tag, index)] * 5 for index in range(layers))


class ParticipantTests(unittest.TestCase):
    def test_a_participant_carries_its_own_frontier_pages_and_state(self):
        entry = participant(engine(900, 3), 16, 12, snapshots('ck'), snapshots('slot'))
        self.assertEqual(entry['start'], 900)
        self.assertEqual(entry['rows'], 16)
        self.assertEqual(entry['prefix'], 12)
        self.assertEqual(len(entry['checkpoints']), GDN_LAYERS)
        self.assertEqual(len(entry['slots']), GDN_LAYERS)

    def test_missing_per_layer_state_or_a_prefix_past_the_rows_is_refused(self):
        for rows, prefix, checkpoints, slots in ((16, 17, snapshots('ck'), snapshots('s')),
                                                 (16, 12, snapshots('ck', 47), snapshots('s')),
                                                 (16, 12, snapshots('ck'), snapshots('s', 47)),
                                                 (0, 0, snapshots('ck'), snapshots('s'))):
            with self.assertRaises(ValueError):
                participant(engine(0, 1), rows, prefix, checkpoints, slots)

    def test_a_participant_needs_its_own_single_page_table(self):
        broken = SimpleNamespace(position=0, pages=torch.zeros(2, 8, dtype=torch.int32))
        with self.assertRaises(ValueError):
            participant(broken, 16, 12, snapshots('ck'), snapshots('s'))


class BuildPackTests(unittest.TestCase):
    def two(self):
        return [participant(engine(100, 1), 16, 12, snapshots('ckA'), snapshots('slotA')),
                participant(engine(5000, 2), 16, 9, snapshots('ckB'), snapshots('slotB'))]

    def test_two_users_fill_the_block(self):
        pack = build_pack(self.two())
        self.assertEqual([user['rows'] for user in pack], [16, 16])
        self.assertEqual([user['start'] for user in pack], [100, 5000])
        self.assertEqual([user['prefix'] for user in pack], [12, 9])

    def test_rows_must_fill_exactly_one_legal_width(self):
        users = self.two()
        users[1]['rows'] = 8
        with self.assertRaises(ValueError):
            build_pack(users)
        with self.assertRaises(ValueError):
            build_pack([])

    def test_a_shared_page_table_is_refused(self):
        """Two users reading one table would be two users reading one KV cache."""
        shared = engine(100, 1)
        users = [participant(shared, 16, 12, snapshots('ckA'), snapshots('slotA')),
                 participant(shared, 16, 9, snapshots('ckB'), snapshots('slotB'))]
        with self.assertRaises(ValueError):
            build_pack(users)

    def test_a_shared_carried_state_is_refused(self):
        """Sharing a carried state is the seam bug wearing a different hat."""
        shared = snapshots('shared')
        users = [participant(engine(100, 1), 16, 12, snapshots('ckA'), shared),
                 participant(engine(5000, 2), 16, 9, snapshots('ckB'), shared)]
        with self.assertRaises(ValueError):
            build_pack(users)

    def test_the_pack_is_what_model_batch_validates(self):
        from model_batch import validate_pack

        packed = validate_pack(build_pack(self.two()))
        self.assertEqual(packed['segments'], ((0, 16), (16, 32)))
        self.assertEqual(packed['prefixes'], (12, 9))
        self.assertTrue(bool((packed['pages'][:16] == 1).all()))
        self.assertTrue(bool((packed['pages'][16:] == 2).all()))
        self.assertTrue(torch.equal(packed['positions'][:3],
                                    torch.tensor([100, 101, 102], dtype=torch.int32)))
        self.assertTrue(torch.equal(packed['positions'][16:19],
                                    torch.tensor([5000, 5001, 5002], dtype=torch.int32)))


class AllocateSlotsTests(unittest.TestCase):
    def test_each_layer_is_allocated_and_seeded_from_the_live_state(self):
        helpers = [SimpleNamespace(allocate=Mock(side_effect=lambda index=index: ['s%d' % index] * 5),
                                   save=Mock()) for index in range(GDN_LAYERS)]
        slots = allocate_slots(helpers)
        self.assertEqual(len(slots), GDN_LAYERS)
        for helper, snapshot in zip(helpers, slots):
            helper.allocate.assert_called_once()
            helper.save.assert_called_once_with(snapshot)

    def test_a_short_helper_list_is_refused(self):
        with self.assertRaises(ValueError):
            allocate_slots([SimpleNamespace(allocate=Mock(), save=Mock())])


if __name__ == '__main__':
    unittest.main()
