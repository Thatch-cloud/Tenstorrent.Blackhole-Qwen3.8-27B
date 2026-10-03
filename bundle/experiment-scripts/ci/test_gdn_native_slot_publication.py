import unittest
from unittest.mock import Mock, patch

import gdn_native_slot_publication as candidate


class NativeSlotPublicationTests(unittest.TestCase):
    def fixture(self):
        compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
        shapes = compact + [(16, 24, 128, 128)] + [(1, 16, 5120)] * 4
        shapes += [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4 + compact
        return [[Mock(shape=shape) for shape in shapes] for layer in range(48)]

    def test_zero_refresh_is_inside_each_invocation_before_publication(self):
        layers = self.fixture()
        events = []
        with patch.object(candidate, 'copy_active', side_effect=lambda source, target: events.append((source, target))), \
                patch.object(candidate, 'prepare_compact', return_value=lambda: events.append('publish')):
            operation = candidate.prepare(object(), layers, 0, experimental=True)
            self.assertEqual(events, [])
            for replay in range(2):
                operation()
                offset = replay * 49
                self.assertEqual(events[offset:offset + 48], [(layer[10:15], layer[:5]) for layer in layers])
                self.assertEqual(events[offset + 48], 'publish')

    def test_every_positive_prefix_avoids_entry_refresh(self):
        layers = self.fixture()
        with patch.object(candidate, 'copy_active') as refresh, patch.object(candidate, 'prepare_compact') as prepare:
            for prefix in range(1, 17):
                candidate.prepare(object(), layers, prefix, experimental=True)()
            refresh.assert_not_called()
            self.assertEqual(prepare.return_value.call_count, 16)

    def test_default_and_invalid_prefix_rejected(self):
        with patch.object(candidate, 'prepare_compact') as prepare:
            with self.assertRaises(ValueError):
                candidate.prepare(object(), self.fixture(), 0)
            with self.assertRaises(ValueError):
                candidate.prepare(object(), self.fixture(), 17, experimental=True)
            prepare.assert_not_called()
