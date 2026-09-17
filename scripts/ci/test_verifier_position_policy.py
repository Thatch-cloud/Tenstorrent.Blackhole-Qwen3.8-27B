from types import SimpleNamespace
import unittest

from attention_batch import OrderedCacheWriter, SerialCacheWriter
from attention_replay import ReplayAttentionReader
from verifier_position_policy import requires_singletons


class PositionPolicyTests(unittest.TestCase):
    def fixture(self):
        reader = ReplayAttentionReader.__new__(ReplayAttentionReader)
        reader.audit = None
        return SimpleNamespace(replay_reader=reader, readers=[reader] * 16,
            writers=[OrderedCacheWriter.__new__(OrderedCacheWriter) for index in range(16)], ordered_cache=True)

    def test_default_preserves_uploads(self):
        self.assertTrue(requires_singletons(self.fixture(), '0'))

    def test_proven_unused_positions(self):
        self.assertFalse(requires_singletons(self.fixture(), '1'))

    def test_unknown_or_live_consumers_preserve_uploads(self):
        for change in ('audit', 'reader', 'writer', 'count', 'ordered', 'missing'):
            fixture = self.fixture()
            if change == 'audit':
                fixture.replay_reader.audit = object()
            elif change == 'reader':
                fixture.readers[0] = object()
            elif change == 'writer':
                fixture.writers[0] = SerialCacheWriter.__new__(SerialCacheWriter)
            elif change == 'count':
                fixture.writers.pop()
            elif change == 'ordered':
                fixture.ordered_cache = False
            else:
                fixture = SimpleNamespace()
            with self.subTest(change=change):
                self.assertTrue(requires_singletons(fixture, '1'))

    def test_invalid_switch(self):
        with self.assertRaises(ValueError):
            requires_singletons(self.fixture(), 'yes')
