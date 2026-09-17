from dataclasses import replace
import unittest

from prefill_prefix_lookup import PrefixIdentity, PrefixLookup


class PrefixLookupTests(unittest.TestCase):
    def fixture(self):
        identity = PrefixIdentity('a' * 64, 'b' * 64, 'c' * 64, 'coding-session', 0)
        tokens = list(range(4097))
        pages = list(range(65))
        lookup = PrefixLookup()
        lookup.publish(identity, tokens, 4096, pages, inactive_pages=[100])
        return lookup, identity, tokens, pages

    def test_changed_suffix_hits_but_changed_prefix_does_not(self):
        lookup, identity, tokens, pages = self.fixture()
        tokens[-1] = 99999
        self.assertEqual(lookup.match(identity, tokens, pages), 4096)
        tokens[0] = 99998
        self.assertEqual(lookup.match(identity, tokens, pages), 0)

    def test_identity_changes_invalidate(self):
        for change in (dict(target_sha256='d' * 64), dict(drafter_sha256='e' * 64),
                dict(recipe_sha256='f' * 64), dict(session='other'), dict(allocation_generation=1)):
            lookup, identity, tokens, pages = self.fixture()
            self.assertEqual(lookup.match(replace(identity, **change), tokens, pages), 0)
            self.assertIsNone(lookup.identity)

    def test_prefix_remapping_invalidates_but_suffix_remapping_is_allowed(self):
        lookup, identity, tokens, pages = self.fixture()
        pages[-1] = 99
        self.assertEqual(lookup.match(identity, tokens, pages), 4096)
        pages[0] = 98
        self.assertEqual(lookup.match(identity, tokens, pages), 0)
        self.assertIsNone(lookup.identity)

    def test_other_slot_and_internal_aliases_rejected(self):
        lookup, identity, tokens, pages = self.fixture()
        with self.assertRaises(ValueError):
            lookup.match(identity, tokens, pages, inactive_pages=[pages[0]])
        pages[-1] = pages[0]
        with self.assertRaises(ValueError):
            lookup.match(identity, tokens, pages)

    def test_failed_publication_clears_old_entry(self):
        lookup, identity, tokens, pages = self.fixture()
        with self.assertRaises(ValueError):
            lookup.publish(identity, tokens, 128, pages)
        self.assertEqual(lookup.match(identity, tokens, pages), 0)

    def test_identical_prefix_without_suffix_misses(self):
        lookup, identity, tokens, pages = self.fixture()
        self.assertEqual(lookup.match(identity, tokens[:4096], pages[:64]), 0)


if __name__ == '__main__':
    unittest.main()
