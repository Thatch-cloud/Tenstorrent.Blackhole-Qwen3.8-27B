import re
import unittest

from gdn_grouped_gather import ORIGINAL, grouped_copy, reader
from gdn_shared_qk_recurrence import CACHE


class GroupedGatherTests(unittest.TestCase):
    def test_only_copy_loop_changes(self):
        candidate = reader(CACHE)
        self.assertEqual(candidate.replace(grouped_copy(), ORIGINAL), CACHE)
        with self.assertRaises(ValueError):
            reader(candidate)
        with self.assertRaises(ValueError):
            reader(CACHE + CACHE)

    def test_every_word_has_the_same_bitwise_source_and_destination(self):
        source = grouped_copy()
        reads = dict((name, int(offset)) for name, offset in
            re.findall(r'const uint32_t (\w+) = source\[word \+ (\d+)\];', source))
        stores = [(int(offset), name) for offset, name in
            re.findall(r'target\[word \+ (\d+)\] = (\w+);', source)]
        self.assertEqual(len(reads), 8)
        self.assertEqual(len(stores), 8)
        visited = []
        for word in range(0, 16, 4):
            for offset, name in stores:
                self.assertEqual(reads[name], offset)
                visited.append(word + offset)
        self.assertEqual(sorted(visited), list(range(16)) + list(range(256, 272)))
        self.assertLess(source.rfind('= source['), source.index('target['))
