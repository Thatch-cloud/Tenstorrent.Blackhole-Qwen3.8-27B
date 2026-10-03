"""The long-8 dry run (tp4/seats8-262k, B6): eight disjoint 261,856-token real-text prompts from a corpus the size of the image's (vLLM 0.25.1's
source: about 27 M characters), with the suite's fake tokenizer.

L9b's top rung and the deepest churn users are 253,920 to 261,856 tokens, eight users at a time. real_text_prompts gives user i the window
[i * len // N, (i + 1) * len // N): eight windows of about 3.4 M characters. This holds that the build is feasible BEFORE the card window - eight
exact-length prompts, disjoint windows, bounded tokenizer work and padding - so a window is never lost to a prompt the corpus cannot supply."""

import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import real_text_prompts as rtp  # noqa: E402
import test_real_text_prompts as base  # noqa: E402

CORPUS_CHARACTERS = 27_000_000
TARGET = 261856


def corpus():
    """A unique-per-position corpus of 27 M characters (no repeated window): numbered python-looking definitions."""
    parts, size, index = [], 0, 0
    while size < CORPUS_CHARACTERS:
        text = 'def function_%d(argument):\n    return argument + %d\n\n' % (index, index)
        parts.append(text)
        size += len(text)
        index += 1
    return ''.join(parts)[:CORPUS_CHARACTERS]


class LongEightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = corpus()
        cls.built = rtp.build_prompts(8, None, tokenizer=base.FakeTokenizer(), corpus=cls.corpus, targets=[TARGET] * 8)

    def test_eight_prompts_of_exactly_261856_tokens(self):
        users = self.built['users']
        self.assertEqual(len(users), 8)
        for entry in users:
            with self.subTest(user=entry['user']):
                self.assertEqual((entry['prompt_tokens'], len(entry['tokens'])), (TARGET, TARGET))
                self.assertEqual(entry['chunks_of_2048'], 128)
                self.assertLessEqual(entry['padding_tokens'], rtp.MAX_PADDING)
                self.assertEqual(entry['framing'], 'repository')

    def test_the_windows_are_disjoint_contiguous_and_start_at_i_times_len_over_n(self):
        users = self.built['users']
        length = self.built['corpus']['cleaned_characters']
        self.assertEqual(length, CORPUS_CHARACTERS)
        for index, entry in enumerate(users):
            self.assertEqual(entry['window_start'], index * length // 8)
            self.assertEqual(entry['window_end'], (index + 1) * length // 8)
            self.assertLessEqual(entry['excerpt_end'], entry['window_end'])
            self.assertEqual(entry['excerpt_start'], entry['window_start'])
        for previous, following in zip(users, users[1:]):
            self.assertEqual(previous['window_end'], following['window_start'])
            self.assertLessEqual(previous['excerpt_end'], following['excerpt_start'], 'no two users read the same characters')

    def test_the_prompts_are_all_different_and_each_uses_a_small_share_of_the_corpus(self):
        users = self.built['users']
        self.assertEqual(len({entry['prompt_sha256'] for entry in users}), 8)
        self.assertEqual(len({entry['excerpt_sha256'] for entry in users}), 8)
        used = sum(entry['excerpt_characters'] for entry in users)
        self.assertLess(used, CORPUS_CHARACTERS * 0.5, 'eight windows of 261k tokens need under half of an image-sized corpus')
        for entry in users:
            window = entry['window_end'] - entry['window_start']
            self.assertLess(entry['excerpt_characters'], window, 'the window holds more than the prompt takes')

    def test_the_tokenizer_work_is_bounded(self):
        for entry in self.built['users']:
            self.assertLessEqual(entry['tokenizer_calls'], rtp.MAX_CALLS)
        self.assertLess(self.built['tokenizer_calls'], 8 * rtp.MAX_CALLS)

    def test_a_corpus_too_small_for_eight_windows_of_this_length_is_refused(self):
        with self.assertRaises(ValueError):
            rtp.build_prompts(8, None, tokenizer=base.FakeTokenizer(), corpus=self.corpus[:200_000], targets=[TARGET] * 8)

    def test_the_gate_ladders_rungs_are_buildable_lengths(self):
        import c2_serving_gate as driver

        for length in set(driver.LADDER8_262K) | set(driver.CHURN_LENGTHS_262K):
            self.assertGreaterEqual(length, rtp.COMPACT_MIN_TARGET)
        self.assertEqual(max(driver.LADDER8_262K), TARGET)


if __name__ == '__main__':
    unittest.main()
