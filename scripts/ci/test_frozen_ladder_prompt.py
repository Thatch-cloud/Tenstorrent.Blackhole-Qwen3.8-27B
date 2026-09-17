import unittest
from unittest.mock import Mock, patch

from coding_request import TASK
from frozen_context_geometry import CONTEXTS
from frozen_ladder_prompt import exact_frontier, make_context_prompt


class FrozenLadderPromptTests(unittest.TestCase):
    def test_exact_existing_prompts_are_not_changed(self):
        encode = Mock()
        best = (100, [1] * 4096)
        self.assertIs(exact_frontier(encode, best, 200, 4096), best)
        encode.assert_not_called()

    def test_nonmonotonic_token_boundary_is_searched_locally(self):
        encode = Mock(side_effect=lambda characters: [1] * (4096 if characters == 102 else 4095))
        characters, tokens = exact_frontier(encode, (100, [1] * 4095), 200, 4096)
        self.assertEqual(characters, 102)
        self.assertEqual(len(tokens), 4096)

    def test_short_prompt_is_never_returned_as_full_context(self):
        with self.assertRaisesRegex(ValueError, 'best binary-search length was 4095'):
            exact_frontier(lambda characters: [1] * 4095, (100, [1] * 4095), 200, 4096)

    def test_all_contexts_preserve_task_and_template(self):
        tokenizer = Mock()

        def encode(messages, **options):
            self.assertTrue(messages[1]['content'].endswith(TASK))
            self.assertFalse(options['enable_thinking'])
            self.assertTrue(options['add_generation_prompt'])
            return [1] * (len(messages[1]['content']) // 4)

        tokenizer.apply_chat_template.side_effect = encode
        for context in CONTEXTS:
            with self.subTest(context=context):
                tokens, metadata = make_context_prompt(tokenizer, context_tokens=context)
                self.assertEqual(len(tokens), context)
                self.assertEqual(metadata['actual_context'], context)
                self.assertEqual(len(metadata['sources']), 12)
        self.assertGreater(metadata['corpus_repetitions'], 1)

    def test_tampered_corpus_fails_before_tokenizing(self):
        with patch('frozen_ladder_prompt.Path.read_bytes', return_value=b'{}'):
            with self.assertRaisesRegex(ValueError, 'corpus changed'):
                make_context_prompt(Mock())

    def test_invalid_tokenizer_output_and_context_fail(self):
        for context in (True, 123, '4096'):
            with self.assertRaises(ValueError):
                make_context_prompt(Mock(), context_tokens=context)
        for tokens in ([], [True], [[1]], [-1]):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = tokens
            with self.assertRaises(ValueError):
                make_context_prompt(tokenizer)


if __name__ == '__main__':
    unittest.main()
