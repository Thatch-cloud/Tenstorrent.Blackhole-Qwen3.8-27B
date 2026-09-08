import unittest
from unittest.mock import Mock

from coding_context_request import make_context_prompt
from coding_request import TASK


class CodingContextRequestTests(unittest.TestCase):
    def test_full_template_and_task_survive_bounded_recorded_repository_excerpt(self):
        tokenizer = Mock()
        def encode(messages, **kwargs):
            self.assertEqual(messages[0]['content'], 'You are a careful coding assistant.')
            self.assertTrue(messages[1]['content'].endswith(TASK))
            self.assertIn('</repository_context>', messages[1]['content'])
            self.assertIs(kwargs['enable_thinking'], False)
            return [1, *(ord(character) for character in messages[1]['content']), 2]
        tokenizer.apply_chat_template.side_effect = encode
        first, metadata = make_context_prompt(tokenizer)
        second, repeated = make_context_prompt(tokenizer)
        self.assertEqual(first, second)
        self.assertEqual(metadata, repeated)
        self.assertEqual(metadata['actual_context'], len(first))
        self.assertEqual(len(first), 4096)
        self.assertEqual(metadata['requested_context'], 4096)
        self.assertEqual(len(metadata['sources']), 4)
        self.assertEqual(len(metadata['excerpt_sha256']), 64)
        self.assertEqual(len(metadata['prompt_sha256']), 64)

    def test_other_contexts_and_invalid_token_shapes_fail_closed(self):
        for context in (True, 170, 2048, 8192):
            with self.assertRaises(ValueError):
                make_context_prompt(Mock(), context_tokens=context)
        for encoded in ([], [True], [[1]], 'tokens', [-1]):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = encoded
            with self.assertRaises(ValueError):
                make_context_prompt(tokenizer)

    def test_too_short_repository_tokenization_cannot_claim_4k(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'cannot fill'):
            make_context_prompt(tokenizer)


if __name__ == '__main__':
    unittest.main()
