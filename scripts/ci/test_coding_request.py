from unittest.mock import Mock
import unittest

from coding_request import make_prompt, TASK


class CodingRequestTests(unittest.TestCase):
    def test_single_task_without_padding_and_explicit_no_thinking(self):
        tokenizer = Mock()
        for encoded in ([1, 2, 3], {'input_ids': [1, 2, 3]}):
            tokenizer.apply_chat_template.return_value = encoded
            self.assertEqual(make_prompt(tokenizer), [1, 2, 3])
            arguments = tokenizer.apply_chat_template.call_args
            self.assertEqual(arguments.args[0][1]['content'], TASK)
            self.assertIs(arguments.kwargs['enable_thinking'], False)

    def test_invalid_token_shape_is_rejected(self):
        for encoded in ([], [True], [-1], [[1]], '1'):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = encoded
            with self.assertRaises(ValueError):
                make_prompt(tokenizer)
