from unittest.mock import Mock
import os
from pathlib import Path
import subprocess
import sys
import unittest

from coding_request import make_prompt, TASK


class CodingRequestTests(unittest.TestCase):
    def test_long_context_engine_is_rejected_before_model_loading(self):
        environment = dict(os.environ, QWEN_CODING_REQUEST='1', QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('full-prefix.py')),
            '--request-pilot', '--attention-engine'], env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Short coding workload is not qualified', result.stderr)

    def test_wrong_ci_suite_fails_before_docker_or_build(self):
        environment = dict(os.environ, QWEN_CODING_REQUEST='1', QWEN_CARDS_ALLOCATED='1',
            QWEN_RUN_MODE='full-attention-engine-wide')
        result = subprocess.run(['bash', str(Path(__file__).with_name('run-baseline.sh'))],
            env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Short coding workload requires full-norm-engine', result.stderr)

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
