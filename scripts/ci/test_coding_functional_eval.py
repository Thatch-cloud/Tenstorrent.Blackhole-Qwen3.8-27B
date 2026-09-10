import unittest

from coding_functional_eval import command, extract_source


class CodingFunctionalEvalTests(unittest.TestCase):
    def test_extract_does_not_execute_generated_code(self):
        source = 'def stable_unique(values):\n    raise RuntimeError("not executed")'
        self.assertEqual(extract_source('```python\n' + source + '\n```', 'stable_unique'), source)

    def test_incomplete_or_extra_top_level_code_rejected(self):
        for source in ('def stable_unique(', 'print("bad")', 'def other():\n    pass',
                'import os\ndef stable_unique(values):\n    pass'):
            with self.assertRaises((ValueError, SyntaxError)):
                extract_source('```python\n' + source + '\n```', 'stable_unique')

    def test_isolation_is_mandatory(self):
        arguments = command('qwen-coding-eval-test')
        for option in ('DynamicUser=yes', 'PrivateNetwork=yes', 'PrivateDevices=yes',
                'ProtectSystem=strict', 'ProtectHome=yes', 'NoNewPrivileges=yes',
                'RootDirectory=/opt/ttsim/coding-eval-root', 'RuntimeMaxSec=10', 'KillMode=control-group'):
            self.assertIn(option, arguments)
        self.assertNotIn('/mnt/d', ' '.join(arguments))
