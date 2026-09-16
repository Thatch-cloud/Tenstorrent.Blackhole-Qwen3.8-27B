import os
import subprocess
import unittest
from unittest.mock import patch

from frozen_context_geometry import CONTEXTS
from frozen_recipe_context import REVISION
from frozen_runtime_context import FILES, adapt_runtime_sources


class RuntimeContextTests(unittest.TestCase):
    def test_context_transport_does_not_overwrite_or_bypass_admission(self):
        sources = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True) for name in FILES}
        adapted = adapt_runtime_sources(sources)
        namespace = {}
        exec(adapted['dspark_context_selection.py'], namespace)
        for context in CONTEXTS:
            with patch.dict(os.environ, {'QWEN_DSPARK_REQUEST_CONTEXT': str(context)}):
                self.assertEqual(namespace['request_context'](), context)
        self.assertNotIn('export QWEN_DSPARK_REQUEST_CONTEXT=8192', adapted['dspark-hardware-suite.sh'])
        self.assertIn('-e "QWEN_DSPARK_REQUEST_CONTEXT=', adapted['run-dspark-hardware.sh'])
        self.assertEqual(sources['dspark_context_selection.py'].split('def validate_history_capacity', 1)[1],
            adapted['dspark_context_selection.py'].split('def validate_history_capacity', 1)[1])
        self.assertIn('max_batch_size=8, max_seq_len=selected_geometry()', adapted['dspark-target-hardware.py'])
        for name, source in adapted.items():
            if name.endswith('.py'):
                compile(source, name, 'exec')


if __name__ == '__main__':
    unittest.main()
