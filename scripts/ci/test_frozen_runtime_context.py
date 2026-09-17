import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from frozen_context_geometry import CONTEXTS, validate_position_limits
from frozen_recipe_context import REVISION
from frozen_runtime_context import FILES, adapt_runtime_sources


class RuntimeContextTests(unittest.TestCase):
    def test_context_cannot_silently_extend_target_or_draft_positions(self):
        draft = {'max_position_embeddings': 262144}
        target = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144))
        for context in CONTEXTS[:-1]:
            validate_position_limits(target, draft, context)
        with self.assertRaisesRegex(ValueError, '262400'):
            validate_position_limits(target, draft, 262144)
        with self.assertRaisesRegex(ValueError, 'draft positional limit'):
            validate_position_limits({'max_position_embeddings': 524288}, draft, 262144)
        with self.assertRaisesRegex(ValueError, 'target positional limit'):
            validate_position_limits({'text_config': {'max_position_embeddings': 65536}}, draft, 65536)
        with self.assertRaisesRegex(ValueError, 'Explicit target positional limit'):
            validate_position_limits({}, draft, 8192)

    def test_history_geometry_preserves_publication_and_admission(self):
        sources = {name: subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True) for name in FILES}
        adapted = adapt_runtime_sources(sources)
        original = sources['dspark_8k_scope.py']
        changed = adapted['dspark_8k_scope.py']
        start = '            FullHistoryKV.__init__'
        end = '    return EightKHistory'
        self.assertEqual(original.split(start)[1].split(end)[0],
            changed.split(start)[1].split(end)[0])
        start = '        evidence = stack.enter_context(admitted_request'
        end = '        stack.enter_context(patch.object(dspark_full_attention'
        self.assertEqual(original.split(start)[1].split(end)[0],
            changed.split(start)[1].split(end)[0])
        namespace = {}
        admission = subprocess.check_output(
            ['git', 'show', f'{REVISION}:scripts/ci/dspark_8k_admission.py'], text=True)
        exec(admission, namespace)
        namespace['validate_request'](8192, 256)
        for context in CONTEXTS:
            if context != 8192:
                with self.assertRaises(ValueError):
                    namespace['validate_request'](context, 256)

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
        self.assertNotIn('allocate_kv_cache((1032,', adapted['dspark-target-hardware.py'])
        self.assertNotIn('torch.arange(1024,', adapted['dspark-target-hardware.py'])
        self.assertIn("selected_geometry()['target_cache_blocks']", adapted['dspark-target-hardware.py'])
        self.assertIn("selected_geometry()['target_page_count']", adapted['dspark-target-hardware.py'])
        for name, source in adapted.items():
            if name.endswith('.py'):
                compile(source, name, 'exec')


if __name__ == '__main__':
    unittest.main()
