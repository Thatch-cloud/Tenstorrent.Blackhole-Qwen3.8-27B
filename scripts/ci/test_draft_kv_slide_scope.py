from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from draft_kv_slide_scope import scoped_publication
from test_draft_kv_slide_adapter import SlideAdapterTests


class PublicationScopeTests(unittest.TestCase):
    def exercise(self, fail=False):
        fixture = SlideAdapterTests()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'draft_kv_history.py'
            path.write_text(fixture.source)
            module = ModuleType('draft_kv_history')
            module.__file__ = str(path)
            module.SimpleNamespace = SimpleNamespace
            module.project_key_value = fixture.namespace['project_key_value']
            original = lambda *args, **kwargs: None
            module.DraftKVHistory = type('DraftKVHistory', (), dict(prepare=original))
            hardware = ModuleType('mlp_block_stream_runtime')
            hardware.require_hardware = lambda environment: None
            with patch.dict('sys.modules', draft_kv_history=module, mlp_block_stream_runtime=hardware), \
                    patch.dict('os.environ', QWEN_DRAFT_KV_SLIDE_EXPERIMENT='1'), \
                    patch('draft_kv_slide_scope.qualify', return_value={'passed': True}), \
                    patch('draft_kv_slide_scope.transport', side_effect=fixture.transport):
                try:
                    with scoped_publication(root, root) as audit:
                        module.DraftKVHistory.prepare(fixture.cache, object(), 16, position=4096)
                        if fail:
                            raise RuntimeError('request failed')
                except RuntimeError:
                    if not fail:
                        raise
                self.assertIs(module.DraftKVHistory.prepare, original)
                self.assertTrue(audit['restored'])
                self.assertEqual(audit['prepare_calls'], 1)
                self.assertEqual(audit['tensor_copies'], 10)
                self.assertIs(fixture.cache.active, fixture.active)

    def test_complete_scoped_publication(self):
        self.exercise()

    def test_exception_restores_native_method(self):
        self.exercise(fail=True)
