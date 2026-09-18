import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from serving_startup import recipe_paths, start, stop
from test_serving_fast_policy import FastPolicyTests


class StartupTests(unittest.TestCase):
    def test_explicit_paths_match_loaded_runtime(self):
        config = FastPolicyTests().fixture()
        with TemporaryDirectory() as directory:
            config.additional_config['qwen_fast_runtime'] = {
                name: directory for name in ('directory', 'runtime_root', 'fixtures', 'target_snapshot')}
            with patch.dict(os.environ, {'TT_METAL_HOME': directory}):
                paths = recipe_paths(config)
                self.assertEqual(paths['runtime_root'], Path(directory).resolve())
            with patch.dict(os.environ, {'TT_METAL_HOME': str(Path(directory) / 'wrong')}):
                with self.assertRaises(ValueError):
                    recipe_paths(config)

    def test_no_implicit_runtime_or_fixture_defaults(self):
        config = FastPolicyTests().fixture()
        with self.assertRaises(ValueError):
            recipe_paths(config)
        config.additional_config['qwen_fast_runtime'] = {'directory': '.'}
        with self.assertRaises(ValueError):
            recipe_paths(config)

    def test_start_cannot_replace_an_existing_owner(self):
        worker = SimpleNamespace(_qwen_fast_resources=object())
        with self.assertRaises(ValueError):
            start(worker)

    def test_stop_releases_once_and_retains_owner_on_failure(self):
        resources = Mock()
        worker = SimpleNamespace(_qwen_fast_resources=resources, _qwen_fast_attachment=object())
        resources.close.side_effect = RuntimeError('cleanup failed')
        with self.assertRaises(RuntimeError):
            stop(worker)
        self.assertIs(worker._qwen_fast_resources, resources)
        resources.close.side_effect = None
        stop(worker)
        stop(worker)
        self.assertEqual(resources.close.call_count, 2)
        self.assertIsNone(worker._qwen_fast_resources)
