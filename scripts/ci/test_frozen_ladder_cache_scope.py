import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import frozen_ladder_cache_scope as scope
import ordered_cache


class CacheScopeTests(unittest.TestCase):
    def test_executed_scope_restores_and_rechecks_sources(self):
        kernels = dict(reader='reader', writer='writer', compute='compute')
        report = dict(generated_hashes={role: hashlib.sha256(source.encode()).hexdigest()
            for role, source in kernels.items()})
        environment = dict(QWEN_FROZEN_COMBINED_RUNTIME='1', QWEN_HARDWARE_TESTS='1',
            QWEN_CARDS_ALLOCATED='1', QWEN_DSPARK_REQUEST_CONTEXT='65536', TT_METAL_HOME='fixture')
        original = ordered_cache.validate_shapes
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'frozen-cache-evidence').mkdir()
            (root / 'frozen-cache-evidence/reports.json').write_text(json.dumps({'65536': 'a' * 64}))
            for raises in (False, True):
                with patch.dict(os.environ, environment, clear=True), \
                        patch.object(scope, 'qualify', return_value=report) as qualify, \
                        patch.object(scope, 'load_kernels', return_value=kernels):
                    def execute():
                        with scope.runtime_scope(root) as evidence:
                            self.assertEqual(ordered_cache.validate_shapes(
                                (1036, 2, 64, 256), (1, 16, 32, 256), (16,), (16, 1028)), 16)
                            if raises:
                                raise RuntimeError('request failed')
                        return evidence
                    if raises:
                        with self.assertRaisesRegex(RuntimeError, 'request failed'):
                            execute()
                    else:
                        evidence = execute()
                        self.assertTrue(evidence['restored'])
                        self.assertEqual(evidence['calls'], 1)
                        self.assertFalse(evidence['hardware_qualified'])
                    self.assertEqual(qualify.call_count, 2)
                    self.assertIs(ordered_cache.validate_shapes, original)
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(scope, 'qualify', return_value=report), \
                    patch.object(scope, 'load_kernels', return_value=dict(reader='wrong')):
                with self.assertRaisesRegex(ValueError, 'generated cache kernels'):
                    with scope.runtime_scope(root):
                        self.fail('Unqualified kernels entered scope')

    def test_serving_and_simulator_rejected(self):
        for environment in ({}, dict(QWEN_FROZEN_COMBINED_RUNTIME='1', QWEN_HARDWARE_TESTS='1',
                QWEN_CARDS_ALLOCATED='1', TT_METAL_SIMULATOR='sim')):
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(ValueError):
                with scope.runtime_scope('unused'):
                    self.fail('Non-hardware request entered scope')


if __name__ == '__main__':
    unittest.main()
