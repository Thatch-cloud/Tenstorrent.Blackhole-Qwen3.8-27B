import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import frozen_sim_build_cache as cache


class FrozenBuildCacheTests(unittest.TestCase):
    def test_scratch_requires_explicit_selection_and_pinned_sources(self):
        with patch.dict(os.environ, {'QWEN_FROZEN_TARGET_SCRATCH': 'yes'}):
            with self.assertRaisesRegex(ValueError, 'Explicit target scratch'):
                cache.main()
        with patch.dict(os.environ, {'QWEN_FROZEN_TARGET_SCRATCH': '1'}), \
                patch('sdpa_tree_scratch.audit', side_effect=ValueError('source mismatch')), \
                patch.object(cache.subprocess, 'run') as execute:
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                cache.main()
            execute.assert_not_called()

    def test_cold_build_reuse_missing_and_corrupt_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / 'metal'
            result = directory / 'results'
            result.mkdir()
            store = directory / 'cache'
            binaries = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
            for name in (cache.baseline.SOURCE,
                    'ttnn/cpp/ttnn/operations/transformer/sources.cmake',
                    'ttnn/cpp/ttnn/operations/transformer/op.cpp',
                    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h', *binaries):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'original')

            def location(value):
                text = str(value)
                if text == '/opt/tt-metal':
                    return root
                if text == '/frozen-simulator-cache/native-stats-v1':
                    return store
                if text.startswith('/experiment/results/'):
                    return result / text.rsplit('/', 1)[1]
                return Path(value)

            def process(command, *arguments, **keywords):
                if command[0] == 'ninja':
                    for name in binaries:
                        (root / name).write_bytes(b'built')
                return subprocess.CompletedProcess(command, 0)

            def baseline():
                cache.subprocess.run(['ninja', '-C', str(root / 'build_Release'), '-j', '8', 'ttnncpp'],
                    check=True, timeout=1800)
                cache.subprocess.run(['python3', '-c', 'import ttnn'], check=True)

            with patch.object(cache, 'Path', side_effect=location), \
                    patch.object(cache.baseline, 'implementation_sources', return_value=('op.cpp',)), \
                    patch.object(cache.baseline, 'main', side_effect=baseline), \
                    patch.object(cache.baseline, 'validate_manifest') as audit, \
                    patch.object(cache.subprocess, 'run', side_effect=process) as execute:
                with patch.dict(os.environ, {'QWEN_FROZEN_BUILD_ONLY': '0'}):
                    with self.assertRaisesRegex(ValueError, 'prepare it separately'):
                        cache.main()
                execute.assert_not_called()
                audit.assert_not_called()
                with patch.dict(os.environ, {'QWEN_FROZEN_BUILD_ONLY': '1'}):
                    cache.main()
                cold = json.loads((result / 'frozen-build-cache.json').read_text())
                self.assertFalse(cold['cache_hit'])
                audit.assert_called_once()
                for name in binaries:
                    (root / name).write_bytes(b'original')
                execute.reset_mock()
                with patch.dict(os.environ, {'QWEN_FROZEN_BUILD_ONLY': '0'}):
                    cache.main()
                warm = json.loads((result / 'frozen-build-cache.json').read_text())
                self.assertTrue(warm['cache_hit'])
                self.assertEqual(cold['cache_key'], warm['cache_key'])
                self.assertEqual(execute.call_count, 1)
                self.assertEqual(execute.call_args.args[0][0], 'python3')
                for name in binaries:
                    self.assertEqual((root / name).read_bytes(), b'built')
                    (root / name).write_bytes(b'original')
                (store / cold['cache_key'] / '_ttnncpp.so').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'binary hash changed'):
                    cache.main()


if __name__ == '__main__':
    unittest.main()
