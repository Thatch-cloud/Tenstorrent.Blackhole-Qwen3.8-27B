import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import dspark_sim_build_cache as candidate


class SimulatorCacheTests(unittest.TestCase):
    def test_miss_then_hit_still_runs_builder_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / 'metal'
            cache = base / 'cache'
            output = base / 'report.json'
            for parent in ('build_Release/lib', 'build_Release/ttnn'):
                (root / parent).mkdir(parents=True)
            built = root / 'build_Release/ttnn/_ttnncpp.so'
            loaded = root / 'build_Release/lib/_ttnncpp.so'
            counts = dict(build=0, validate=0)

            def paths(value):
                return {'/opt/tt-metal': root, '/simulator-build-cache/ladder-v1': cache,
                    '/experiment/results/dspark-simulator-build-cache.json': output,
                    '/dev/tenstorrent': base / 'absent-device'}.get(str(value), Path(value))

            def compile(command, **keywords):
                counts['build'] += 1
                built.write_bytes(b'compiled-library')
                return subprocess.CompletedProcess(command, 0)

            def builder():
                candidate.subprocess.run(['ninja', '-C', str(root / 'build_Release'), '-j', '8', 'ttnncpp'])
                loaded.write_bytes(built.read_bytes())
                self.assertEqual(loaded.read_bytes(), b'compiled-library')
                counts['validate'] += 1

            environment = dict(QWEN_SIM_ONLY='1', TT_METAL_SIMULATOR='simulator.so', QWEN_SCORE_BITWISE='1')
            with patch.dict(os.environ, environment, clear=True), patch.object(candidate, 'Path', paths), \
                    patch.object(candidate, 'build_inputs', return_value={'fixture': 'pinned'}), \
                    patch.object(candidate.subprocess, 'run', compile), patch.object(candidate.ladder, 'main', builder):
                candidate.main()
                self.assertIn('"cache_hit": false', output.read_text())
                built.write_bytes(b'stock')
                candidate.main()
                self.assertIn('"cache_hit": true', output.read_text())
            self.assertEqual(counts, dict(build=1, validate=2))

    def test_hardware_environment_rejected(self):
        with patch.dict(os.environ, {'QWEN_CARDS_ALLOCATED': '1'}, clear=True):
            with self.assertRaises(ValueError):
                candidate.main()
