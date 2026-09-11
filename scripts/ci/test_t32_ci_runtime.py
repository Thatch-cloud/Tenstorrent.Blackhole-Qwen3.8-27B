from pathlib import Path
import tempfile
import unittest
import json
import os
from unittest.mock import patch

from sim_memory_budget import require_clean
from t32_ci_runtime import PINS, build_evidence, snapshot


class CiRuntimeTests(unittest.TestCase):
    def test_disposable_build_matches_sources_and_binaries(self):
        from t32_attention_fp32_patch import SOURCE_PATH, SOURCE_SHA256

        base = {name: value for name, value in PINS.items() if name.endswith('.so')}
        candidate = '12e5cc308f777ebf01bbfd5b48fee71335af916eead62763e0db8953c720972e'
        binaries = {name: 'a' * 64 for name in base}
        report = dict(stage='built', base_binaries=base, original_factory=SOURCE_SHA256,
                      candidate_factory=candidate, binaries=binaries, registration_patch='b' * 64)
        hashes = {str(Path('/runtime') / name): value for name, value in binaries.items()}
        hashes[str(Path('/runtime') / SOURCE_PATH)] = candidate
        hashes['/simulator-support/sdpa-graft-registration.patch'] = 'b' * 64
        with patch.dict(os.environ, {'QWEN_T32_FP32_BUILD': '1', 'QWEN_SIM_ONLY': '1',
                                    'QWEN_HARDWARE_TESTS': '0', 'QWEN_CARDS_ALLOCATED': '0'}), \
                patch('t32_ci_runtime.digest', side_effect=lambda path: hashes[str(path)]):
            with patch.object(Path, 'read_text', return_value=json.dumps(report)):
                self.assertEqual(build_evidence('/runtime'), report)
            for field in ('stage', 'candidate_factory', 'binaries', 'registration_patch'):
                with patch.object(Path, 'read_text', return_value=json.dumps(dict(report, **{field: None}))), \
                        self.assertRaises(ValueError):
                    build_evidence('/runtime')

    def test_enforced_quota_and_no_oom_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {'memory.max': str(64 * 1024**3), 'memory.swap.max': '0',
                'memory.current': '1024', 'memory.peak': '2048', 'memory.swap.current': '0',
                'cpu.max': '1600000 100000', 'memory.events': 'oom 0\noom_kill 0\nhigh 0\n',
                'boot': 'test-boot'}
            for name, value in values.items():
                (root / name).write_text(value)
            before = snapshot(root, root / 'boot')
            require_clean(before, snapshot(root, root / 'boot'))
            (root / 'memory.events').write_text('oom 1\noom_kill 0\nhigh 0\n')
            with self.assertRaises(ValueError):
                require_clean(before, snapshot(root, root / 'boot'))
            for value in ('max 100000', '800000 100000', '1600000 0'):
                (root / 'cpu.max').write_text(value)
                with self.assertRaises(ValueError):
                    snapshot(root, root / 'boot')
            (root / 'cpu.max').write_text(values['cpu.max'])
            (root / 'memory.max').write_text('max')
            with self.assertRaises(ValueError):
                snapshot(root, root / 'boot')


if __name__ == '__main__':
    unittest.main()
