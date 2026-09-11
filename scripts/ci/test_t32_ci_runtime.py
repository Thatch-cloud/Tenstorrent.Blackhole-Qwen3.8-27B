from pathlib import Path
import tempfile
import unittest

from sim_memory_budget import require_clean
from t32_ci_runtime import snapshot


class CiRuntimeTests(unittest.TestCase):
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
