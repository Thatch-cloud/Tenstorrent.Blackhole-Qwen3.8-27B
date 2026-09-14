from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dspark_splitk_correction_stride import AFTER, BEFORE, HEADER, correction_stride_scope, transform


class CorrectionStrideTests(unittest.TestCase):
    def test_unique_patch_and_failure_restoration(self):
        source = 'inline void calculate_fused_max_sub_exp_add_tile(int scale) {\n' + BEFORE + '\n}\n'
        self.assertIn(AFTER, transform(source))
        with self.assertRaises(ValueError):
            transform(source + source)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / HEADER
            path.parent.mkdir(parents=True)
            path.write_text(source)
            original = path.read_bytes()
            with patch.dict(os.environ, {'TT_METAL_HOME': root, 'QWEN_SIM_ONLY': '1',
                    'TT_METAL_SIMULATOR': 'test', 'QWEN_PRECISE_DRAFT_ACTIVE': '1'}), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    with correction_stride_scope():
                        self.assertIn(AFTER, path.read_text())
                        raise RuntimeError('injected')
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
