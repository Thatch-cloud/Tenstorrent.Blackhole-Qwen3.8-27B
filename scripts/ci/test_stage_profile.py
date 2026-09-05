"""The profiling adaptation changes only the scoped cache allocation dtype."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("stage_profile", Path(__file__).with_name("stage-profile.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

SOURCE = '''import os
def unrelated():
    return make(dtype=ttnn.bfloat16)
def test_attention_tp_paged():
    def mk_cache():
        return make(
            zeros(dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
        )
    assert accuracy > 0.99
'''


class StageTests(unittest.TestCase):
    def test_scoped_replacement(self):
        updated = stage.transform(SOURCE)
        replacement = 'dtype=ttnn.bfloat8_b if os.environ.get("QWEN_SDPA_BF8") == "1" else ttnn.bfloat16,'
        self.assertEqual(updated.replace(replacement, "dtype=ttnn.bfloat16,"), SOURCE)
        self.assertEqual(updated.count(replacement), 1)

    def test_reject_changed_anchor(self):
        with self.assertRaises(ValueError):
            stage.transform(SOURCE.replace("dtype=ttnn.bfloat16,", "dtype=ttnn.bfloat8_b,"))

    def test_reject_ambiguous_test(self):
        with self.assertRaises(ValueError):
            stage.transform(SOURCE + SOURCE)


if __name__ == "__main__":
    unittest.main()
