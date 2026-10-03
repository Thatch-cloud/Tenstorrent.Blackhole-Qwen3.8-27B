"""Keep text-only admission identical across control and chunked arms."""

import json
import os
from pathlib import Path
import subprocess
import unittest


class InterleaveArgumentsTests(unittest.TestCase):
    def arguments(self, ratio):
        path = Path(__file__).with_name("interleave-args.sh")
        environment = dict(os.environ, QWEN_INTERLEAVE_RATIO=str(ratio),
                           QWEN_PREFILL_CONTINUATION="0", TT_PREFILL_DECODE_INTERLEAVE="0")
        output = subprocess.check_output([
            "bash", "-eu", "-c",
            'source "$1"; printf "%s\\n" "$QWEN_PREFILL_CONTINUATION" "$TT_PREFILL_DECODE_INTERLEAVE" "${extra_args[@]}"',
            "test", str(path)], env=environment, text=True).splitlines()
        return output[:2], output[2:]

    def test_all_arms_reject_multimodal_inputs(self):
        for ratio in (0, 1, 2, 4):
            with self.subTest(ratio=ratio):
                flags, arguments = self.arguments(ratio)
                self.assertEqual(json.loads(arguments[arguments.index("--limit-mm-per-prompt") + 1]),
                                 {"image": 0, "video": 0})
                self.assertIn("--no-enable-mm-embeds", arguments)
                self.assertEqual(flags, ["1", "1"] if ratio else ["0", "0"])

    def test_chunk_budget_not_increased_to_fit_vision(self):
        for ratio in (1, 2, 4):
            _, arguments = self.arguments(ratio)
            self.assertIn("--enable-chunked-prefill", arguments)
            self.assertEqual(arguments[arguments.index("--max-num-batched-tokens") + 1], "2048")
        _, control = self.arguments(0)
        self.assertNotIn("--enable-chunked-prefill", control)
        self.assertNotIn("--max-num-batched-tokens", control)


if __name__ == "__main__":
    unittest.main()
