import unittest
from unittest.mock import Mock

import torch

from mtp_prefill import AlignedMTPStep, initialize_prompt


class MTPPrefillTests(unittest.TestCase):
    def test_shifted_prompt_pairs_exclude_padding_and_leave_next_seed_unconsumed(self):
        rows = torch.full((1, 1, 8, 5120), float('nan'), dtype=torch.bfloat16)
        rows[:, :, :3] = torch.tensor([100, 101, 102], dtype=torch.bfloat16).reshape(1, 1, 3, 1)
        calls, anchor = [], []
        def step(token, hidden, position, *, select):
            calls.append((token, hidden, position, select))
        report = initialize_prompt(step, [11, 12, 13], rows, anchor,
            stage_row=lambda value: float(value[0, 0, 0, 0]), copy_hidden=lambda source, target: target.append(source))
        self.assertEqual(calls, [(12, 100, 0, False), (13, 101, 1, False)])
        self.assertEqual(anchor, [102])
        self.assertEqual(report['next_mtp_position'], 2)
        self.assertEqual(report['excluded_padding'], 5)
        AlignedMTPStep(step)(14, 102, 3, select=True)
        self.assertEqual(calls[-1], (14, 102, 2, True))

    def test_single_token_prompt_has_no_fictitious_prefix_kv(self):
        step = Mock()
        report = initialize_prompt(step, [10], torch.ones(1, 1, 1, 5120, dtype=torch.bfloat16), [],
            stage_row=lambda value: value, copy_hidden=lambda source, target: target.append(source))
        step.assert_not_called()
        self.assertEqual(report['initialized_mtp_rows'], 0)

    def test_invalid_active_features_and_tokens_fail_before_step(self):
        for prompt, rows in (([], torch.ones(1, 1, 1, 5120, dtype=torch.bfloat16)),
                             ([True], torch.ones(1, 1, 1, 5120, dtype=torch.bfloat16)),
                             ([1], torch.full((1, 1, 1, 5120), float('nan'), dtype=torch.bfloat16))):
            step = Mock()
            with self.assertRaises(ValueError):
                initialize_prompt(step, prompt, rows, [], stage_row=Mock(), copy_hidden=Mock())
            step.assert_not_called()
        with self.assertRaises(ValueError):
            AlignedMTPStep(Mock())(1, object(), 0, select=True)
