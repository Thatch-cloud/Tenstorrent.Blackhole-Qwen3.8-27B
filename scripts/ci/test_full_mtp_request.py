import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from full_mtp_request import prefill_with_hidden, validate_prompt_hidden


class FullMTPRequestTests(unittest.TestCase):
    def test_prefill_snapshot_preserves_seed_and_releases_only_owned_device_rows(self):
        hidden = torch.ones(1, 1, 32, 5120, dtype=torch.bfloat16)
        model = SimpleNamespace(layers=[SimpleNamespace(forward=Mock(return_value=hidden)) for _ in range(64)],
            norm=Mock(side_effect=lambda value, mode: value.clone()))
        operations = SimpleNamespace(clone=torch.clone, deallocate=Mock(), to_torch=lambda value: value,
            get_device_tensors=lambda value: [value, value.clone()])
        def prefill(prompt):
            self.assertEqual(prompt, [10, 11, 12])
            model.layers[63].forward()
            return 37
        module = SimpleNamespace(Mode=SimpleNamespace(PREFILL='prefill'))
        with patch.dict(sys.modules, {'models.tt_transformers.tt.common': module}), patch(
                'full_mtp_request.addresses', side_effect=lambda operations, value: (id(value), id(value))):
            seed, rows = prefill_with_hidden(operations, model, [10, 11, 12], prefill)
        self.assertEqual(seed, 37)
        self.assertTrue(torch.equal(rows, hidden))
        self.assertNotEqual(rows.data_ptr(), hidden.data_ptr())
        self.assertEqual(operations.deallocate.call_count, 2)
        self.assertTrue(all(call.args[0] is not hidden for call in operations.deallocate.call_args_list))
        model.norm.assert_called_once()

    def test_prompt_replica_validation_excludes_padding_and_owns_copy(self):
        first = torch.ones(1, 1, 32, 5120, dtype=torch.bfloat16)
        second = first.clone()
        first[:, :, 3:].fill_(float('nan'))
        result = validate_prompt_hidden([first, second], 3)
        self.assertNotEqual(result.data_ptr(), first.data_ptr())
        self.assertTrue(torch.equal(result[:, :, :3], second[:, :, :3]))
        second[:, :, 2, 0] = 7
        with self.assertRaises(ValueError):
            validate_prompt_hidden([first, second], 3)

    def test_missing_shard_incomplete_rows_and_invalid_geometry_fail(self):
        correct = torch.zeros(1, 1, 32, 5120, dtype=torch.bfloat16)
        for parts, length in (([correct], 3), ([correct, correct], 33), ([correct, correct], True),
                ([correct.float(), correct.float()], 3), ([correct[..., :2560], correct[..., :2560]], 3)):
            with self.assertRaises(ValueError):
                validate_prompt_hidden(parts, length)

    def test_mtp_mode_requires_bounded_explicit_hardware_configuration(self):
        for drafts in ('2', '7'):
            environment = dict(os.environ, QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1',
                QWEN_MTP_DRAFTS=drafts, QWEN_CODING_REQUEST='0')
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('full-prefix.py'))],
                env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('MTP requires the explicit short coding norm-engine', result.stderr)

    def test_mtp_rejects_conflicting_abba_before_docker(self):
        environment = dict(os.environ, QWEN_RUN_MODE='full-mtp-request', QWEN_CARDS_ALLOCATED='1',
            QWEN_LOOKUP_CAP_ABBA='1')
        result = subprocess.run(['bash', str(Path(__file__).with_name('run-baseline.sh'))],
            env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('docker:', result.stderr)
