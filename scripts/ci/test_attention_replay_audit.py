from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from attention_replay_audit import AttentionMismatch, AttentionReplayAudit


class AttentionAuditTests(unittest.TestCase):
    def fixture(self, **kwargs):
        operations = SimpleNamespace(DRAM_MEMORY_CONFIG='dram',
            clone=lambda value, **kwargs: value.clone(), get_device_tensors=lambda value: [value, value],
            to_torch=lambda value: value, deallocate=Mock(),
            slice=lambda value, start, end, **kwargs: value[tuple(slice(first, last) for first, last in zip(start, end))])
        query = torch.ones(1, 8, 12, 256, dtype=torch.bfloat16)
        oracle = Mock(side_effect=lambda query, *args, **kwargs: query.clone())
        audit = AttentionReplayAudit(operations, oracle, **kwargs)
        for _ in range(16):
            audit.capture(query, query, query, query, scale=0.0625, program_config='native')
        return audit, query, oracle

    def test_all_layers_and_both_chips_are_checked(self):
        audit, query, oracle = self.fixture()
        audit.check(170, [1] * 8)
        self.assertEqual(oracle.call_count, 16)
        self.assertEqual(len(audit.owned), 48)
        self.assertTrue(all(value is not query for value in audit.owned))
        audit.records[-1]['candidate'][0, 7, 11, 255] = 3
        with self.assertRaises(AttentionMismatch) as caught:
            audit.check(170, [1] * 8)
        evidence = caught.exception.evidence
        self.assertEqual(evidence['attention_index'], 15)
        self.assertEqual(evidence['coordinates'], [[0, 7, 11, 255]])
        self.assertEqual(evidence['actual_values'], [3.0])
        self.assertEqual(evidence['expected_values'], [1.0])
        self.assertEqual(evidence['input_tokens'], [1] * 8)

    def test_second_chip_difference_is_not_hidden_by_first_replica(self):
        audit, _, _ = self.fixture()
        damaged = audit.records[0]['candidate'].clone()
        damaged[0, 0, 0, 0] = float('nan')
        actual = audit.records[0]['candidate']
        audit.operations.get_device_tensors = lambda value: [value, damaged if value is actual else value]
        with self.assertRaises(AttentionMismatch) as caught:
            audit.check(256, [1] * 8)
        self.assertEqual(caught.exception.evidence['chip'], 1)
        self.assertFalse(caught.exception.evidence['actual']['finite'])

    def test_failure_exports_bounded_real_inputs_for_simulation(self):
        with TemporaryDirectory() as directory:
            audit, _, _ = self.fixture(pages=torch.arange(4).reshape(1, 4), output_directory=directory)
            audit.records[0]['candidate'][0, 0, 0, 0] = 2
            cache = torch.ones(1024, 2, 64, 256, dtype=torch.bfloat16)
            audit.records[0].update(keys=cache, values=cache)
            with self.assertRaises(AttentionMismatch) as caught:
                audit.check(170, [1] * 8)
            saved = torch.load(Path(directory) / 'real-query.pt', weights_only=True)
            self.assertEqual(tuple(saved['keys'].shape), (4, 2, 64, 256))
            self.assertEqual(saved['position'], 170)
            self.assertEqual(len(caught.exception.evidence['fixture']['sha256']), 64)
            self.assertEqual(audit.operations.deallocate.call_count, 2)

    def test_capture_budget_and_cleanup_exclude_borrowed_cache(self):
        audit, query, _ = self.fixture()
        with self.assertRaises(ValueError):
            audit.capture(query, query, query, query, scale=0.0625)
        with patch('attention_replay_audit.release_owned') as release:
            audit.close()
            audit.close()
        release.assert_called_once()
        with self.assertRaises(ValueError):
            audit.check(170, [1] * 8)
