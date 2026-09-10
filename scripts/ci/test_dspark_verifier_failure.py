from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dspark_verifier_failure import compare_first_block


class VerifierFailureTests(unittest.TestCase):
    def test_compares_first_rows_and_restores_after_diagnostic(self):
        native = [[torch.ones(1, 1, 1, 4) for chip in range(2)] for layer in range(2)]
        batched = [[torch.ones(1, 1, 16, 4) for chip in range(2)] for layer in range(2)]
        batched[1][0][..., 0, 0] = 2
        observers = [SimpleNamespace(capture=nullcontext, outputs=lambda data=data: data, close=Mock())
            for data in (native, batched)]
        operations = SimpleNamespace(synchronize_device=Mock(), get_device_tensors=lambda value: value,
            to_torch=lambda value: value, deallocate=Mock())
        fixture = SimpleNamespace(run=Mock(return_value=object()), close=Mock())
        engine = SimpleNamespace(pending=SimpleNamespace(position=4096, tokens=tuple(range(16))),
            session=SimpleNamespace(committed_blocks=0), position=4096, operations=operations,
            model=object(), mesh=object(), sampler=None, restore_initial=Mock(),
            buckets={16: {'checkpoints': []}}, pending_key=16, fixture=Mock(return_value=fixture))
        with patch('dspark_verifier_failure.LayerOutputCapture', side_effect=observers), patch(
                'dspark_verifier_failure.stage_inputs') as stage, patch('dspark_verifier_failure.release_owned'):
            result = compare_first_block(engine, Mock(), lambda: ['initial'], ['initial'],
                batched[1][0][..., :1, :], tap=1, chip=0)
        self.assertEqual(engine.restore_initial.call_count, 3)
        stage.assert_called_once_with(fixture, tuple(range(16)), 4096)
        self.assertTrue(result['traced_equals_eager_batch'])
        self.assertFalse(result['traced_equals_native'])
        self.assertEqual([item['exact'] for item in result['layers']], [True, True, False, True])
        fixture.close.assert_called_once()
