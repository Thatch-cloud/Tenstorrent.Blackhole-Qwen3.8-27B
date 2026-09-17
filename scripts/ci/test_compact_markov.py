from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import compact_markov as candidate


class CompactMarkovTests(unittest.TestCase):
    def test_simulator_only_before_enqueue(self):
        operations = Mock()
        with patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(ValueError, 'simulator-only'):
            candidate.execute(operations, None, None, None, None, None, [])
        operations.embedding.assert_not_called()

    def test_native_matmul_and_device_feedback(self):
        operations = Mock()
        tokens = [object() for _ in range(3)]
        diagnostics = [object() for _ in range(3)]
        operations.slice.side_effect = tokens
        anchor = object()
        observed = []
        with patch.dict('os.environ', {'QWEN_SIM_ONLY': '1', 'TT_METAL_SIMULATOR': 'test'}), \
                patch.object(candidate, 'validate', return_value=(3, 248320)), \
                patch.object(candidate, 'execute_local_winners'), \
                patch.object(candidate, 'reduce_winners', side_effect=diagnostics):
            records = candidate.execute(operations, SimpleNamespace(shape=[1, 2]), anchor,
                object(), object(), object(), [], on_step_enqueued=observed.append)
        self.assertEqual(observed, [0, 1, 2])
        for step, previous in enumerate([anchor, *tokens[:-1]]):
            self.assertIs(operations.embedding.call_args_list[step].args[0], previous)
            self.assertEqual(operations.slice.call_args_list[step].args,
                             (diagnostics[step], (0, 0, 0, 3), (1, 1, 1, 4)))
            self.assertIs(records[step]['diagnostic'], diagnostics[step])
        operations.argmax.assert_not_called()
        self.assertEqual(operations.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs[
            'compute_with_storage_grid_size'], (10, 10))
        self.assertFalse(operations.WormholeComputeKernelConfig.call_args.kwargs['math_approx_mode'])

    def test_invalid_step_rejects_whole_chain_and_safe_token_field(self):
        records = [[[7, 0, 0x3f800000, 7, 0, 0, 0, 0] for _ in range(2)] for _ in range(15)]
        tokens = [[7, 7] for _ in range(15)]
        self.assertTrue(candidate.validate_readback(records, tokens, 248320))
        records[4][1] = [0xffffffff, 1, 0, 0, 0, 0, 0, 0]
        tokens[4][1] = 0
        with self.assertRaisesRegex(ValueError, 'discard entire proposal'):
            candidate.validate_readback(records, tokens, 248320)
        source = Path(__file__).with_name('compact_score_reduce.cpp').read_text()
        self.assertIn('result[3] = invalid ? 0 : best_token;', source)
