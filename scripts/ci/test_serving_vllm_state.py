from types import SimpleNamespace
import unittest

import numpy as np

from serving_fast_request import CommittedOutput
from serving_vllm_state import apply_committed_output


class RunnerStateTests(unittest.TestCase):
    def fixture(self):
        state = SimpleNamespace(prompt_token_ids=[1, 2], output_token_ids=[10], num_computed_tokens=2)
        token_ids = np.full((1, 64), -1, dtype=np.int32)
        token_ids[0, :3] = [1, 2, 10]
        batch = SimpleNamespace(num_reqs=1, req_id_to_index={'request': 0},
            req_output_token_ids=[state.output_token_ids], token_ids_cpu=token_ids,
            num_tokens=np.array([3]), num_computed_tokens_cpu=np.array([2]))
        runner = SimpleNamespace(requests={'request': state}, input_batch=batch,
            model_config=SimpleNamespace(max_model_len=64, get_vocab_size=lambda: 100))
        return runner, state

    def test_variable_prefix_updates_both_views_once(self):
        for count in (1, 2, 8, 16):
            runner, state = self.fixture()
            tokens = tuple(range(11, 11 + count))
            output = CommittedOutput('request', tokens, 2 + count, False)
            apply_committed_output(runner, state, output)
            self.assertEqual(state.output_token_ids, [10, *tokens])
            self.assertEqual(runner.input_batch.token_ids_cpu[0, 3:3 + count].tolist(), list(tokens))
            self.assertTrue(np.all(runner.input_batch.token_ids_cpu[0, 3 + count:] == -1))
            self.assertEqual(state.num_computed_tokens, 2 + count)
            self.assertEqual(runner.input_batch.num_computed_tokens_cpu[0], 2 + count)
            with self.assertRaises(ValueError):
                apply_committed_output(runner, state, output)
            self.assertEqual(state.output_token_ids, [10, *tokens])

    def test_identity_reuse_alias_and_capacity_fail_before_mutation(self):
        for fault in ('identity', 'alias', 'capacity', 'frontier'):
            runner, state = self.fixture()
            if fault == 'identity':
                runner.requests['request'] = SimpleNamespace(**state.__dict__)
            elif fault == 'alias':
                runner.input_batch.req_output_token_ids[0] = list(state.output_token_ids)
            elif fault == 'capacity':
                runner.model_config.max_model_len = 3
            else:
                runner.input_batch.num_computed_tokens_cpu[0] = 1
            before = runner.input_batch.token_ids_cpu.copy()
            with self.assertRaises(ValueError):
                apply_committed_output(runner, state, CommittedOutput('request', (11, 12), 4, False))
            np.testing.assert_array_equal(runner.input_batch.token_ids_cpu, before)
            self.assertEqual(state.output_token_ids, [10])

    def test_cancelled_output_does_not_advance(self):
        runner, state = self.fixture()
        apply_committed_output(runner, state, CommittedOutput('request', (), 2, True, True))
        self.assertEqual(state.output_token_ids, [10])
        self.assertEqual(runner.input_batch.num_tokens[0], 3)

    def test_rejects_invalid_ids_and_frontier(self):
        for tokens, position in (((True,), 3), ((100,), 3), ((-1,), 3), ((11,), 18)):
            runner, state = self.fixture()
            with self.assertRaises(ValueError):
                apply_committed_output(runner, state, CommittedOutput('request', tokens, position, False))
            self.assertEqual(state.output_token_ids, [10])
