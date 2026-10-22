"""The REAL PackedVerifierEngine.verify with the sampled readback (QWEN_FAST_SAMPLED, docs/tp4-sampled-verify.md), pure CPU, at the pair's geometry (two chips: the fake ttnn of
test_packed_verifier is the pair's; the readback hook reads its widths from tp_shapes and is the same code at four chips, which test_sampled_packed_path drives).

test_packed_verifier's four-user block (the fake ttnn, the fake ModelBatch, the real per-user replay readers) is built as ever and then switched to the sampled readback the way
its constructor would (`sampled`, `shard_argmax`), with the trace's outputs the fake mesh holds: the argmax ids the shard path read back and the eight top-64 tensors. verify() then
runs unchanged around the hook: the staging, the replay, the readback, the audit cadence, the metrics. Held: a sampled entry's rows are the keyed draws at its ticket's positions, a
greedy entry's the argmax rows, the entries' order is not the segments', the metrics say what the readback cost, and with the readback off the same round is the old one."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import batched_reads_tp  # noqa: E402
import sampled_accept as sa  # noqa: E402
import sampled_test_support as support  # noqa: E402
import sampled_verify as sv  # noqa: E402
import test_packed_verifier as pv  # noqa: E402


class SampledBlockRoundTests(pv.FourUserFixture):
    def setUp(self):
        super().setUp()
        sv.reset()
        self.addCleanup(sv.reset)
        patcher = mock.patch.dict(os.environ, support.clean_environment(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []
        logger = mock.patch.object(sv, 'log_line', side_effect=self.log.append)
        logger.start()
        self.addCleanup(logger.stop)
        self.mesh_ops = support.Ops()

    def verify(self, block, entries):
        return block.verify(entries)

    def switched(self, array):
        """The block as built, switched to the sampled readback with the trace outputs of `array`'s logits."""
        block = self.build()
        logits = support.logits_tensor(array, chips=2)           # (test_packed_verifier's fake ttnn is the pair's: two chips of 124,160 columns, four 32,768-column chunks; the hook is width-agnostic)
        slices = sv.device_slice(self.mesh_ops, logits, 64)
        block.output = (logits, None, None, None, *slices)
        block.sampled, block.shard_argmax, block.shard_audit = True, True, False
        host = [int(token) for token in torch.from_numpy(array).bfloat16().float().numpy().argmax(axis=1)]
        block.shard_predictions = mock.Mock(return_value=host)
        patcher = mock.patch.object(sv, 'read_tensors', lambda operations, mesh, tensors: batched_reads_tp.served(self.mesh_ops, tensors))
        patcher.start()
        self.addCleanup(patcher.stop)
        return block, host

    def rows(self, seed=5):
        rng = np.random.default_rng(seed)
        return (np.round(rng.normal(size=(64, support.VOCAB)) * 16) / 8).astype(np.float32)

    def test_the_real_verify_returns_the_keyed_draws_of_the_sampled_entries_and_the_argmax_of_the_rest(self):
        array = self.rows()
        block, host = self.switched(array)
        entries = self.four()                                # presented C, A, D, B: segments 2, 0, 3, 1
        specs = {0: sv.Sampling(sa.SamplingSpec(0.8, 0.95, 20), 17), 1: None, 2: sv.Sampling(sa.SamplingSpec(1.0, 1.0, 64), 2 ** 63 + 1), 3: sv.Sampling(sa.SamplingSpec(0.4, 0.9, 5), 9)}
        owners = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        for item in entries:
            item['request'].session.sampling = specs[owners[item['request_id']]]
        predictions, metrics = self.verify(block, entries)
        self.assertEqual(metrics['segments'], (2, 0, 3, 1))
        bf16 = torch.from_numpy(array).bfloat16().float().numpy()
        for item, segment, rows in zip(entries, metrics['segments'], predictions):
            sampling = item['request'].session.sampling
            if sampling is None:
                self.assertEqual(rows, host[16 * segment:16 * segment + 16], 'the greedy entry keeps the argmax rows')
            else:
                position = item['ticket'].position
                self.assertEqual(rows, [sa.sample_logits(bf16[16 * segment + row], sampling.spec, sampling.seed, position + 1 + row) for row in range(16)], item['request_id'])
        self.assertEqual(metrics['sampled']['entries'], 3)
        self.assertEqual(metrics['sampled']['rows'], 48)
        self.assertEqual(sorted(metrics['sampled']), ['draw_ms', 'entries', 'merge_ms', 'read_ms', 'rows'])
        self.assertEqual(block.sampled_rounds, 1)
        self.assertEqual((block.phase, block.pending_segments), ('verified', {0, 1, 2, 3}))
        self.assertEqual(block.describe()['sampled'], dict(k=64, rounds=1))

    def test_the_sampled_rows_feed_the_real_session_commit_path_unchanged(self):
        array = self.rows(6)
        block, host = self.switched(array)
        entries = self.four(order=(0, 1, 2, 3))
        for item in entries:
            item['request'].session.sampling = sv.Sampling(sa.SamplingSpec(), 3)
        predictions, metrics = self.verify(block, entries)
        for segment, prefix in zip(metrics['segments'], (16, 3, 0, 9)):
            block.commit_user(segment, prefix)
        self.assertEqual((block.phase, block.pending_segments), ('idle', set()))
        again, metrics = self.verify(block, self.four(order=(0, 1, 2, 3)))
        self.assertEqual(block.rounds, 2)

    def test_without_the_readback_the_round_is_the_old_one(self):
        block = self.build()
        self.assertFalse(block.sampled)
        entries = self.four()
        for item in entries:
            item['request'].session.sampling = sv.Sampling(sa.SamplingSpec(), 3)         # (a session carrying a Sampling is inert in a block that did not engage)
        predictions, metrics = self.verify(block, entries)
        self.assertEqual(predictions, [list(range(1032, 1048)), list(range(1000, 1016)), list(range(1048, 1064)), list(range(1016, 1032))])
        self.assertNotIn('sampled', metrics)
        self.assertNotIn('sampled', block.describe())

    def test_an_all_greedy_round_in_a_sampled_block_reads_nothing(self):
        array = self.rows(8)
        block, host = self.switched(array)
        entries = self.four()
        for item in entries:
            item['request'].session.sampling = None
        reads = (self.mesh_ops.single_reads, self.mesh_ops.composed_reads, self.mesh_ops.copies)
        predictions, metrics = self.verify(block, entries)
        self.assertEqual(metrics['sampled'], dict(entries=0))
        self.assertEqual((self.mesh_ops.single_reads, self.mesh_ops.composed_reads, self.mesh_ops.copies), reads)
        self.assertEqual(predictions, [host[16 * segment:16 * segment + 16] for segment in metrics['segments']])
        self.assertEqual(block.sampled_rounds if hasattr(block, 'sampled_rounds') else 0, 0)


if __name__ == '__main__':
    unittest.main()
