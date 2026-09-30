import unittest

import torch

from gdn_recurrence_clock import MAGIC, ZONES
from gdn_recurrence_clock_capture import RecurrenceClockCapture
import test_mlp_compute_clock_capture as fixtures


class RecurrenceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ComputeCaptureTests()
        self.fixture.setUp()
        self.owned = []
        self.capture = RecurrenceClockCapture(self.fixture.operations, self.fixture.mesh, self.owned)

    def execute(self):
        for shard in self.capture.buffer.shards:
            for processor in range(3):
                for index in range(len(ZONES)):
                    shard.words[processor, index * 6:index * 6 + 6] = torch.tensor([
                        100 * index, 0, 100 * index + 20, 0, index,
                        MAGIC ^ index ^ (processor << 16) ^ (self.capture.token << 8)])

    def test_poison_before_after_and_complete_two_chip_samples(self):
        self.assertEqual(self.owned, [self.capture.buffer])
        self.assertTrue(self.capture.reject_missing_execution())
        self.capture.prepare()
        self.execute()
        result = self.capture.collect('replay')
        self.assertEqual(len(result['samples']), 42)
        self.assertEqual({row['token'] for row in result['samples']}, {8})
        self.assertEqual({(row['chip'], row['processor']) for row in result['samples']},
            {(chip, processor) for chip in range(2) for processor in range(3)})
        self.assertTrue(self.capture.reject_missing_execution())
        self.assertEqual(len(self.capture.records), 1)

    def test_partial_sample_and_changed_identity_rejected(self):
        self.capture.prepare()
        with self.assertRaisesRegex(ValueError, 'must be collected'):
            self.capture.prepare()
        self.execute()
        self.capture.buffer.shards[1].words[2].fill_(0xffffffff)
        with self.assertRaisesRegex(ValueError, 'Missing, out-of-order'):
            self.capture.collect('partial')
        self.assertFalse(self.capture.records)
        self.assertFalse(self.capture.pending)
        with self.assertRaisesRegex(ValueError, 'Poison-before'):
            self.capture.collect('duplicate')
        self.capture.buffer.shards[0].buffer_address = lambda: 999999
        with self.assertRaisesRegex(ValueError, 'bindings changed'):
            self.capture.prepare()

    def test_invalid_token_and_changed_placement_rejected(self):
        for token in (True, -1, 16):
            with self.assertRaises(ValueError):
                RecurrenceClockCapture(self.fixture.operations, self.fixture.mesh, [], token=token)
        self.capture.prepare()
        self.execute()
        self.capture.buffer.memory_config = lambda: 'interleaved'
        with self.assertRaisesRegex(ValueError, 'logical core'):
            self.capture.collect('moved')


if __name__ == '__main__':
    unittest.main()
