from types import SimpleNamespace
import unittest

import torch

from mlp_compute_clock import MAGIC, ZONES
from mlp_compute_clock_capture import ComputeClockCapture
from mlp_compute_clock_projection import sample_memory


class ComputeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.mesh, self.counter, self.events = object(), 0, []
        self.operations = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row',
            CoreCoord=lambda *values: values, CoreRange=lambda *values: values,
            CoreRangeSet=tuple, ShardSpec=lambda grid, shape, orientation: (grid, tuple(shape), orientation),
            ShardOrientation=SimpleNamespace(ROW_MAJOR='rows'),
            TensorMemoryLayout=SimpleNamespace(HEIGHT_SHARDED='height'), BufferType=SimpleNamespace(L1='l1'),
            MemoryConfig=lambda *values: values, ReplicateTensorToMesh=lambda mesh: mesh,
            from_torch=self.upload, get_device_tensors=lambda tensor: tensor.shards,
            to_torch=lambda shard: shard.words.clone(),
            synchronize_device=lambda mesh: self.events.append('sync'), copy_host_to_device_tensor=self.copy)

    def upload(self, words, **options):
        self.counter += 1024
        address = self.counter
        return SimpleNamespace(shape=words.shape, dtype='uint32', layout='row', device=lambda: self.mesh,
            memory_config=lambda: options.get('memory_config'),
            shards=[SimpleNamespace(buffer_address=lambda: address, words=words.clone()) for chip in range(2)])

    def copy(self, payload, tensor):
        self.events.append('poison')
        for source, target in zip(payload.shards, tensor.shards, strict=True):
            target.words.copy_(source.words)

    def execute(self, capture):
        self.events.append('execute')
        for shard in capture.buffer.shards:
            for processor in range(3):
                for index in range(len(ZONES)):
                    shard.words[processor, index * 6:index * 6 + 6] = torch.tensor(
                        [100 * index, 0, 100 * index + 20, 0, index, MAGIC ^ index ^ processor])

    def test_l1_storage_and_fresh_complete_capture(self):
        owned = []
        capture = ComputeClockCapture(self.operations, self.mesh, owned)
        self.assertEqual(len(owned), 1)
        self.assertEqual(capture.buffer.memory_config(), sample_memory(self.operations))
        self.assertTrue(capture.reject_missing_execution())
        self.events.clear()
        capture.prepare()
        self.execute(capture)
        result = capture.collect('replay')
        self.assertEqual(self.events, ['sync', 'poison', 'sync', 'execute', 'sync'])
        self.assertEqual(len(result['samples']), 36)
        self.assertEqual({(sample['chip'], sample['processor']) for sample in result['samples']},
            {(chip, processor) for chip in range(2) for processor in range(3)})
        self.assertTrue(capture.reject_missing_execution())
        self.assertEqual(len(capture.records), 1)

    def test_partial_processor_write_or_rebinding_rejected(self):
        capture = ComputeClockCapture(self.operations, self.mesh, [])
        capture.prepare()
        with self.assertRaisesRegex(ValueError, 'must be collected'):
            capture.prepare()
        self.execute(capture)
        capture.buffer.shards[1].words[1].fill_(0xffffffff)
        with self.assertRaisesRegex(ValueError, 'Missing, malformed'):
            capture.collect('partial')
        self.assertFalse(capture.pending)
        self.assertFalse(capture.records)
        with self.assertRaisesRegex(ValueError, 'Poison-before'):
            capture.collect('duplicate')
        capture.buffer.shards[0].buffer_address = lambda: 999999
        with self.assertRaisesRegex(ValueError, 'bindings changed'):
            capture.prepare()

    def test_wrong_core_placement_rejected(self):
        capture = ComputeClockCapture(self.operations, self.mesh, [])
        capture.buffer.memory_config = lambda: 'interleaved'
        with self.assertRaisesRegex(ValueError, 'logical core'):
            capture.prepare()
