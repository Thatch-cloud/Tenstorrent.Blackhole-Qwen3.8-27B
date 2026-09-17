from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mlp_clock_capture import ClockCapture
from mlp_clock_samples import MAGIC


class ClockCaptureTests(unittest.TestCase):
    def setUp(self):
        self.mesh = object()
        self.counter = 0
        self.events = []
        self.operations = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram',
            ReplicateTensorToMesh=lambda mesh: mesh, from_torch=self.upload,
            get_device_tensors=lambda tensor: tensor.shards, to_torch=lambda shard: shard.words.clone(),
            synchronize_device=lambda mesh: self.events.append('sync'), copy_host_to_device_tensor=self.copy)
        self.patch = patch.dict('sys.modules', ttnn=self.operations)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def upload(self, value, **options):
        self.counter += 1024
        address = self.counter
        return SimpleNamespace(shape=value.shape, dtype='uint32', layout='row',
            device=lambda: self.mesh, memory_config=lambda: 'dram',
            shards=[SimpleNamespace(device=lambda chip=chip: SimpleNamespace(id=lambda: chip),
                buffer_address=lambda: address, words=value.clone()) for chip in (0, 2)])

    def copy(self, payload, tensor):
        self.events.append('poison')
        for source, target in zip(payload.shards, tensor.shards, strict=True):
            target.words.copy_(source.words)

    def execute(self, capture):
        self.events.append('execute')
        for tensor, role in zip(capture.buffers, ('input', 'weights'), strict=True):
            for shard in tensor.shards:
                for worker in range(len(shard.words)):
                    indices = (0, 4) if role == 'input' and worker == 1 else (0, 1, 2, 3)
                    for index in indices:
                        shard.words[worker, index * 6:index * 6 + 6] = torch.tensor(
                            [100 * index, 0, 100 * index + 20, 0, index, MAGIC ^ index])

    def test_poison_order_all_chips_and_stale_record_rejection(self):
        owned = []
        capture = ClockCapture(self.operations, self.mesh, owned)
        self.assertEqual(len(owned), 2)
        self.assertTrue(capture.reject_missing_execution())
        self.events.clear()
        capture.prepare()
        self.execute(capture)
        record = capture.collect('replay')
        self.assertEqual(self.events, ['sync', 'poison', 'poison', 'sync', 'execute', 'sync'])
        self.assertEqual(len(record['samples']), 20)
        self.assertEqual({entry['chip'] for entry in record['samples']}, {0, 1})
        self.assertTrue(capture.reject_missing_execution())
        self.assertEqual(len(capture.records), 1)
        with self.assertRaisesRegex(ValueError, 'Poison-before'):
            capture.collect('duplicate')

    def test_missing_chip_or_changed_bindings_fail_closed(self):
        capture = ClockCapture(self.operations, self.mesh, [])
        capture.prepare()
        with self.assertRaisesRegex(ValueError, 'not been collected'):
            capture.prepare()
        self.execute(capture)
        capture.buffers[0].shards[1].words.fill_(0xffffffff)
        with self.assertRaises(ValueError):
            capture.collect('partial')
        self.assertFalse(capture.pending)
        self.assertFalse(capture.records)
        capture.buffers[0].shards[0].buffer_address = lambda: 123456
        with self.assertRaisesRegex(ValueError, 'bindings changed'):
            capture.prepare()
