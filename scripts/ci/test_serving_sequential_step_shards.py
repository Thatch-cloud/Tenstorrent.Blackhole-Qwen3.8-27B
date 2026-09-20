"""The shard-equality diagnostic behind QWEN_FAST_SHARD_CHECK: off by default,
silent with one user, and exact about who wrote over whom when it fires."""

import importlib
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import serving_sequential_step as module


class FakeOperations:
    """DFlashDevice.operations stand-in: a 'tensor' is the list of its chip shards."""

    def __init__(self):
        self.reads = 0

    def get_device_tensors(self, value):
        self.reads += 1
        return value

    def to_torch(self, shard):
        return shard


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, template, *values):
        self.lines.append(template.format(*values))


def replicated(rows=4, columns=8):
    left = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns).to(torch.bfloat16)
    return [left, left.clone()]


def device(operations, *, layers=1, corrupt=None, proposal_calls=1):
    """`corrupt` is (buffer name, elements): change that many on chip 1 only."""
    tensors = {'history': replicated(), 'spare_history': replicated()}
    active = []
    for layer in range(layers):
        cache = {name: replicated() for name in ('k', 'v')}
        for name in ('k', 'v'):
            tensors['kv_history[%d].%s' % (layer, name)] = cache[name]
        active.append(cache)
    if corrupt is not None:
        name, count = corrupt
        tensors[name][1][0, :count] += 2.0
    return SimpleNamespace(operations=operations, history=tensors['history'],
                           spare_history=tensors['spare_history'],
                           kv_history=SimpleNamespace(active=active), proposal_calls=proposal_calls)


def closed_device(operations):
    return SimpleNamespace(operations=operations, history=None, spare_history=None,
                           kv_history=SimpleNamespace(active=[]), proposal_calls=0)


def entry(request_id, drafter, stepped):
    def step(name, *, cancelled):
        stepped.append(name)
        return SimpleNamespace(request_id=name, token_ids=[1])

    return dict(request_id=request_id, ticket=SimpleNamespace(request_id=request_id),
                request=SimpleNamespace(step=step, runtime=SimpleNamespace(drafter=drafter)))


class ShardCheckTests(unittest.TestCase):
    def setUp(self):
        self.logger = FakeLogger()
        for patcher in (patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=self.logger)}),
                        patch.object(module, 'SHARD_CHECK', True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_equal_shards_pass_and_the_log_proves_the_check_ran(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('B', device(operations, layers=2), stepped),
                   entry('A', device(operations, layers=2), stepped)]
        outputs = module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual(self.logger.lines, ['[PINDIAG] shards equal after step of B: 6 buffers',
                                             '[PINDIAG] shards equal after step of A: 6 buffers'])
        self.assertEqual(operations.reads, 12, 'history, spare and two K/V per layer, for the other user, twice')

    def test_a_diverged_buffer_names_actor_victim_size_and_order(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations, proposal_calls=2), stepped),
                   entry('B', device(operations, corrupt=('history', 3)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        message = str(raised.exception)
        for expected in ('after step of A (entry 0)', 'victim=B (entry 1)', 'buffer=history',
                         'differing=3 of 32', 'max_abs=2', "scheduler order=['A', 'B']",
                         'proposal_calls=[2, 1]'):
            self.assertIn(expected, message)
        self.assertEqual(stepped, ['A'], 'the victim is caught before it steps on the damage')
        self.assertEqual(self.logger.lines, [])

    def test_a_kv_layer_is_named_with_its_index_and_head(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, layers=3, corrupt=('kv_history[1].v', 5)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('buffer=kv_history[1].v differing=5 of 32', str(raised.exception))

    def test_the_switch_off_means_no_checks(self):
        with patch.object(module, 'SHARD_CHECK', False):
            stepped, operations = [], FakeOperations()
            entries = [entry('A', device(operations), stepped),
                       entry('B', device(operations, corrupt=('history', 3)), stepped)]
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(stepped, ['A', 'B'])
        self.assertEqual((operations.reads, self.logger.lines), (0, []))

    def test_one_entry_means_no_checks(self):
        stepped, operations = [], FakeOperations()
        module.sequential_packed_step([entry('A', device(operations, corrupt=('history', 3)), stepped)],
                                      cancelled=lambda: False)
        self.assertEqual(stepped, ['A'])
        self.assertEqual((operations.reads, self.logger.lines), (0, []))

    def test_a_released_device_has_nothing_to_check(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', closed_device(operations), stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(self.logger.lines[0], '[PINDIAG] shards equal after step of A: 0 buffers')

    def test_the_switch_is_read_once_at_import_and_defaults_off(self):
        with patch.dict(os.environ, {'QWEN_FAST_SHARD_CHECK': '1'}):
            self.assertTrue(importlib.reload(module).SHARD_CHECK)
        with patch.dict(os.environ):
            os.environ.pop('QWEN_FAST_SHARD_CHECK', None)
            self.assertFalse(importlib.reload(module).SHARD_CHECK)


if __name__ == '__main__':
    unittest.main()
