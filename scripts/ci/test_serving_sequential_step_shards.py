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


class FakeShard:
    def __init__(self, data, address):
        self.data, self.address = data, address

    def buffer_address(self):
        return self.address


class FakeTensor:
    def __init__(self, shards, shape):
        self.shards, self.shape = shards, shape


class FakeOperations:
    """DFlashDevice.operations stand-in over FakeTensor chip shards."""

    def __init__(self):
        self.reads = 0

    def get_device_tensors(self, value):
        self.reads += 1
        return value.shards

    def to_torch(self, shard):
        return shard.data


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, template, *values):
        self.lines.append(template.format(*values))


def replicated(address, rows=4, columns=8):
    """One replicated buffer: equal bits on both chips, chip 1 one page up."""
    left = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns).to(torch.bfloat16)
    return FakeTensor([FakeShard(left, address), FakeShard(left.clone(), address + 0x100)], (1, 1, rows, columns))


def device(operations, *, kv_layers=1, corrupt=None, proposal_calls=1):
    """`corrupt` is (buffer name, elements): change that many on chip 1 only.
    Addresses follow construction order: layer-0 weights lowest, K/V highest."""
    named = [('weight.layers[0].attention.norm', 0x1000), ('weight.layers[0].attention.convolution', 0x2000),
             ('weight.layers[0].mlp.device_norm', 0x3000), ('weight.final_norm', 0x4000),
             ('weight.selector_projection', 0x5000), ('history', 0x6000), ('spare_history', 0x7000)]
    named.extend(('kv_history[%d].%s' % (layer, head), 0x8000 + layer * 0x1000 + index * 0x800)
                 for layer in range(kv_layers) for index, head in enumerate(('k', 'v')))
    tensors = {name: replicated(address) for name, address in named}
    if corrupt is not None:
        name, count = corrupt
        tensors[name].shards[1].data[0, :count] += 2.0
    attention = dict(norm=tensors['weight.layers[0].attention.norm'],
                     convolution=tensors['weight.layers[0].attention.convolution'])
    mlp = dict(device_norm=tensors['weight.layers[0].mlp.device_norm'])
    active = [{head: tensors['kv_history[%d].%s' % (layer, head)] for head in ('k', 'v')} for layer in range(kv_layers)]
    return SimpleNamespace(operations=operations, closed=False, layers=[(attention, mlp, None, None)],
                           selector_projection=tensors['weight.selector_projection'],
                           final_norm=tensors['weight.final_norm'],
                           history=tensors['history'], spare_history=tensors['spare_history'],
                           kv_history=SimpleNamespace(active=active), proposal_calls=proposal_calls)


def closed_device(operations):
    return SimpleNamespace(operations=operations, closed=True, layers=[], selector_projection=None,
                           final_norm=None, history=None, spare_history=None, kv_history=None, proposal_calls=0)


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
        module.RECORDED.clear()
        self.addCleanup(module.RECORDED.clear)

    def test_equal_shards_pass_and_the_log_proves_the_check_ran(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('B', device(operations, kv_layers=2), stepped),
                   entry('A', device(operations, kv_layers=2), stepped)]
        outputs = module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual([line for line in self.logger.lines if 'equal' in line],
                         ['[PINDIAG] shards equal after step of B: 11 buffers',
                          '[PINDIAG] shards equal after step of A: 11 buffers'],
                         'five weights, two histories and two K/V per layer, for the other user')
        # eleven buffers, each read once to record its address and once to compare, for each of two users
        self.assertEqual(operations.reads, 44)

    def test_addresses_are_logged_once_per_request_in_construction_order(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', device(operations), stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        module.sequential_packed_step(entries, cancelled=lambda: False)
        recorded = [line for line in self.logger.lines if 'addresses' in line]
        self.assertEqual(len(recorded), 2, 'once per request, not once per step')
        self.assertTrue(recorded[0].startswith('[PINDIAG] draft buffer addresses for B: {'))
        self.assertIn("'weight.layers[0].attention.norm': (4096, 4352)", recorded[0])
        self.assertIn("'kv_history[0].v': (34816, 35072)", recorded[0])
        self.assertEqual(list(module.RECORDED['B']), ['weight.selector_projection', 'weight.final_norm',
            'weight.layers[0].attention.norm', 'weight.layers[0].attention.convolution',
            'weight.layers[0].mlp.device_norm', 'history', 'spare_history', 'kv_history[0].k', 'kv_history[0].v'])

    def test_records_are_pruned_to_the_live_requests(self):
        stepped, operations = [], FakeOperations()
        first, second = entry('A', device(operations), stepped), entry('B', device(operations), stepped)
        module.sequential_packed_step([first, second], cancelled=lambda: False)
        self.assertEqual(set(module.RECORDED), {'A', 'B'})
        module.sequential_packed_step([second, entry('C', device(operations), stepped)], cancelled=lambda: False)
        self.assertEqual(set(module.RECORDED), {'B', 'C'})

    def test_a_diverged_history_names_actor_victim_size_address_and_order(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations, proposal_calls=2), stepped),
                   entry('B', device(operations, corrupt=('history', 3)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        message = str(raised.exception)
        for expected in ('replicated draft history differs', 'after step of A (entry 0)', 'victim=B (entry 1)',
                         'buffer=history', 'shape=(1, 1, 4, 8)', 'address=(24576, 24832)', 'first_seen_as=history',
                         'differing=3 of 32', 'max_abs=2', "scheduler order=['A', 'B']", 'proposal_calls=[2, 1]'):
            self.assertIn(expected, message)
        self.assertEqual(stepped, ['A'], 'the victim is caught before it steps on the damage')
        self.assertEqual([line for line in self.logger.lines if 'equal' in line], [])

    def test_a_diverged_weight_is_named_weight_dot_key_with_shape_and_address(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, corrupt=('weight.layers[0].attention.convolution', 7)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('replicated draft weight differs', str(raised.exception))
        self.assertIn('buffer=weight.layers[0].attention.convolution shape=(1, 1, 4, 8) address=(8192, 8448)',
                      str(raised.exception))
        self.assertIn('differing=7 of 32', str(raised.exception))

    def test_the_selector_projection_is_checked_first(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, corrupt=('weight.selector_projection', 1)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('buffer=weight.selector_projection shape=(1, 1, 4, 8) address=(20480, 20736)',
                      str(raised.exception))
        self.assertEqual(operations.reads, 9 + 1 + 1,
                         'nine addresses recorded, the first comparison fires, its address is re-read for the message')

    def test_a_kv_layer_is_named_with_its_index_and_head(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, kv_layers=3, corrupt=('kv_history[1].v', 5)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('replicated draft kv differs', str(raised.exception))
        self.assertIn('buffer=kv_history[1].v shape=(1, 1, 4, 8) address=(38912, 39168) first_seen_as=kv_history[1].v '
                      'differing=5 of 32', str(raised.exception))

    def test_a_swapped_history_reports_the_name_it_was_first_seen_under(self):
        """After a commit the histories trade roles, so the address is what identifies the buffer."""
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        victim.history, victim.spare_history = victim.spare_history, victim.history
        victim.history.shards[1].data[0, :2] += 2.0
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('buffer=history shape=(1, 1, 4, 8) address=(28672, 28928) first_seen_as=spare_history',
                      str(raised.exception))

    def test_a_buffer_allocated_after_first_sight_has_no_origin(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        victim.history = replicated(0xF000)
        victim.history.shards[1].data[0, :1] += 2.0
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('address=(61440, 61696) first_seen_as=None', str(raised.exception))

    def test_the_switch_off_means_no_checks(self):
        with patch.object(module, 'SHARD_CHECK', False):
            stepped, operations = [], FakeOperations()
            entries = [entry('A', device(operations), stepped),
                       entry('B', device(operations, corrupt=('history', 3)), stepped)]
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(stepped, ['A', 'B'])
        self.assertEqual((operations.reads, self.logger.lines, module.RECORDED), (0, [], {}))

    def test_one_entry_means_no_checks(self):
        stepped, operations = [], FakeOperations()
        module.sequential_packed_step([entry('A', device(operations, corrupt=('history', 3)), stepped)],
                                      cancelled=lambda: False)
        self.assertEqual(stepped, ['A'])
        self.assertEqual((operations.reads, self.logger.lines, module.RECORDED), (0, [], {}))

    def test_a_released_device_has_nothing_to_check(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', closed_device(operations), stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(self.logger.lines[:2], ['[PINDIAG] draft buffer addresses for B: {}',
                                                 '[PINDIAG] shards equal after step of A: 0 buffers'])

    def test_the_switch_is_read_once_at_import_and_defaults_off(self):
        with patch.dict(os.environ, {'QWEN_FAST_SHARD_CHECK': '1'}):
            self.assertTrue(importlib.reload(module).SHARD_CHECK)
        with patch.dict(os.environ):
            os.environ.pop('QWEN_FAST_SHARD_CHECK', None)
            self.assertFalse(importlib.reload(module).SHARD_CHECK)


if __name__ == '__main__':
    unittest.main()
