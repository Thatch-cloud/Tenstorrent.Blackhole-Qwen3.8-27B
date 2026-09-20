"""The shard-equality diagnostic behind QWEN_FAST_SHARD_CHECK: off by default,
silent with one user, short enough for the log capture, and exact about who
wrote over whom when it fires - raising at '1', continuing at 'warn'."""

import importlib
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import serving_sequential_step as module

# The log capture truncates around 250 characters; every diagnostic line stays under this.
LINE_BUDGET = 180
# What a vLLM request id looks like on the rig: 'cmpl-' plus 32 hex plus '-0'.
LONG_A, LONG_B = 'cmpl-' + '8e11' * 8 + '-0', 'cmpl-' + 'b76d' * 8 + '-0'


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


def device(operations, *, kv_layers=1, corrupt=(), proposal_calls=1, base=0):
    """`corrupt` is (buffer name, elements) pairs: change that many on chip 1 only.
    Addresses follow construction order from `base`: layer-0 weights lowest, K/V highest."""
    named = [('weight.layers[0].attention.norm', 0x1000), ('weight.layers[0].attention.convolution', 0x2000),
             ('weight.layers[0].mlp.device_norm', 0x3000), ('weight.final_norm', 0x4000),
             ('weight.selector_projection', 0x5000), ('history', 0x6000), ('spare_history', 0x7000)]
    named.extend(('kv_history[%d].%s' % (layer, head), 0x8000 + layer * 0x1000 + index * 0x800)
                 for layer in range(kv_layers) for index, head in enumerate(('k', 'v')))
    tensors = {name: replicated(base + address) for name, address in named}
    for name, count in ([corrupt] if corrupt and isinstance(corrupt[0], str) else corrupt):
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


class ShardCheckBase(unittest.TestCase):
    mode = '1'

    def setUp(self):
        self.logger = FakeLogger()
        for patcher in (patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=self.logger)}),
                        patch.object(module, 'SHARD_CHECK', self.mode)):
            patcher.start()
            self.addCleanup(patcher.stop)
        module.RECORDED.clear()
        module.REPORTED.clear()
        self.addCleanup(module.RECORDED.clear)
        self.addCleanup(module.REPORTED.clear)

    def lines(self, marker):
        return [line for line in self.logger.lines if marker in line]

    def mismatch_lines(self):
        return self.lines('mismatch')


class RaiseModeTests(ShardCheckBase):
    def test_equal_shards_pass_and_the_log_proves_the_check_ran(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('B', device(operations, kv_layers=2), stepped),
                   entry('A', device(operations, kv_layers=2), stepped)]
        outputs = module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual(self.lines('equal'), ['[PINDIAG] shards equal after step of B: 11 buffers',
                                              '[PINDIAG] shards equal after step of A: 11 buffers'],
                         'five weights, two histories and two K/V per layer, for the other user')
        self.assertEqual(self.mismatch_lines(), [])
        # eleven buffers, each read once to record its address and once to compare, for each of two users
        self.assertEqual(operations.reads, 44)

    def test_addresses_are_logged_one_line_per_buffer_once_per_request(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', device(operations), stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        module.sequential_packed_step(entries, cancelled=lambda: False)
        recorded = self.lines('[PINDIAG] address ')
        self.assertEqual(len(recorded), 18, 'nine buffers for each request, once per request not once per step')
        self.assertEqual(recorded[:2], ['[PINDIAG] address B weight.selector_projection 20480 20736',
                                        '[PINDIAG] address B weight.final_norm 16384 16640'])
        self.assertIn('[PINDIAG] address B weight.layers[0].attention.norm 4096 4352', recorded)
        self.assertIn('[PINDIAG] address A kv_history[0].v 34816 35072', recorded)
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

    def test_a_diverged_history_logs_four_short_lines_then_raises_briefly(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations, proposal_calls=2), stepped),
                   entry('B', device(operations, corrupt=('history', 3)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(self.mismatch_lines(), [
            '[PINDIAG] shard mismatch after step of A (entry 0): victim=B (entry 1) category=history',
            '[PINDIAG] mismatch buffer=history shape=(1, 1, 4, 8) address=(24576, 24832) first_seen_as=history',
            '[PINDIAG] mismatch differing=3 of 32 max_abs=2',
            "[PINDIAG] mismatch scheduler order=['A', 'B'] proposal_calls=[2, 1]"])
        message = str(raised.exception)
        self.assertEqual(message, 'Replicated draft history differs between chips: victim=B buffer=history; '
                                  'see the [PINDIAG] mismatch log lines')
        self.assertLess(len(message), LINE_BUDGET)
        self.assertEqual(stepped, ['A'], 'the victim is caught before it steps on the damage')
        self.assertEqual(self.lines('equal') + self.lines('differ after'), [], 'no summary line after a raise')

    def test_every_line_fits_the_log_capture_with_rig_sized_ids_and_addresses(self):
        stepped, operations = [], FakeOperations()
        entries = [entry(LONG_A, device(operations, base=0x2_0000_0000, kv_layers=5), stepped),
                   entry(LONG_B, device(operations, base=0x2_4000_0000, kv_layers=5,
                                        corrupt=('weight.layers[0].attention.convolution', 7)), stepped)]
        with self.assertRaises(AssertionError):
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(len(self.mismatch_lines()), 4)
        for line in self.logger.lines:
            self.assertLess(len(line), LINE_BUDGET, line)
        self.assertIn('[PINDIAG] mismatch buffer=weight.layers[0].attention.convolution shape=(1, 1, 4, 8) '
                      'address=(9663684608, 9663684864) first_seen_as=weight.layers[0].attention.convolution',
                      self.mismatch_lines())

    def test_a_diverged_weight_is_named_weight_dot_key_with_shape_and_address(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, corrupt=('weight.layers[0].attention.convolution', 7)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('Replicated draft weight differs between chips: victim=B '
                      'buffer=weight.layers[0].attention.convolution', str(raised.exception))
        self.assertIn('[PINDIAG] mismatch buffer=weight.layers[0].attention.convolution shape=(1, 1, 4, 8) '
                      'address=(8192, 8448) first_seen_as=weight.layers[0].attention.convolution', self.mismatch_lines())
        self.assertIn('[PINDIAG] mismatch differing=7 of 32 max_abs=2', self.mismatch_lines())

    def test_the_selector_projection_is_checked_first(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, corrupt=('weight.selector_projection', 1)), stepped)]
        with self.assertRaises(AssertionError):
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('[PINDIAG] mismatch buffer=weight.selector_projection shape=(1, 1, 4, 8) '
                      'address=(20480, 20736) first_seen_as=weight.selector_projection', self.mismatch_lines())
        self.assertEqual(operations.reads, 9 + 1 + 1,
                         'nine addresses recorded, the first comparison fires, its address is re-read for the message')

    def test_a_kv_layer_is_named_with_its_index_and_head(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, kv_layers=3, corrupt=('kv_history[1].v', 5)), stepped)]
        with self.assertRaises(AssertionError) as raised:
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('Replicated draft kv differs', str(raised.exception))
        self.assertIn('[PINDIAG] mismatch buffer=kv_history[1].v shape=(1, 1, 4, 8) address=(38912, 39168) '
                      'first_seen_as=kv_history[1].v', self.mismatch_lines())
        self.assertIn('[PINDIAG] mismatch differing=5 of 32 max_abs=2', self.mismatch_lines())

    def test_a_swapped_history_reports_the_name_it_was_first_seen_under(self):
        """After a commit the histories trade roles, so the address is what identifies the buffer."""
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        victim.history, victim.spare_history = victim.spare_history, victim.history
        victim.history.shards[1].data[0, :2] += 2.0
        with self.assertRaises(AssertionError):
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('[PINDIAG] mismatch buffer=history shape=(1, 1, 4, 8) address=(28672, 28928) '
                      'first_seen_as=spare_history', self.mismatch_lines())

    def test_a_buffer_allocated_after_first_sight_has_no_origin(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        victim.history = replicated(0xF000)
        victim.history.shards[1].data[0, :1] += 2.0
        with self.assertRaises(AssertionError):
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertIn('address=(61440, 61696) first_seen_as=None', self.mismatch_lines()[1])

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
        self.assertEqual(self.logger.lines[0], '[PINDIAG] shards equal after step of A: 0 buffers')
        self.assertEqual(self.lines('address B '), [], 'nothing to record for the released device')
        self.assertEqual(len(self.lines('address A ')), 9, 'the live device is still recorded after B steps')


class WarnModeTests(ShardCheckBase):
    mode = 'warn'

    def test_a_divergence_is_logged_and_decoding_continues(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations, proposal_calls=2), stepped),
                   entry('B', device(operations, corrupt=('history', 3)), stepped)]
        outputs = module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['A', 'B'])
        self.assertEqual(stepped, ['A', 'B'], 'the victim still steps')
        self.assertEqual(self.mismatch_lines(), [
            '[PINDIAG] shard mismatch after step of A (entry 0): victim=B (entry 1) category=history',
            '[PINDIAG] mismatch buffer=history shape=(1, 1, 4, 8) address=(24576, 24832) first_seen_as=history',
            '[PINDIAG] mismatch differing=3 of 32 max_abs=2',
            "[PINDIAG] mismatch scheduler order=['A', 'B'] proposal_calls=[2, 1]"])
        self.assertEqual(self.lines('after step of'), [
            '[PINDIAG] shard mismatch after step of A (entry 0): victim=B (entry 1) category=history',
            '[PINDIAG] shards differ after step of A: 8 equal, 1 diverged',
            '[PINDIAG] shards equal after step of B: 9 buffers'])

    def test_a_buffer_that_stays_diverged_is_reported_once(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', device(operations, corrupt=('history', 3)), stepped)]
        for _ in range(3):
            module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(len(self.mismatch_lines()), 4, 'one four-line report, not one per step')
        self.assertEqual(self.lines('differ after'), ['[PINDIAG] shards differ after step of A: 8 equal, 1 diverged'] * 3,
                         'the per-step summary still says it is diverged')
        self.assertEqual(stepped, ['A', 'B'] * 3)

    def test_a_new_divergence_on_another_buffer_is_reported(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations, corrupt=('history', 3))
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        victim.kv_history.active[0]['k'].shards[1].data[0, :4] += 2.0
        module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(len(self.mismatch_lines()), 8)
        self.assertIn('[PINDIAG] mismatch buffer=kv_history[0].k shape=(1, 1, 4, 8) address=(32768, 33024) '
                      'first_seen_as=kv_history[0].k', self.mismatch_lines()[5])
        self.assertEqual(self.lines('differ after')[-1], '[PINDIAG] shards differ after step of A: 7 equal, 2 diverged')

    def test_every_diverged_buffer_in_one_step_is_reported(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped),
                   entry('B', device(operations, corrupt=(('weight.final_norm', 1), ('kv_history[0].v', 2))), stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual([line for line in self.mismatch_lines() if 'buffer=' in line], [
            '[PINDIAG] mismatch buffer=weight.final_norm shape=(1, 1, 4, 8) address=(16384, 16640) '
            'first_seen_as=weight.final_norm',
            '[PINDIAG] mismatch buffer=kv_history[0].v shape=(1, 1, 4, 8) address=(34816, 35072) '
            'first_seen_as=kv_history[0].v'])
        self.assertEqual(module.REPORTED, {('B', 'weight.final_norm'), ('B', 'kv_history[0].v')})

    def test_reports_are_pruned_with_the_victim(self):
        stepped, operations = [], FakeOperations()
        first = entry('A', device(operations), stepped)
        module.sequential_packed_step([first, entry('B', device(operations, corrupt=('history', 3)), stepped)],
                                      cancelled=lambda: False)
        self.assertEqual(module.REPORTED, {('B', 'history')})
        module.sequential_packed_step([first, entry('C', device(operations), stepped)], cancelled=lambda: False)
        self.assertEqual(module.REPORTED, set())
        # and a returning B with the same damage is reported afresh
        module.sequential_packed_step([first, entry('B', device(operations, corrupt=('history', 3)), stepped)],
                                      cancelled=lambda: False)
        self.assertEqual(len(self.mismatch_lines()), 8)


class SwitchTests(ShardCheckBase):
    mode = '0'

    def test_the_switch_off_means_no_checks(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', device(operations, corrupt=('history', 3)), stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(stepped, ['A', 'B'])
        self.assertEqual((operations.reads, self.logger.lines, module.RECORDED), (0, [], {}))

    def test_the_switch_is_read_once_at_import_and_defaults_off(self):
        self.addCleanup(lambda: importlib.reload(module))
        for value, expected in (('1', '1'), ('warn', 'warn'), ('0', '0')):
            with patch.dict(os.environ, {'QWEN_FAST_SHARD_CHECK': value}):
                self.assertEqual(importlib.reload(module).SHARD_CHECK, expected)
        with patch.dict(os.environ):
            os.environ.pop('QWEN_FAST_SHARD_CHECK', None)
            self.assertEqual(importlib.reload(module).SHARD_CHECK, '0')
        with patch.dict(os.environ, {'QWEN_FAST_SHARD_CHECK': 'yes'}):
            with self.assertRaises(ValueError):
                importlib.reload(module)


if __name__ == '__main__':
    unittest.main()
