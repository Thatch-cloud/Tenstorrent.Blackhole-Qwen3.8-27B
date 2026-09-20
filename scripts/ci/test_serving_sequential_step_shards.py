"""The shard-equality diagnostic behind QWEN_FAST_SHARD_CHECK: off by default,
silent with one user, short enough for the log capture, and exact about who
wrote over whom when it fires - raising at '1', continuing at 'warn'. The
replicated buffers are compared across the chips; the K/V banks, which differ
per chip by construction, are compared per chip against their owner's last step."""

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
# Construction order: layer-0 weights lowest, then the histories, then the K/V.
REPLICATED = [('weight.layers[0].attention.norm', 0x1000), ('weight.layers[0].attention.convolution', 0x2000),
              ('weight.layers[0].mlp.device_norm', 0x3000), ('weight.final_norm', 0x4000),
              ('weight.selector_projection', 0x5000), ('history', 0x6000), ('spare_history', 0x7000)]


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


def buffer(address, *, offset=0, rows=4, columns=8):
    """One buffer, chip 1 one page up. `offset` is what chip 1 adds to every
    element: zero for a replicated buffer, nonzero for a K/V bank, whose two
    chips hold different heads by construction."""
    left = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return FakeTensor([FakeShard(left.to(torch.bfloat16), address),
                       FakeShard((left + offset).to(torch.bfloat16), address + 0x100)], (1, 1, rows, columns))


def device(operations, *, kv_layers=1, history_rows=4, proposal_calls=1, base=0):
    named = {name: buffer(base + address) for name, address in REPLICATED}
    for layer in range(kv_layers):
        for index, head in enumerate(('k', 'v')):
            named['kv_history[%d].%s' % (layer, head)] = buffer(base + 0x8000 + layer * 0x1000 + index * 0x800, offset=100)
            named['kv_spare[%d].%s' % (layer, head)] = buffer(base + 0x20000 + layer * 0x1000 + index * 0x800, offset=100)
    attention = dict(norm=named['weight.layers[0].attention.norm'],
                     convolution=named['weight.layers[0].attention.convolution'])
    mlp = dict(device_norm=named['weight.layers[0].mlp.device_norm'])
    active = [{head: named['kv_history[%d].%s' % (layer, head)] for head in ('k', 'v')} for layer in range(kv_layers)]
    spare = [{head: named['kv_spare[%d].%s' % (layer, head)] for head in ('k', 'v')} for layer in range(kv_layers)]
    return SimpleNamespace(operations=operations, closed=False, layers=[(attention, mlp, None, None)],
                           selector_projection=named['weight.selector_projection'],
                           final_norm=named['weight.final_norm'],
                           history=named['history'], spare_history=named['spare_history'],
                           kv_history=SimpleNamespace(active=active, spare=spare, history_rows=history_rows),
                           proposal_calls=proposal_calls, named=named)


def closed_device(operations):
    return SimpleNamespace(operations=operations, closed=True, layers=[], selector_projection=None,
                           final_norm=None, history=None, spare_history=None, kv_history=None, proposal_calls=0)


def damage(tensor, count, *, row=0, chip=1):
    """Change `count` elements of one row on one chip only."""
    tensor.shards[chip].data[row, :count] += 2.0


def entry(request_id, drafter, stepped, on_step=None):
    def step(name, *, cancelled):
        stepped.append(name)
        if on_step is not None:
            on_step(drafter)
        return SimpleNamespace(request_id=name, token_ids=[1])

    return dict(request_id=request_id, ticket=SimpleNamespace(request_id=request_id),
                request=SimpleNamespace(step=step, runtime=SimpleNamespace(drafter=drafter)))


def commit(drafter):
    """What the owner's own step does to its K/V: publish into the spare, swap, grow."""
    drafter.kv_history.active, drafter.kv_history.spare = drafter.kv_history.spare, drafter.kv_history.active
    damage(drafter.kv_history.active[0]['k'], 8, row=0, chip=0)
    damage(drafter.kv_history.active[0]['k'], 8, row=0, chip=1)
    drafter.kv_history.history_rows = min(4, drafter.kv_history.history_rows + 1)


class ShardCheckBase(unittest.TestCase):
    mode = '1'

    def setUp(self):
        self.logger = FakeLogger()
        for patcher in (patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=self.logger)}),
                        patch.object(module, 'SHARD_CHECK', self.mode)):
            patcher.start()
            self.addCleanup(patcher.stop)
        for table in (module.RECORDED, module.REPORTED, module.SNAPSHOTS):
            table.clear()
            self.addCleanup(table.clear)

    def lines(self, marker):
        return [line for line in self.logger.lines if marker in line]

    def reports(self):
        return [line for line in self.logger.lines if 'mismatch' in line or 'drift' in line]

    def run_rounds(self, entries, rounds=1):
        for _ in range(rounds):
            module.sequential_packed_step(entries, cancelled=lambda: False)


class RaiseModeTests(ShardCheckBase):
    def test_equal_shards_pass_and_the_summary_carries_proposal_calls(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('B', device(operations, kv_layers=2), stepped),
                   entry('A', device(operations, kv_layers=2, proposal_calls=3), stepped)]
        outputs = module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['B', 'A'])
        self.assertEqual(self.lines('equal'), ['[PINDIAG] shards equal after step of B: 11 buffers proposal_calls=[1, 3]',
                                              '[PINDIAG] shards equal after step of A: 11 buffers proposal_calls=[1, 3]'],
                         'five weights, two histories and two K/V banks per layer, for the other user')
        self.assertEqual(self.reports(), [])
        # Bounded: 4 banks snapshotted per request before the round (8); then per
        # step, 4 to refresh the actor's snapshot, 15 to record the other's
        # addresses once, 7 replicated compares and 4 bank compares (30 twice).
        self.assertEqual(operations.reads, 8 + 30 + 30)

    def test_addresses_are_logged_one_line_per_buffer_once_per_request(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', device(operations), stepped)]
        self.run_rounds(entries, 2)
        recorded = self.lines('[PINDIAG] address ')
        self.assertEqual(len(recorded), 22, 'eleven per request - weights, histories, active and spare K/V - once each')
        self.assertEqual(recorded[:2], ['[PINDIAG] address B weight.selector_projection 20480 20736',
                                        '[PINDIAG] address B weight.final_norm 16384 16640'])
        self.assertIn('[PINDIAG] address B weight.layers[0].attention.norm 4096 4352', recorded)
        self.assertIn('[PINDIAG] address B kv_spare[0].v 133120 133376', recorded)
        self.assertIn('[PINDIAG] address A kv_history[0].v 34816 35072', recorded)
        self.assertEqual(list(module.RECORDED['B']), ['weight.selector_projection', 'weight.final_norm',
            'weight.layers[0].attention.norm', 'weight.layers[0].attention.convolution',
            'weight.layers[0].mlp.device_norm', 'history', 'spare_history', 'kv_history[0].k', 'kv_history[0].v',
            'kv_spare[0].k', 'kv_spare[0].v'])

    def test_records_and_snapshots_are_pruned_to_the_live_requests(self):
        stepped, operations = [], FakeOperations()
        first, second = entry('A', device(operations), stepped), entry('B', device(operations), stepped)
        self.run_rounds([first, second])
        self.assertEqual((set(module.RECORDED), set(module.SNAPSHOTS)), ({'A', 'B'}, {'A', 'B'}))
        self.run_rounds([second, entry('C', device(operations), stepped)])
        self.assertEqual((set(module.RECORDED), set(module.SNAPSHOTS)), ({'B', 'C'}, {'B', 'C'}))

    def test_a_diverged_history_logs_four_short_lines_then_raises_briefly(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['history'], 3)
        entries = [entry('A', device(operations, proposal_calls=2), stepped), entry('B', victim, stepped)]
        with self.assertRaises(AssertionError) as raised:
            self.run_rounds(entries)
        self.assertEqual(self.reports(), [
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

    def test_a_kv_bank_that_differs_between_the_chips_is_not_a_mismatch(self):
        """Run 35482551725: every bank differed across the chips on every step,
        because each chip holds its own heads. That is structure, not a scribble."""
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations, kv_layers=5), stepped), entry('B', device(operations, kv_layers=5), stepped)]
        self.run_rounds(entries, 4)
        self.assertEqual(self.reports(), [])
        self.assertEqual(len(self.lines('shards equal after step')), 8)

    def test_a_kv_bank_rewritten_between_its_owners_steps_is_a_drift_on_that_chip(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations, kv_layers=3, history_rows=3)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        self.run_rounds(entries)
        damage(victim.kv_history.active[1]['v'], 5, row=2, chip=0)
        with self.assertRaises(AssertionError) as raised:
            self.run_rounds(entries)
        self.assertEqual(self.reports(), [
            '[PINDIAG] kv drift after step of A (entry 0): victim=B (entry 1) chip=0 rows=3',
            '[PINDIAG] drift buffer=kv_history[1].v shape=(1, 1, 4, 8) address=(38912, 39168) first_seen_as=kv_history[1].v',
            '[PINDIAG] drift differing=5 of 24 max_abs=2',
            "[PINDIAG] drift scheduler order=['A', 'B'] proposal_calls=[1, 1]"])
        self.assertEqual(str(raised.exception), "Draft K/V drifted on chip 0 between its owner's steps: victim=B "
                                                'buffer=kv_history[1].v; see the [PINDIAG] drift log lines')
        self.assertEqual(stepped, ['A', 'B', 'A'])

    def test_kv_rows_beyond_the_committed_history_are_not_checked(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations, history_rows=3)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        self.run_rounds(entries)
        damage(victim.kv_history.active[0]['k'], 8, row=3, chip=0)
        damage(victim.kv_history.active[0]['k'], 8, row=3, chip=1)
        self.run_rounds(entries)
        self.assertEqual(self.reports(), [])
        self.assertEqual(self.lines('after step of A')[-1], '[PINDIAG] shards equal after step of A: 9 buffers proposal_calls=[1, 1]')

    def test_a_kv_bank_rewritten_by_its_owners_own_step_is_not_a_drift(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations, history_rows=2), stepped, on_step=commit),
                   entry('B', device(operations, history_rows=2), stepped, on_step=commit)]
        self.run_rounds(entries, 3)
        self.assertEqual(self.reports(), [])
        self.assertEqual(len(self.lines('shards equal after step')), 6)

    def test_a_drifted_bank_that_was_the_spare_reports_the_name_it_was_first_seen_under(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped, on_step=commit)]
        self.run_rounds(entries)
        damage(victim.kv_history.active[0]['k'], 2, row=1, chip=1)
        with self.assertRaises(AssertionError):
            self.run_rounds(entries)
        self.assertEqual(self.reports()[:2], [
            '[PINDIAG] kv drift after step of A (entry 0): victim=B (entry 1) chip=1 rows=4',
            '[PINDIAG] drift buffer=kv_history[0].k shape=(1, 1, 4, 8) address=(131072, 131328) first_seen_as=kv_spare[0].k'])

    def test_every_line_fits_the_log_capture_with_rig_sized_ids_and_addresses(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations, base=0x2_4000_0000, kv_layers=5)
        damage(victim.named['weight.layers[0].attention.convolution'], 7)
        entries = [entry(LONG_A, device(operations, base=0x2_0000_0000, kv_layers=5), stepped), entry(LONG_B, victim, stepped)]
        with self.assertRaises(AssertionError):
            self.run_rounds(entries)
        self.assertIn('[PINDIAG] mismatch buffer=weight.layers[0].attention.convolution shape=(1, 1, 4, 8) '
                      'address=(9663684608, 9663684864) first_seen_as=weight.layers[0].attention.convolution', self.reports())
        damage(victim.named['weight.layers[0].attention.convolution'], 7, chip=0)
        damage(victim.kv_history.active[4]['v'], 3, row=1, chip=1)
        with self.assertRaises(AssertionError):
            self.run_rounds(entries)
        self.assertEqual(len(self.reports()), 8, 'one mismatch report, then one drift report')
        self.assertIn('[PINDIAG] drift buffer=kv_history[4].v shape=(1, 1, 4, 8) address=(9663727616, 9663727872) '
                      'first_seen_as=kv_history[4].v', self.reports())
        for line in self.logger.lines:
            self.assertLess(len(line), LINE_BUDGET, line)

    def test_a_diverged_weight_is_named_weight_dot_key_with_shape_and_address(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['weight.layers[0].attention.convolution'], 7)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        with self.assertRaises(AssertionError) as raised:
            self.run_rounds(entries)
        self.assertIn('Replicated draft weight differs between chips: victim=B '
                      'buffer=weight.layers[0].attention.convolution', str(raised.exception))
        self.assertIn('[PINDIAG] mismatch buffer=weight.layers[0].attention.convolution shape=(1, 1, 4, 8) '
                      'address=(8192, 8448) first_seen_as=weight.layers[0].attention.convolution', self.reports())
        self.assertIn('[PINDIAG] mismatch differing=7 of 32 max_abs=2', self.reports())

    def test_the_selector_projection_is_checked_first(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['weight.selector_projection'], 1)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        with self.assertRaises(AssertionError):
            self.run_rounds(entries)
        self.assertIn('[PINDIAG] mismatch buffer=weight.selector_projection shape=(1, 1, 4, 8) '
                      'address=(20480, 20736) first_seen_as=weight.selector_projection', self.reports())
        # two banks snapshotted per request before the round, two to refresh the
        # actor's, eleven addresses recorded, the first comparison fires, its
        # address is re-read for the message
        self.assertEqual(operations.reads, 4 + 2 + 11 + 1 + 1)

    def test_a_swapped_history_reports_the_name_it_was_first_seen_under(self):
        """After a commit the histories trade roles, so the address is what identifies the buffer."""
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        self.run_rounds(entries)
        victim.history, victim.spare_history = victim.spare_history, victim.history
        damage(victim.history, 2)
        with self.assertRaises(AssertionError):
            self.run_rounds(entries)
        self.assertIn('[PINDIAG] mismatch buffer=history shape=(1, 1, 4, 8) address=(28672, 28928) '
                      'first_seen_as=spare_history', self.reports())

    def test_a_buffer_allocated_after_first_sight_has_no_origin(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        self.run_rounds(entries)
        victim.history = buffer(0xF000)
        damage(victim.history, 1)
        with self.assertRaises(AssertionError):
            self.run_rounds(entries)
        self.assertIn('address=(61440, 61696) first_seen_as=None', self.reports()[1])

    def test_one_entry_means_no_checks(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['history'], 3)
        module.sequential_packed_step([entry('A', victim, stepped)], cancelled=lambda: False)
        self.assertEqual(stepped, ['A'])
        self.assertEqual((operations.reads, self.logger.lines, module.RECORDED, module.SNAPSHOTS), (0, [], {}, {}))

    def test_a_released_device_has_nothing_to_check(self):
        stepped, operations = [], FakeOperations()
        entries = [entry('A', device(operations), stepped), entry('B', closed_device(operations), stepped)]
        self.run_rounds(entries)
        self.assertEqual(self.logger.lines[0], '[PINDIAG] shards equal after step of A: 0 buffers proposal_calls=[1, 0]')
        self.assertEqual(self.lines('address B '), [], 'nothing to record for the released device')
        self.assertEqual(len(self.lines('address A ')), 11, 'the live device is still recorded after B steps')


class WarnModeTests(ShardCheckBase):
    mode = 'warn'

    def test_a_divergence_is_logged_and_decoding_continues(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['history'], 3)
        entries = [entry('A', device(operations, proposal_calls=2), stepped), entry('B', victim, stepped)]
        outputs = module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual([output.request_id for output in outputs], ['A', 'B'])
        self.assertEqual(stepped, ['A', 'B'], 'the victim still steps')
        self.assertEqual(self.reports(), [
            '[PINDIAG] shard mismatch after step of A (entry 0): victim=B (entry 1) category=history',
            '[PINDIAG] mismatch buffer=history shape=(1, 1, 4, 8) address=(24576, 24832) first_seen_as=history',
            '[PINDIAG] mismatch differing=3 of 32 max_abs=2',
            "[PINDIAG] mismatch scheduler order=['A', 'B'] proposal_calls=[2, 1]"])
        self.assertEqual(self.lines('after step of'), [
            '[PINDIAG] shard mismatch after step of A (entry 0): victim=B (entry 1) category=history',
            '[PINDIAG] shards differ after step of A: 8 equal, 1 diverged proposal_calls=[2, 1]',
            '[PINDIAG] shards equal after step of B: 9 buffers proposal_calls=[2, 1]'])

    def test_a_buffer_that_stays_diverged_is_reported_once(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['history'], 3)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        self.run_rounds(entries, 3)
        self.assertEqual(len(self.reports()), 4, 'one four-line report, not one per step')
        self.assertEqual(self.lines('differ after'),
                         ['[PINDIAG] shards differ after step of A: 8 equal, 1 diverged proposal_calls=[1, 1]'] * 3,
                         'the per-step summary still says it is diverged')
        self.assertEqual(stepped, ['A', 'B'] * 3)

    def test_a_kv_drift_is_reported_per_chip_and_counted_per_bank(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        self.run_rounds(entries)
        damage(victim.kv_history.active[0]['k'], 4, chip=0)
        damage(victim.kv_history.active[0]['k'], 6, chip=1)
        self.run_rounds(entries, 2)
        self.assertEqual([line for line in self.reports() if 'kv drift' in line], [
            '[PINDIAG] kv drift after step of A (entry 0): victim=B (entry 1) chip=0 rows=4',
            '[PINDIAG] kv drift after step of A (entry 0): victim=B (entry 1) chip=1 rows=4'])
        self.assertEqual(len(self.reports()), 8, 'each chip once')
        self.assertEqual(module.REPORTED, {('B', 'kv_history[0].k', 0), ('B', 'kv_history[0].k', 1)})
        # The owner's own step is the baseline, so once B has stepped the written
        # rows are what B is expected to hold and the next round reads equal.
        self.assertEqual(self.lines('after step of A')[-2:], [
            '[PINDIAG] shards differ after step of A: 8 equal, 1 diverged proposal_calls=[1, 1]',
            '[PINDIAG] shards equal after step of A: 9 buffers proposal_calls=[1, 1]'])
        self.assertEqual(stepped, ['A', 'B'] * 3, 'decoding continued')

    def test_a_new_divergence_on_another_buffer_is_reported(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['weight.final_norm'], 1)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        self.run_rounds(entries)
        damage(victim.kv_history.active[0]['v'], 2, chip=1)
        self.run_rounds(entries)
        self.assertEqual(len(self.reports()), 8)
        self.assertEqual(self.reports()[5], '[PINDIAG] drift buffer=kv_history[0].v shape=(1, 1, 4, 8) address=(34816, 35072) '
                                            'first_seen_as=kv_history[0].v')
        self.assertEqual(self.lines('differ after')[-1],
                         '[PINDIAG] shards differ after step of A: 7 equal, 2 diverged proposal_calls=[1, 1]')
        self.assertEqual(module.REPORTED, {('B', 'weight.final_norm', None), ('B', 'kv_history[0].v', 1)})

    def test_reports_are_pruned_with_the_victim(self):
        stepped, operations = [], FakeOperations()
        first = entry('A', device(operations), stepped)
        victim = device(operations)
        damage(victim.named['history'], 3)
        self.run_rounds([first, entry('B', victim, stepped)])
        self.assertEqual(module.REPORTED, {('B', 'history', None)})
        self.run_rounds([first, entry('C', device(operations), stepped)])
        self.assertEqual(module.REPORTED, set())
        # and a returning B with the same damage is reported afresh
        self.run_rounds([first, entry('B', victim, stepped)])
        self.assertEqual(len(self.reports()), 8)


class SwitchTests(ShardCheckBase):
    mode = '0'

    def test_the_switch_off_means_no_checks(self):
        stepped, operations = [], FakeOperations()
        victim = device(operations)
        damage(victim.named['history'], 3)
        entries = [entry('A', device(operations), stepped), entry('B', victim, stepped)]
        module.sequential_packed_step(entries, cancelled=lambda: False)
        self.assertEqual(stepped, ['A', 'B'])
        self.assertEqual((operations.reads, self.logger.lines, module.RECORDED, module.SNAPSHOTS), (0, [], {}, {}))

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
