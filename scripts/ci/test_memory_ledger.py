"""memory_ledger (QWEN_FAST_MEMORY_LEDGER): inert by default; with the flag, a read-only
per-phase itemisation whose checks can fail - an allocator delta the walk cannot explain is
UNMATCHED, and a residual at P7 over 1.5 GB is FAILED with the unmatched phases listed."""

import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import memory_ledger
from memory_ledger import MemoryLedger

BANKS = 8
TOTAL = 33_640_000_000


class FakeShard:
    def __init__(self, device, address, shape, dtype, buffer_type):
        self._device, self.address, self.padded_shape, self.dtype = device, address, shape, dtype
        self.buffer_type = buffer_type

    def device(self):
        return self._device

    def buffer_address(self):
        return self.address

    def memory_config(self):
        return SimpleNamespace(buffer_type=self.buffer_type)


class FakeTensor:
    """A two-chip tensor; `sharded` halves the first dimension per chip."""

    def __init__(self, operations, shape, dtype='bf16', sharded=False, buffer_type='dram'):
        self.shards = []
        for chip, device in enumerate(operations.devices):
            local = (shape[0] // 2,) + tuple(shape[1:]) if sharded else tuple(shape)
            address = operations.next_address[chip]
            operations.next_address[chip] += 0x10000
            self.shards.append(FakeShard(device, address, local, dtype, buffer_type))


class FakeOperations:
    Tensor = FakeTensor
    bfloat16, bfloat8_b, bfloat4_b, float32, uint32 = 'bf16', 'bf8', 'bf4', 'fp32', 'u32'
    BufferType = SimpleNamespace(DRAM='dram', L1='l1')

    def __init__(self):
        self.devices = [SimpleNamespace(name='chip0'), SimpleNamespace(name='chip1')]
        self.allocated = {id(device): 0 for device in self.devices}
        self.next_address = [0x100000, 0x100000]
        self.views = 0

    def tensor(self, shape, **options):
        """A tensor the fake allocator charges for exactly (the walk should match it)."""
        tensor = FakeTensor(self, shape, **options)
        if options.get('buffer_type', 'dram') == 'dram':
            ledger = MemoryLedger(self, None, log=lambda line: None, emit=lambda text: None)
            for shard in tensor.shards:
                self.allocated[id(shard.device())] += ledger.shard_bytes(shard)
        return tensor

    def charge(self, per_chip):
        for device in self.devices:
            self.allocated[id(device)] += per_chip

    def get_device_tensors(self, tensor):
        return tensor.shards

    def get_memory_view(self, device, kind):
        self.views += 1
        allocated = self.allocated[id(device)]
        return SimpleNamespace(num_banks=BANKS, total_bytes_per_bank=TOTAL // BANKS,
                               total_bytes_allocated_per_bank=allocated // BANKS,
                               total_bytes_free_per_bank=(TOTAL - allocated) // BANKS,
                               largest_contiguous_bytes_free_per_bank=(TOTAL - allocated) // BANKS // 2)


def ledger_for(operations):
    probe = FakeTensor(operations, (32, 32))
    lines, reports = [], []
    ledger = MemoryLedger(operations, probe, log=lines.append, emit=lambda text: reports.append(json.loads(text)))
    return ledger, lines, reports


class InertByDefaultTests(unittest.TestCase):
    def test_without_the_flag_nothing_is_built_and_every_hook_returns_at_once(self):
        operations = FakeOperations()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(memory_ledger.FLAG, None)
            self.assertFalse(memory_ledger.enabled())
            self.assertIsNone(memory_ledger.begin(operations, 'probe'))
            self.assertIsNone(memory_ledger.active())
            with patch.object(MemoryLedger, 'phase', side_effect=AssertionError('ran')):
                self.assertIsNone(memory_ledger.record('P2', pool=object()))
                self.assertIsNone(memory_ledger.engine_admitted('request', engine_request=object()))
                self.assertIsNone(memory_ledger.first_packed_round(packed_block=object()))
        self.assertEqual(operations.views, 0)

    def test_only_the_value_one_enables_it(self):
        for value, expected in (('1', True), ('0', False), ('true', False), ('', False)):
            with self.subTest(value=value):
                self.assertEqual(memory_ledger.enabled({memory_ledger.FLAG: value}), expected)

    def test_with_the_flag_begin_registers_and_end_clears(self):
        operations = FakeOperations()
        with patch.dict(os.environ, {memory_ledger.FLAG: '1'}):
            try:
                ledger = memory_ledger.begin(operations, 'probe', log=lambda line: None, emit=lambda text: None)
                self.assertIsInstance(ledger, MemoryLedger)
                self.assertIs(memory_ledger.active(), ledger)
            finally:
                memory_ledger.end()
        self.assertIsNone(memory_ledger.active())


class SizingTests(unittest.TestCase):
    def test_tile_bytes_per_dtype(self):
        operations = FakeOperations()
        ledger = MemoryLedger(operations, None)
        shard = lambda dtype: FakeShard(None, 0, (1, 1, 32, 1024), dtype, 'dram')
        self.assertEqual(ledger.shard_bytes(shard('bf16')), 32 * 1024 * 2)
        self.assertEqual(ledger.shard_bytes(shard('bf8')), 32 * 1088)
        self.assertEqual(ledger.shard_bytes(shard('bf4')), 32 * 576)
        self.assertEqual(ledger.shard_bytes(shard('fp32')), 32 * 1024 * 4)
        with self.assertRaisesRegex(ValueError, 'unsized dtype'):
            ledger.shard_bytes(shard('mystery'))

    def test_a_buffers_page_is_its_reported_page_else_its_tile_or_row(self):
        operations = FakeOperations()
        operations.Layout = SimpleNamespace(TILE='tile', ROW_MAJOR='row')
        ledger = MemoryLedger(operations, None, log=lambda line: None, emit=lambda text: None)
        shard = lambda dtype: FakeShard(None, 0, (1, 1, 64, 1000), dtype, 'dram')
        self.assertEqual(ledger.page_bytes(shard('bf16')), 2048)
        self.assertEqual(ledger.page_bytes(shard('bf8')), 1088)
        self.assertEqual(ledger.page_bytes(shard('bf4')), 576)
        row = shard('bf16')
        row.layout = 'row'
        self.assertEqual(ledger.page_bytes(row), 2048)           # 1000 x 2 B, aligned up to 64
        reported = shard('bf4')
        reported.buffer = lambda: SimpleNamespace(page_size=lambda: 13824)
        self.assertEqual(ledger.page_bytes(reported), 13824)


class PhaseTests(unittest.TestCase):
    def test_a_phase_whose_delta_the_walk_explains_is_matched(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        weights = [operations.tensor((64, 5120), sharded=True) for _ in range(3)]
        ledger.phase('P0', model=SimpleNamespace(layers=[SimpleNamespace(w=weight) for weight in weights]))
        self.assertEqual(ledger.checks[0]['status'], 'first')
        self.assertEqual(ledger.checks[0]['chips'][0]['residual'], 0)
        stream = [operations.tensor((32, 5120), dtype='bf4') for _ in range(4)]
        report = ledger.phase('P1', block_stream=dict(streams=stream))
        self.assertEqual(report['check']['status'], 'matched')
        self.assertEqual(report['known']['block_stream']['bytes'], {0: 4 * 32 * 5120 * 576 // 1024, 1: 4 * 32 * 5120 * 576 // 1024})
        self.assertTrue(any(line.startswith('[MEMLEDGER] phase=P1 check=delta status=matched') for line in lines))
        self.assertEqual(reports[-1]['stage'], 'memory_ledger')

    def test_an_allocation_the_walk_does_not_name_is_unmatched(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        ledger.phase('P0')
        operations.charge(200_000_000)        # e.g. TT_CCL buffers nobody walked
        sampler = operations.tensor((32, 1024))
        report = ledger.phase('P3', serving_sampler=sampler)
        self.assertEqual(report['check']['status'], 'UNMATCHED')
        self.assertEqual(report['check']['chips'][0]['unexplained'], 200_000_000)
        self.assertEqual(ledger.unmatched()[0], dict(phase='P3', chip=0, unexplained=200_000_000))

    def test_a_walk_that_names_an_earlier_allocation_is_unmatched_too(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        early = operations.tensor((1024, 5120))
        ledger.phase('P0')                     # allocated but not walked here
        report = ledger.phase('P1', late=early)
        self.assertEqual(report['check']['status'], 'UNMATCHED')
        self.assertLess(report['check']['chips'][0]['unexplained'], 0)

    def test_the_block_streams_measured_rounding_is_matched(self):
        # 64 BF4 layers of 50,319,360 B per chip, allocated as 3,227,516,928 B: one 13,824 B
        # page per bank per buffer (docs/mlp-block-stream.md:28,73), with this fake ttnn
        # reporting no page size (so the tile page, 576 B, bounds the per-buffer term and
        # RELATIVE_TOLERANCE carries the rest).
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        ledger.phase('P0')
        stream = [operations.tensor((5120, 17472), dtype='bf4') for _ in range(64)]
        operations.charge(3_227_516_928 - 64 * 50_319_360)
        report = ledger.phase('P1', block_stream=dict(streams=stream))
        self.assertEqual(report['check']['chips'][0]['walked'], 64 * 50_319_360)
        self.assertEqual(report['check']['status'], 'matched')
        self.assertEqual(report['check']['chips'][0]['unexplained'], 64 * BANKS * 13824)

    def test_bank_rounding_is_one_page_per_bank_per_buffer_and_no_more(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        ledger.phase('P0')
        buffers = [operations.tensor((32, 32)) for _ in range(4)]
        operations.charge(4 * (BANKS - 1) * 2048)   # four one-tile buffers, each a tile in every bank
        self.assertEqual(ledger.phase('P1', stream=buffers)['check']['status'], 'matched')
        more = [operations.tensor((32, 32)) for _ in range(4)]
        operations.charge(4 * BANKS * 2048 + 4096)
        self.assertEqual(ledger.phase('P2', stream=more)['check']['status'], 'UNMATCHED')

    def test_walking_many_small_buffers_buys_no_slack_for_an_unwalked_allocation(self):
        # The old flat allowance (32 KiB per bank per buffer) let 1,000 walked 32x32 tiles
        # hide 200 MB nobody walked (tolerance 262 MB). One page per bank is 16 MB.
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        ledger.phase('P0')
        small = [operations.tensor((32, 32)) for _ in range(1000)]
        operations.charge(200_000_000)
        report = ledger.phase('P2', buffer_pool=small)
        self.assertEqual(report['check']['status'], 'UNMATCHED')
        self.assertEqual(report['check']['chips'][0]['unexplained'], 200_000_000)
        self.assertEqual(report['check']['chips'][0]['tolerance'], 1000 * BANKS * 2048 + int(0.005 * 1000 * 2048))

    def test_the_relative_tolerance_is_half_a_percent(self):
        for over, status in ((4_000_000, 'matched'), (10_000_000, 'UNMATCHED')):
            with self.subTest(over=over):
                operations = FakeOperations()
                ledger, lines, reports = ledger_for(operations)
                ledger.phase('P0')
                weight = operations.tensor((1, 1, 16384, 30518))     # ~1.0 GB of BF16 per chip
                operations.charge(over)
                self.assertEqual(ledger.phase('P1', weight=weight)['check']['status'], status)
        self.assertEqual(memory_ledger.RELATIVE_TOLERANCE, 0.005)

    def test_the_whole_allowance_is_capped_per_phase(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        ledger.phase('P0')
        weights = [operations.tensor((1, 1, 32768, 32768)) for _ in range(9)]    # 19.3 GB per chip
        operations.charge(80_000_000)          # under 0.5% of 19.3 GB, over the cap
        report = ledger.phase('P1', weights=weights)
        self.assertEqual(report['check']['chips'][0]['tolerance'], memory_ledger.TOLERANCE_CAP)
        self.assertEqual(report['check']['status'], 'UNMATCHED')
        self.assertEqual(memory_ledger.TOLERANCE_CAP, 64 * 2 ** 20)

    def test_the_p7_residual_passes_under_the_limit_and_fails_over_it_naming_the_unmatched(self):
        for extra, status in ((1_000_000_000, 'passed'), (2_000_000_000, 'FAILED')):
            with self.subTest(extra=extra):
                operations = FakeOperations()
                ledger, lines, reports = ledger_for(operations)
                ledger.phase('P0')
                operations.charge(extra)
                report = ledger.phase('P7', point='after_attach')
                self.assertEqual(report['residual']['status'], status)
                self.assertEqual(report['residual']['residuals'], [extra, extra])
                self.assertEqual(report['residual']['unmatched'][0]['phase'], 'P7:after_attach')
                listed = [line for line in lines if ' unmatched rank=' in line]
                self.assertEqual(len(listed), 2)
                self.assertTrue(any('check=residual status=%s' % status in line for line in lines))

    def test_a_buffer_reached_twice_counts_once_and_l1_is_not_dram(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        shared = operations.tensor((32, 1024))
        local = operations.tensor((32, 1024), buffer_type='l1')
        report = ledger.phase('P0', first=[shared, local], second={'again': shared})
        self.assertEqual(report['known']['first']['buffers'], {0: 1, 1: 1})
        self.assertEqual(report['known']['second']['buffers'], {})
        self.assertEqual(report['check']['chips'][0]['residual'], 0)

    def test_the_walk_descends_objects_and_containers_but_keeps_no_tensor_alive(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        tensor = operations.tensor((32, 1024))
        holder = SimpleNamespace(inner=SimpleNamespace(table={'x': [(tensor,)]}))
        holder.inner.back = holder              # a cycle
        ledger.phase('P0', holder=holder)
        self.assertEqual(len(ledger.known), 2)
        watch = weakref.ref(tensor)
        del holder, tensor
        gc.collect()
        self.assertIsNone(watch(), 'the ledger must hold addresses and sizes, never the tensor')

    def test_a_failing_reading_is_logged_and_never_raised(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        with patch('serving_buffer_pool.dram_statistics', side_effect=RuntimeError('boom')):
            self.assertIsNone(ledger.phase('P4', audit={}))
        self.assertEqual(lines, ['[MEMLEDGER] phase=P4 error=RuntimeError: boom'])

    def test_an_unavailable_allocator_view_is_reported(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        del FakeOperations.get_memory_view
        try:
            report = ledger.phase('P0')
        finally:
            FakeOperations.get_memory_view = FakeOperationsView
        self.assertIn('unavailable', report['chips'])
        self.assertTrue(lines[0].startswith('[MEMLEDGER] phase=P0 dram unavailable'))

    def test_every_log_line_fits_the_capture(self):
        # The message budget is dflash_device.AUDIT_LINE_BUDGET's 180 (loguru's prefix takes
        # the rest of the ~250 captured), at every phase, with the 48-character request ids
        # serving passes, UNMATCHED checks, the residual listing and negative figures.
        request = 'chatcmpl-' + 'f' * 32 + '-0-abcd'
        self.assertEqual(len(request), 48)
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        with patch.object(memory_ledger, '_active', ledger):
            memory_ledger.record('P0', **{'model.gdn_states_and_scratch.%d' % index: operations.tensor((32, 1024))
                                          for index in range(5)})
            operations.charge(3_000_000_000)
            memory_ledger.record('P7', point='after_attach')
            memory_ledger.record('prefill', point='before prompt=131072')
            operations.charge(-12_000_000_000)
            memory_ledger.record('prefill', point='after req=%s' % memory_ledger.short_id(request), request=request,
                                 model_after_prefill=operations.tensor((32, 1024)))
            for index in range(6):
                operations.charge(-1_234_567_890)
                memory_ledger.engine_admitted(request, engine_request=operations.tensor((32, 1024)))
            memory_ledger.first_packed_round(packed_block=operations.tensor((32, 1024)))
            with patch('serving_buffer_pool.dram_statistics', side_effect=RuntimeError('x' * 400)):
                memory_ledger.record('P13', point='before_shutdown')
        self.assertTrue(any('check=delta status=UNMATCHED chip1' in line for line in lines))
        self.assertTrue(any(line.startswith('[MEMLEDGER] phase=P8 point=req=%s ' % request[-12:]) for line in lines))
        self.assertLessEqual(max(len(line) for line in lines), 180, max(lines, key=len))
        self.assertEqual(memory_ledger.LINE_BUDGET, 180)
        # The JSON keeps the full id.
        self.assertEqual([report['request'] for report in reports if report['phase'] == 'P8'], [request])

    def test_an_over_long_message_continues_rather_than_being_lost(self):
        lines = []
        ledger = MemoryLedger(FakeOperations(), None, log=lines.append, emit=lambda text: None)
        ledger.log('[MEMLEDGER] ' + 'y' * 400)
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(len(line) <= 180 for line in lines))
        self.assertEqual(''.join([lines[0]] + [line[len('[MEMLEDGER] ...'):] for line in lines[1:]]),
                         '[MEMLEDGER] ' + 'y' * 400)


class ReportFileTests(unittest.TestCase):
    def test_the_report_goes_to_the_named_file_else_the_gate_results_else_stdout(self):
        self.assertEqual(memory_ledger.report_path({memory_ledger.REPORT_ENV: '/x/ledger.jsonl'}), '/x/ledger.jsonl')
        with patch('os.path.isdir', return_value=True):
            self.assertEqual(memory_ledger.report_path({}),
                             os.path.join('/experiment-results-gate', 'memory-ledger.jsonl'))
        with patch('os.path.isdir', return_value=False):
            self.assertIsNone(memory_ledger.report_path({}))

    def test_each_phase_appends_one_json_line_and_begin_names_the_file(self):
        operations = FakeOperations()
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'ledger.jsonl'
            lines = []
            with patch.dict(os.environ, {memory_ledger.FLAG: '1', memory_ledger.REPORT_ENV: str(target)}):
                try:
                    ledger = memory_ledger.begin(operations, FakeTensor(operations, (32, 32)), log=lines.append)
                    memory_ledger.record('P0', weight=operations.tensor((32, 1024)))
                    memory_ledger.record('P1')
                finally:
                    memory_ledger.end()
            self.assertEqual(lines[0], '[MEMLEDGER] report=%s' % target)
            written = target.read_text(encoding='utf-8').splitlines()
            self.assertEqual([json.loads(line)['phase'] for line in written], ['P0', 'P1'])
            self.assertEqual(ledger.report, str(target))


FakeOperationsView = FakeOperations.get_memory_view


class HookTests(unittest.TestCase):
    def test_engines_are_p8_to_p11_then_numbered_and_the_first_packed_round_is_p12_once(self):
        operations = FakeOperations()
        ledger, lines, reports = ledger_for(operations)
        with patch.object(memory_ledger, '_active', ledger):
            memory_ledger.record('P7', point='after_attach')
            for index in range(5):
                memory_ledger.engine_admitted('request-%d' % index)
            memory_ledger.first_packed_round(packed_block=None)
            memory_ledger.first_packed_round(packed_block=None)
            memory_ledger.record('P13', point='before_shutdown')
        self.assertEqual([(check['phase'], check['point']) for check in ledger.checks], [
            ('P7', 'after_attach'), ('P8', 'req=request-0'), ('P9', 'req=request-1'),
            ('P10', 'req=request-2'), ('P11', 'req=request-3'), ('engine5', 'req=request-4'),
            ('P12', 'first_packed_round'), ('P13', 'before_shutdown')])
        self.assertTrue(all(line.startswith('[MEMLEDGER] phase=') for line in lines))


if __name__ == '__main__':
    unittest.main()
