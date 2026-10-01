"""trace_census: the sequential-hang diagnostics. Flag-gated, host-only; the capture twin is bound at four cards only and
returns the original's result, with the original's calls, unless QWEN_FAST_TRACE_CENSUS=1."""

import hashlib
import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import attention_batch
import dflash_proposal_trace
import packed_verifier
import tp_addresses
import trace_census
import verifier_engine

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_CAPTURE = attention_batch.capture_operation
ORIGINAL_NOTE = verifier_engine.note_packed_step


class FakeShard:
    def __init__(self, device, address, shape):
        self._device, self.address, self.padded_shape, self.dtype = device, address, shape, 'bf16'

    def device(self):
        return self._device

    def buffer_address(self):
        return self.address

    def memory_config(self):
        return SimpleNamespace(buffer_type='dram')


class FakeTensor:
    """A two-chip replicated tensor of 2-byte elements; each chip's allocator hands out the next 64 KiB."""

    def __init__(self, operations, shape=(32, 32)):
        self.shards = []
        for chip, device in enumerate(operations.devices):
            self.shards.append(FakeShard(device, operations.next_address[chip], shape))
            operations.next_address[chip] += 0x10000


class FakeOperations:
    Tensor = FakeTensor
    bfloat16 = 'bf16'
    BufferType = SimpleNamespace(DRAM='dram', L1='l1')

    def __init__(self):
        self.devices = [SimpleNamespace(name='chip0'), SimpleNamespace(name='chip1')]
        self.next_address = [0x100000, 0x100000]
        self.views = 0

    def get_device_tensors(self, tensor):
        return tensor.shards

    def get_memory_view(self, device, kind):
        self.views += 1
        return SimpleNamespace(num_banks=8, total_bytes_per_bank=4000, total_bytes_allocated_per_bank=0,
                               total_bytes_free_per_bank=4000, largest_contiguous_bytes_free_per_bank=2000)


def alloc(address, size, kind='DRAM'):
    return dict(node_type='buffer_allocate', params=dict(address=address, size=size, type=kind))


def free(address):
    return dict(node_type='buffer_deallocate', params=dict(address=address))


class CaptureOperations(FakeOperations):
    """A fake ttnn with the trace and graph captures capture_operation and the census call."""

    def __init__(self, nodes=None, graph_refuses=False):
        super().__init__()
        self.calls = []
        self.nodes, self.graph_refuses = nodes or [], graph_refuses

        def begin_graph_capture(mode):
            self.calls.append(('graph_begin', mode))
            if self.graph_refuses:
                raise RuntimeError('graph capture inside a trace capture')

        self.graph = SimpleNamespace(RunMode=SimpleNamespace(NORMAL='normal'), begin_graph_capture=begin_graph_capture,
                                     end_graph_capture=lambda: self.calls.append(('graph_end',)) or self.nodes)

    def begin_trace_capture(self, mesh, cq_id):
        self.calls.append(('begin', cq_id))
        return 'trace'

    def end_trace_capture(self, mesh, trace, cq_id):
        self.calls.append(('end', cq_id))

    def release_trace(self, mesh, trace):
        self.calls.append(('release',))


class InstalledTwin(unittest.TestCase):
    def setUp(self):
        stack = patch.dict(os.environ)
        stack.start()
        self.addCleanup(stack.stop)
        for name in (trace_census.CENSUS_FLAG, trace_census.GRAPH_FLAG, trace_census.STAGE_LOG_FLAG,
                     trace_census.CCL_LOG_FLAG, trace_census.FAIL_CLOSED_FLAG, trace_census.HANDLE_GUARD_FLAG, 'QWEN_FAST_TP'):
            os.environ.pop(name, None)
        trace_census.reset()
        self.addCleanup(trace_census.reset)
        self.lines = []
        logger = patch.object(trace_census, 'log', self.lines.append)
        logger.start()
        self.addCleanup(logger.stop)
        self.addCleanup(tp_addresses.uninstall)


class InstallTests(InstalledTwin):
    def test_four_cards_rebind_every_module_holding_the_capture_function(self):
        os.environ['QWEN_FAST_TP'] = '4'
        tp_addresses.install()
        for module in (attention_batch, verifier_engine, packed_verifier, dflash_proposal_trace):
            with self.subTest(module=module.__name__):
                self.assertIsNot(module.capture_operation, ORIGINAL_CAPTURE)
                self.assertIs(module.capture_operation.census_of, ORIGINAL_CAPTURE)
        # quad_draft imports it at call time, from the rebound attribute
        from attention_batch import capture_operation
        self.assertIs(capture_operation.census_of, ORIGINAL_CAPTURE)
        self.assertIs(verifier_engine.note_packed_step.census_of, ORIGINAL_NOTE)
        self.assertIs(packed_verifier.note_packed_step.census_of, ORIGINAL_NOTE)

    def test_ttnns_release_trace_is_rebound_where_ttnn_is_loaded(self):
        os.environ['QWEN_FAST_TP'] = '4'
        released = lambda mesh, trace: 'released'
        fake = SimpleNamespace(release_trace=released)
        with patch.dict(sys.modules, {'ttnn': fake}):
            tp_addresses.install()
            self.assertIs(fake.release_trace.census_of, released)
            tp_addresses.uninstall()
            self.assertIs(fake.release_trace, released)

    def test_a_second_install_changes_nothing_and_uninstall_puts_the_originals_back(self):
        os.environ['QWEN_FAST_TP'] = '4'
        tp_addresses.install()
        self.assertEqual(trace_census.install(), [])
        tp_addresses.uninstall()
        self.assertIs(attention_batch.capture_operation, ORIGINAL_CAPTURE)
        self.assertIs(verifier_engine.capture_operation, ORIGINAL_CAPTURE)
        self.assertIs(verifier_engine.note_packed_step, ORIGINAL_NOTE)

    def test_the_pair_refuses_to_install_and_nothing_changes(self):
        with self.assertRaises(ValueError):
            tp_addresses.install()
        self.assertIs(attention_batch.capture_operation, ORIGINAL_CAPTURE)
        self.assertIs(verifier_engine.capture_operation, ORIGINAL_CAPTURE)
        self.assertIs(packed_verifier.capture_operation, ORIGINAL_CAPTURE)

    @unittest.skipUnless(os.path.exists(os.path.join(HERE, 'tp2_pinned_sources.json')),
                         'no tp2_pinned_sources.json (inside the image)')
    def test_the_pairs_pinned_attention_batch_is_byte_identical(self):
        with open(os.path.join(HERE, 'tp2_pinned_sources.json'), encoding='utf-8') as handle:
            pins = json.load(handle)
        pinned = next(group['attention_batch.py'] for group in pins.values()
                      if isinstance(group, dict) and 'attention_batch.py' in group)['pinned']
        with open(os.path.join(HERE, 'attention_batch.py'), 'rb') as handle:
            self.assertEqual(hashlib.sha256(handle.read().replace(b'\r\n', b'\n')).hexdigest(), pinned)

    def test_the_packed_step_counter_counts_and_still_calls_the_original(self):
        calls = []
        twin = trace_census.count_packed_step(lambda: calls.append(1) or 'result')
        self.assertEqual((twin(), twin()), ('result', 'result'))
        self.assertEqual((trace_census.PACKED_NOTES, calls), (2, [1, 1]))


class CaptureTests(InstalledTwin):
    def twin(self):
        return trace_census.census_capture(ORIGINAL_CAPTURE)

    def test_without_the_flag_the_twin_is_the_originals_call_for_call(self):
        outputs = []
        for capture in (ORIGINAL_CAPTURE, self.twin()):
            operations = CaptureOperations()
            result = capture(operations, 'mesh', lambda: 'output')
            outputs.append((result, operations.calls))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], (('trace', 'output'), [('begin', 0), ('end', 0)]))
        self.assertEqual(self.lines, [])

    def test_with_the_flag_a_capture_logs_its_site_views_and_freed_temporaries(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.GRAPH_FLAG] = '1'
        operations = CaptureOperations(nodes=[alloc(0x1000, 256), alloc(0x2000, 512), free(0x1000),
                                              alloc(0x3000, 64, 'L1'), free(0x3000)])
        mesh = SimpleNamespace(get_devices=lambda: operations.devices)
        trace, output = self.twin()(operations, mesh, lambda: 'output')
        self.assertEqual((trace, output), ('trace', 'output'))
        self.assertEqual(operations.calls, [('graph_begin', 'normal'), ('begin', 0), ('end', 0), ('graph_end',)])
        self.assertRegex(self.lines[0], r'^\[PINDIAG\] trace census seq=1 site=test_trace_census\.test_with_the_flag.* '
                                        r'temporaries=2 bytes_per_bank=96$')
        views = [line for line in self.lines if ' chip' in line]
        self.assertEqual(len(views), 2)
        self.assertIn('dram allocated=0.0->0.0 largest_free=', views[0])
        self.assertIn(' l1 allocated=', views[0])
        # per-bank extents: the 256-byte DRAM buffer over the view's 8 banks, the L1 buffer's own size
        self.assertEqual([(lo, hi, kind) for lo, hi, kind in trace_census.TRACES[0]['ranges']],
                         [(0x1000, 0x1020, 'DRAM'), (0x3000, 0x3040, 'L1')])
        self.assertEqual(trace_census.TRACES[0]['handle'], 'trace')

    def test_a_graph_capture_that_raises_logs_once_and_the_capture_still_succeeds(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.GRAPH_FLAG] = '1'
        operations = CaptureOperations(graph_refuses=True)
        for _ in range(2):
            self.assertEqual(self.twin()(operations, SimpleNamespace(), lambda: 'output'), ('trace', 'output'))
        unavailable = [line for line in self.lines if 'trace census unavailable' in line]
        self.assertEqual(len(unavailable), 1)
        self.assertIn('graph capture RuntimeError', unavailable[0])
        self.assertNotIn(('graph_end',), operations.calls)
        self.assertEqual(trace_census.TRACES, [])

    def test_the_graph_part_is_off_by_default_and_only_exactly_1_turns_it_on(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        for value in (None, '0', 'true', '', '2'):
            if value is None:
                os.environ.pop(trace_census.GRAPH_FLAG, None)
            else:
                os.environ[trace_census.GRAPH_FLAG] = value
            operations = CaptureOperations()
            self.twin()(operations, SimpleNamespace(), lambda: 'output')
            self.assertEqual(operations.calls, [('begin', 0), ('end', 0)], value)
            self.assertFalse(trace_census.graph_census_enabled(), value)
        self.assertEqual(trace_census.TRACES, [], 'no graph, no ranges: a trace is recorded only with its freed extents')
        os.environ[trace_census.GRAPH_FLAG] = '1'
        self.assertTrue(trace_census.graph_census_enabled())
        operations = CaptureOperations()
        self.twin()(operations, SimpleNamespace(), lambda: 'output')
        self.assertEqual(operations.calls, [('graph_begin', 'normal'), ('begin', 0), ('end', 0), ('graph_end',)])

    def test_an_operation_that_raises_ends_the_graph_capture_and_propagates(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.GRAPH_FLAG] = '1'
        operations = CaptureOperations()

        def refuse():
            raise ValueError('operation refused')

        with self.assertRaisesRegex(ValueError, 'operation refused'):
            self.twin()(operations, SimpleNamespace(), refuse)
        self.assertEqual(operations.calls[-2:], [('release',), ('graph_end',)])

    def test_the_collective_state_is_logged_before_and_after(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        collectives = SimpleNamespace(ag_idx=2, barrier_idx=0)
        trace_census.register_collectives(collectives)
        operations = CaptureOperations()
        self.twin()(operations, SimpleNamespace(), lambda: collectives.__setattr__('ag_idx', 5) or 'output')
        self.assertIn('[PINDIAG] trace census seq=1 ccl before{shared{ag_idx=2 barrier_idx=0}} '
                      'after{shared{ag_idx=5 barrier_idx=0}}', self.lines)

    def test_the_capture_line_carries_the_shared_model_and_sampler_collectives(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        model, sampler = SimpleNamespace(ag=1), SimpleNamespace(gather=7)
        trace_census.note_collectives(SimpleNamespace(ag=0), SimpleNamespace(tt_ccl=model),
                                      SimpleNamespace(tt_sampling=SimpleNamespace(tt_ccl=sampler)))
        operations = CaptureOperations()
        self.twin()(operations, SimpleNamespace(), lambda: model.__setattr__('ag', 4) or 'output')
        self.assertIn('[PINDIAG] trace census seq=1 ccl before{shared{ag=0} model{ag=1} sampler{gather=7}} '
                      'after{shared{ag=0} model{ag=4} sampler{gather=7}}', self.lines)

    def test_a_capture_line_too_long_for_the_log_goes_one_object_per_line(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        wide = lambda: SimpleNamespace(**{'index_%d' % index: index for index in range(8)})
        for name in ('shared', 'model', 'sampler'):
            trace_census.register_collectives(wide(), name)
        self.twin()(CaptureOperations(), SimpleNamespace(), lambda: 'output')
        lines = [line for line in self.lines if ' ccl' in line]
        self.assertEqual([line.split(' ccl ')[1].split(' ')[0] for line in lines], ['shared', 'model', 'sampler'])
        self.assertTrue(all(len(line) <= trace_census.LINE_BUDGET and 'index_7=7' in line for line in lines), lines)

    def test_note_collectives_registers_nothing_without_a_flag_and_each_object_once(self):
        shared = SimpleNamespace(ag=0)
        trace_census.note_collectives(shared, SimpleNamespace(tt_ccl=shared), None)
        self.assertEqual(trace_census.COLLECTIVES, {})
        os.environ[trace_census.CCL_LOG_FLAG] = '1'
        trace_census.note_collectives(shared, SimpleNamespace(tt_ccl=shared), SimpleNamespace())
        self.assertEqual(list(trace_census.COLLECTIVES), ['shared'])

    def test_releasing_a_trace_forgets_its_temporaries_and_only_its_own(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.GRAPH_FLAG] = '1'
        operations = CaptureOperations(nodes=[alloc(0x1000, 256), free(0x1000)])
        for handle in ('first', 'second'):
            operations.begin_trace_capture = lambda mesh, cq_id, handle=handle: handle
            self.twin()(operations, SimpleNamespace(), lambda: 'output')
        self.assertEqual([trace['handle'] for trace in trace_census.TRACES], ['first', 'second'])
        released = []
        trace_census.census_release(lambda mesh, trace: released.append(trace) or 'done')('mesh', 'first')
        self.assertEqual(([trace['handle'] for trace in trace_census.TRACES], released), (['second'], ['first']))


class FreedRangeTests(unittest.TestCase):
    def test_a_buffers_extent_is_per_bank(self):
        nodes = [alloc(0x100, 0x800), free(0x100)]
        self.assertEqual(trace_census.freed_ranges(nodes, banks=8), [(0x100, 0x200, 'DRAM')])
        self.assertEqual(trace_census.freed_ranges(nodes, banks=3), [(0x100, 0x100 + 683, 'DRAM')])
        nodes[1]['params']['max_size_per_bank'] = 0x40
        self.assertEqual(trace_census.freed_ranges(nodes, banks=8), [(0x100, 0x140, 'DRAM')], 'the node says so itself')

    def test_the_same_address_in_two_buffer_types_is_two_buffers(self):
        nodes = [alloc(0x100, 64, 'DRAM'), alloc(0x100, 32, 'L1'), dict(node_type='buffer_deallocate', params=dict(
            address=0x100, type='DRAM')), dict(node_type='buffer_deallocate', params=dict(address=0x100, type='L1'))]
        self.assertEqual(trace_census.freed_ranges(nodes), [(0x100, 0x140, 'DRAM'), (0x100, 0x120, 'L1')])

    def test_only_buffers_allocated_and_freed_inside_the_capture_are_temporaries(self):
        nodes = [alloc(0x100, 16), alloc('0x200', 32), free(0x100), free(0x999), dict(node_type='function_start')]
        self.assertEqual(trace_census.freed_ranges(nodes), [(0x100, 0x110, 'DRAM')])
        self.assertEqual(trace_census.freed_ranges(json.dumps(nodes)), [(0x100, 0x110, 'DRAM')])

    def test_an_address_reused_inside_the_capture_is_two_temporaries(self):
        nodes = [alloc(0x100, 16), free(0x100), alloc(0x100, 64), free(0x100)]
        self.assertEqual(trace_census.freed_ranges(nodes), [(0x100, 0x110, 'DRAM'), (0x100, 0x140, 'DRAM')])


class EngineCensusTests(InstalledTwin):
    def request(self, operations):
        resident = FakeTensor(operations, (32, 32))          # 0x100000 on both chips
        later = FakeTensor(operations, (32, 32))             # 0x110000
        engine = SimpleNamespace(buckets={'k': dict(rows=4, fixture=SimpleNamespace(resident=resident),
                                                    output=(later, None))})
        return SimpleNamespace(engine=engine)

    def test_a_buffer_inside_an_earlier_traces_freed_temporaries_is_reported_per_chip_and_fails_closed(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        trace_census.ENGINE_BOUNDARY = 5
        trace_census.TRACES.append(dict(seq=3, site='packed_verifier.capture:909', ranges=[(0x100000, 0x108000, 'DRAM')]))
        with self.assertRaisesRegex(trace_census.TraceOverlap, '2 buffer'):
            trace_census.census_engine('request-E', self.request(operations), operations)
        overlaps = [line for line in self.lines if 'trace overlap' in line]
        self.assertEqual(overlaps, [
            '[PINDIAG] trace overlap request=request-E rows=4 buffer=DRAM chip0@0x100000+2048/bank trace=packed_verifier.capture:909',
            '[PINDIAG] trace overlap request=request-E rows=4 buffer=DRAM chip1@0x100000+2048/bank trace=packed_verifier.capture:909'])
        extents = [line for line in self.lines if 'engine buffers' in line]
        self.assertEqual(extents[0], '[PINDIAG] engine buffers request=request-E rows=4 DRAM chip0 buffers=2 lo=0x100000 hi=0x110800')
        self.assertIn('overlaps=2', self.lines[-1])

    def test_the_fail_closed_flag_turns_the_raise_off_and_the_log_stays(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.FAIL_CLOSED_FLAG] = '0'
        operations = FakeOperations()
        trace_census.ENGINE_BOUNDARY = 5
        trace_census.TRACES.append(dict(seq=3, site='s', ranges=[(0x100000, 0x108000, 'DRAM')]))
        trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertIn('overlaps=2', self.lines[-1])

    def test_with_no_recorded_ranges_the_engine_is_not_walked_at_all(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        trace_census.TRACES.append(dict(seq=1, site='graph off', ranges=[], handle='t'))
        trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertEqual(self.lines, ['[PINDIAG] trace census engine request=request-E traces=0 overlaps=0 skipped=no ranges recorded'])
        self.assertEqual(operations.views, 0, 'no allocator read and no tensor walked: the v174 timeout')

    def test_a_host_tensor_is_never_asked_for_its_address(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        operations.StorageType = SimpleNamespace(DEVICE='device', HOST='host')
        request = self.request(operations)

        def refuse():
            raise RuntimeError('StorageType::HOST does not support buffer_address')

        for tensor in (request.engine.buckets['k']['fixture'].resident, request.engine.buckets['k']['output'][0]):
            for shard in tensor.shards:
                shard.storage_type = lambda: 'host'
                shard.buffer_address = refuse
        trace_census.ENGINE_BOUNDARY = 5
        trace_census.TRACES.append(dict(seq=3, site='s', ranges=[(0x100000, 0x108000, 'DRAM')]))
        trace_census.census_engine('request-E', request, operations)
        self.assertEqual([line for line in self.lines if 'overlap' in line and 'overlaps=' not in line], [])
        self.assertIn('overlaps=0', self.lines[-1])
        self.assertFalse(trace_census.on_device(operations, request.engine.buckets['k']['output'][0].shards[0]))
        self.assertTrue(trace_census.on_device(operations, FakeShard(None, 0, (32, 32))), 'a shard that cannot say is read, guarded')

    def test_an_l1_buffer_is_held_against_the_l1_write_set_and_not_the_drams(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        request = self.request(operations)
        for tensor in (request.engine.buckets['k']['fixture'].resident, request.engine.buckets['k']['output'][0]):
            for shard in tensor.shards:
                shard.memory_config = lambda: SimpleNamespace(buffer_type='l1')
        trace_census.ENGINE_BOUNDARY = 5
        trace_census.TRACES.append(dict(seq=3, site='dram only', ranges=[(0x100000, 0x120000, 'DRAM')]))
        trace_census.census_engine('request-E', request, operations)
        self.assertIn('overlaps=0', self.lines[-1])
        trace_census.TRACES.append(dict(seq=4, site='l1', ranges=[(0x100000, 0x100800, 'L1')]))
        with self.assertRaises(trace_census.TraceOverlap):
            trace_census.census_engine('request-E', request, operations)
        self.assertTrue(any('buffer=L1 chip0@0x100000' in line for line in self.lines))

    def test_disjoint_ranges_and_the_engines_own_traces_report_nothing(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        trace_census.ENGINE_BOUNDARY = 5
        trace_census.TRACES.extend([
            dict(seq=3, site='elsewhere', ranges=[(0x200000, 0x300000, 'DRAM'), (0x100000, 0x108000, 'L1')]),
            dict(seq=6, site='this engine', ranges=[(0x100000, 0x108000, 'DRAM')])])
        trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertEqual([line for line in self.lines if 'trace overlap' in line], [])
        self.assertIn('overlaps=0', self.lines[-1])

    def captured(self, nodes):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.GRAPH_FLAG] = '1'
        operations = CaptureOperations(nodes=nodes)
        mesh = SimpleNamespace(get_devices=lambda: operations.devices)
        trace_census.census_capture(ORIGINAL_CAPTURE)(operations, mesh, lambda: 'output')
        return operations

    def test_a_buffer_beside_a_temporary_of_an_eight_bank_buffer_is_not_inside_it(self):
        # 0x80000 bytes over 8 banks is 0x10000 per bank: [0xF0000, 0x100000), which ends where the resident buffer begins.
        # Read as a total length the range would run to 0x170000 and swallow both of the engine's buffers.
        operations = FakeOperations()
        self.captured([alloc(0xF0000, 0x80000), free(0xF0000)])
        trace_census.ENGINE_BOUNDARY = 5
        trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertEqual([line for line in self.lines if 'trace overlap' in line], [])
        self.assertIn('overlaps=0', self.lines[-1])

    def test_a_buffer_inside_a_live_traces_temporaries_overlaps_and_inside_a_released_ones_does_not(self):
        operations = FakeOperations()
        # per bank [0x100000, 0x110000): the resident buffer is inside it, the later one (0x110000) is not
        ops = self.captured([alloc(0x100000, 0x80000), free(0x100000)])
        trace_census.ENGINE_BOUNDARY = 5
        with self.assertRaises(trace_census.TraceOverlap):
            trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertEqual([line.split(' buffer=')[1].split(' trace=')[0] for line in self.lines if 'trace overlap' in line],
                         ['DRAM chip0@0x100000+2048/bank', 'DRAM chip1@0x100000+2048/bank'])
        del self.lines[:]
        trace_census.census_release(ops.release_trace)('mesh', 'trace')
        trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertEqual([line for line in self.lines if 'trace overlap' in line], [])
        self.assertIn('traces=0 overlaps=0', self.lines[-1])

    def test_an_interleaved_buffers_extent_is_its_pages_over_the_banks(self):
        operations = FakeOperations()
        # the fake shard is 32 x 32 bf16: one 2 KiB page
        import memory_ledger
        ledger = memory_ledger.MemoryLedger(operations, None, log=lambda message: None, emit=lambda text: None)
        small, wide = FakeShard(None, 0, (32, 32)), FakeShard(None, 0, (32, 32 * 20))
        self.assertEqual(trace_census.bank_extent(ledger, small, 8), 2048)
        self.assertEqual(trace_census.bank_extent(ledger, wide, 8), 3 * 2048, '20 pages over 8 banks: 3 pages in a bank')

    def test_without_the_flag_nothing_is_read_or_logged(self):
        operations = FakeOperations()
        trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertEqual((self.lines, operations.views), ([], 0))

    def test_an_engine_that_cannot_be_walked_logs_once_and_never_raises(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        request = SimpleNamespace(engine=SimpleNamespace(buckets=5))
        trace_census.TRACES.append(dict(seq=0, site='s', ranges=[(0, 1, 'DRAM')]))
        trace_census.census_engine('request-E', request, operations)
        trace_census.census_engine('request-E', request, operations)
        self.assertEqual(len([line for line in self.lines if 'unavailable' in line]), 1)

    def test_the_stage_flag_records_when_the_engine_was_built_for_the_first_replay_line(self):
        os.environ[trace_census.STAGE_LOG_FLAG] = '1'
        operations = FakeOperations()
        trace_census.PACKED_NOTES = 7
        trace_census.engine_begin()
        trace_census.PACKED_NOTES = 10
        request = self.request(operations)
        trace_census.census_engine('request-E', request, operations)
        trace_census.first_replay(request.engine, 'request-E', 4)
        self.assertEqual(self.lines, ['[PINDIAG] first replay request=request-E rows=4 packed_notes_at_build=7 '
                                      'packed_notes_since_build=3 program_cache=n/a'])


class WatchTests(InstalledTwin):
    def test_the_flags_are_read_exactly(self):
        for value, enabled in (('1', True), ('0', False), ('true', False)):
            self.assertEqual(trace_census.flag_on(trace_census.CENSUS_FLAG, {trace_census.CENSUS_FLAG: value}), enabled)
        self.assertEqual(trace_census.seq_deadline({trace_census.SEQ_DEADLINE_FLAG: '0.5'}), 0.5)
        self.assertIsNone(trace_census.seq_deadline({}))
        self.assertFalse(trace_census.watching({}))
        self.assertTrue(trace_census.watching({trace_census.SEQ_DEADLINE_FLAG: '120'}))

    def test_a_faulthandler_that_cannot_write_to_stderr_leaves_the_step_unwatched(self):
        os.environ[trace_census.SEQ_DEADLINE_FLAG] = '3'
        with patch('faulthandler.dump_traceback_later', side_effect=OSError('no fileno')), \
                patch('faulthandler.cancel_dump_traceback_later') as cancel:
            self.assertEqual(trace_census.watched_step('r', 1, lambda: 'done'), 'done')
        cancel.assert_not_called()
        self.assertIn('[PINDIAG] seq watchdog unavailable OSError: no fileno', self.lines)


class WriteSetTests(unittest.TestCase):
    def test_a_buffer_allocated_before_the_capture_and_freed_inside_it_joins_the_write_set_when_its_size_is_known(self):
        sized = dict(node_type='buffer_deallocate', params=dict(address=0x500, type='DRAM', size=0x400))
        per_bank = dict(node_type='buffer_deallocate', params=dict(address=0x900, type='L1', max_size_per_bank=0x20))
        unsized = free(0x999)
        self.assertEqual(trace_census.freed_ranges([sized, per_bank, unsized], banks=8),
                         [(0x500, 0x580, 'DRAM'), (0x900, 0x920, 'L1')], 'an address alone says nothing and adds nothing')

    def test_a_range_of_unnamed_type_is_held_against_both_kinds(self):
        traces = [dict(ranges=[(0x100, 0x200, '?')], site='s')]
        self.assertIs(trace_census.overlapping(0x180, 0x190, traces, 'DRAM'), traces[0])
        self.assertIs(trace_census.overlapping(0x180, 0x190, traces, 'L1'), traces[0])
        self.assertIsNone(trace_census.overlapping(0x200, 0x210, traces, 'DRAM'))


class EagerAllocationTests(InstalledTwin):
    def live(self):
        trace_census.TRACES.append(dict(seq=1, site='packed_verifier.capture:909',
                                        ranges=[(0x1000, 0x2000, 'DRAM'), (0x40, 0x80, 'L1')], handle='t'))

    def test_no_live_ranges_means_nothing_is_parsed_or_raised(self):
        self.assertEqual(trace_census.note_eager_allocations([alloc(0x1000, 64)], 'prefill'), 0)
        self.assertEqual(self.lines, [])

    def test_an_allocation_in_a_live_traces_write_set_is_logged_and_fails_closed(self):
        self.live()
        nodes = [alloc(0x1800, 0x100), alloc(0x3000, 0x100), alloc(0x50, 8, 'L1'), alloc(0x50, 8, 'DRAM')]
        with self.assertRaisesRegex(trace_census.TraceOverlap, '2 allocation'):
            trace_census.note_eager_allocations(nodes, 'window snapshot', banks=1)
        self.assertEqual([line for line in self.lines if 'trace overlap' in line], [
            '[PINDIAG] trace overlap site=window snapshot allocation=DRAM@0x1800+256/bank trace=packed_verifier.capture:909',
            '[PINDIAG] trace overlap site=window snapshot allocation=L1@0x50+8/bank trace=packed_verifier.capture:909'])
        self.assertIn('allocations=4 overlaps=2', self.lines[-1])

    def test_log_only_when_fail_closed_is_off_and_clean_work_returns_zero(self):
        self.live()
        os.environ[trace_census.FAIL_CLOSED_FLAG] = '0'
        self.assertEqual(trace_census.note_eager_allocations([alloc(0x1800, 8)], 'x', banks=1), 1)
        self.assertEqual(trace_census.note_eager_allocations([alloc(0x5000, 8)], 'x', banks=1), 0)

    def test_eager_graph_is_a_noop_without_both_flags_and_checks_the_work_with_them(self):
        operations = CaptureOperations(nodes=[alloc(0x1800, 0x100)])
        with trace_census.eager_graph(operations, 'prefill'):
            pass
        os.environ[trace_census.CENSUS_FLAG] = '1'
        with trace_census.eager_graph(operations, 'prefill'):
            pass
        self.assertEqual(operations.calls, [], 'the graph flag is off by default: no graph capture around eager work either')
        os.environ[trace_census.GRAPH_FLAG] = '1'
        self.live()
        with self.assertRaises(trace_census.TraceOverlap):
            with trace_census.eager_graph(operations, 'prefill'):
                operations.calls.append(('work',))
        self.assertEqual(operations.calls, [('graph_begin', 'normal'), ('work',), ('graph_end',)])


class EagerSegmentTests(InstalledTwin):
    def test_without_flags_the_segment_scope_does_nothing_and_with_the_stall_flag_it_is_a_prefill_scope(self):
        import stall_watch
        with trace_census.eager_segment('prefill segment prompt=9'):
            pass
        seen = []

        class Scope:
            def __enter__(self):
                seen.append('enter')

            def __exit__(self, *exc):
                seen.append('exit')
                return False

        with patch.object(stall_watch, 'scope', lambda kind, label: seen.append((kind, label)) or Scope()):
            with trace_census.eager_segment('prefill segment prompt=9'):
                seen.append('work')
        self.assertEqual(seen, [('prefill', 'prefill segment prompt=9'), 'enter', 'work', 'exit'])

    def test_eager_graph_resolves_ttnn_itself_when_given_none(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.GRAPH_FLAG] = '1'
        operations = CaptureOperations(nodes=[])
        with patch.dict(sys.modules, {'ttnn': operations}):
            with trace_census.eager_graph(None, 'prefill'):
                pass
        self.assertEqual(operations.calls, [('graph_begin', 'normal'), ('graph_end',)])


class Collectives:
    """The shape of TT_CCL: two handles per getter, cycled by a host index that a replay does not advance."""

    def __init__(self):
        self.ag = [object(), object()]
        self.barrier = [object(), object()]
        self.index = 0

    def get_and_cycle_ag_semaphore_handles(self):
        handle = self.ag[self.index % 2]
        self.index += 1
        return [handle]

    def get_and_cycle_barrier_semaphore_handle(self, axis=0):
        return self.barrier[self.index % 2]


class HandleGuardTests(InstalledTwin):
    def setUp(self):
        super().setUp()
        os.environ[trace_census.HANDLE_GUARD_FLAG] = '1'
        os.environ[trace_census.CENSUS_FLAG] = '1'
        self.ccl = Collectives()
        trace_census.note_collectives(self.ccl)
        self.assertEqual(len(trace_census.TAPPED), 1)

    def capture(self, trace, *calls):
        operations = CaptureOperations()
        operations.begin_trace_capture = lambda mesh, cq_id: trace

        def work():
            for call in calls:
                call()
            return 'output'

        trace_census.census_capture(ORIGINAL_CAPTURE)(operations, SimpleNamespace(), work)

    def ag(self):
        return self.ccl.get_and_cycle_ag_semaphore_handles()

    def test_the_mode_is_read_exactly(self):
        for value, mode in (('1', 'fail'), ('log', 'log'), ('0', None), ('', None)):
            self.assertEqual(trace_census.handle_guard_mode({trace_census.HANDLE_GUARD_FLAG: value}), mode)
        self.assertIsNone(trace_census.handle_guard_mode({}))
        with self.assertRaises(ValueError):
            trace_census.handle_guard_mode({trace_census.HANDLE_GUARD_FLAG: 'yes'})

    def test_the_guard_alone_records_the_handles_without_the_census_or_any_allocator_read(self):
        del os.environ[trace_census.CENSUS_FLAG]
        operations = CaptureOperations()
        operations.begin_trace_capture = lambda mesh, cq_id: 'solo'
        trace_census.census_capture(ORIGINAL_CAPTURE)(operations, SimpleNamespace(get_devices=lambda: operations.devices),
                                                      lambda: self.ag() and 'output')
        self.assertIn('solo', trace_census.TRACE_SLOTS)
        self.assertEqual(operations.views, 0)
        self.assertEqual([line for line in self.lines if 'census seq' in line], [])
        self.assertEqual(len([line for line in self.lines if 'trace handles' in line]), 1)

    def test_a_capture_records_the_first_and_last_handle_per_slot_and_logs_them(self):
        self.capture('t1', self.ag, self.ag, self.ag)
        slots = trace_census.TRACE_SLOTS['t1']
        (slot, (first, last)), = slots.items()
        self.assertEqual(slot[:2], ('shared', 'get_and_cycle_ag_semaphore_handles'))
        self.assertEqual(first, (id(self.ccl.ag[0]),))
        self.assertEqual(last, (id(self.ccl.ag[0]),), 'three calls: handle 0, 1, 0')
        lines = [line for line in self.lines if 'trace handles' in line]
        self.assertEqual(len(lines), 1)
        self.assertIn('slot=shared.ag_semaphore_handles first=', lines[0])
        self.assertEqual(trace_census.PENDING, {}, 'a capture enqueues nothing')

    def test_a_replay_that_starts_on_the_handle_the_previous_replay_ended_on_is_refused_before_it_is_enqueued(self):
        self.capture('a', self.ag)          # handle 0
        self.ccl.index = 0
        self.capture('b', self.ag)          # handle 0 again: the same slot, the same handle
        ran = []
        execute = trace_census.census_execute(lambda mesh, trace, cq_id=0, blocking=True: ran.append(trace))
        execute('mesh', 'a', cq_id=0, blocking=False)
        with self.assertRaises(trace_census.UnfencedHandleReuse):
            execute('mesh', 'b', cq_id=0, blocking=False)
        self.assertEqual(ran, ['a'], 'the second replay was never enqueued')
        self.assertEqual(len(trace_census.VIOLATIONS), 1)

    def test_a_fence_between_the_replays_makes_the_same_handles_safe(self):
        self.capture('a', self.ag)
        self.ccl.index = 0
        self.capture('b', self.ag)
        ran = []
        execute = trace_census.census_execute(lambda mesh, trace, cq_id=0, blocking=True: ran.append(trace))
        synchronize = trace_census.census_synchronize(lambda mesh: None)
        execute('mesh', 'a', cq_id=0, blocking=False)
        synchronize('mesh')
        execute('mesh', 'b', cq_id=0, blocking=False)
        self.assertEqual((ran, trace_census.VIOLATIONS), (['a', 'b'], []))

    def test_a_blocking_replay_is_a_fence_for_the_one_after_it(self):
        self.capture('a', self.ag)
        self.ccl.index = 0
        self.capture('b', self.ag)
        ran = []
        execute = trace_census.census_execute(lambda mesh, trace, cq_id=0, blocking=True: ran.append(trace))
        execute('mesh', 'a', cq_id=0, blocking=True)
        execute('mesh', 'b', cq_id=0, blocking=True)
        self.assertEqual((ran, trace_census.VIOLATIONS, trace_census.PENDING), (['a', 'b'], [], {}))

    def test_an_eager_collective_after_a_non_blocking_replay_on_the_same_handle_is_refused(self):
        self.capture('a', self.ag)          # ends on handle 0; the index is now 1
        self.ccl.index = 0                  # the eager index happens to be back on 0
        execute = trace_census.census_execute(lambda mesh, trace, cq_id=0, blocking=True: None)
        execute('mesh', 'a', cq_id=0, blocking=False)
        with self.assertRaisesRegex(trace_census.UnfencedHandleReuse, 'eager collective'):
            self.ag()
        trace_census.census_synchronize(lambda mesh: None)('mesh')
        self.ccl.index = 0
        self.ag()                           # after the fence it is fine

    def test_eager_collectives_alternate_by_construction_and_never_trip_it(self):
        for _ in range(10):
            self.ag()
            self.ccl.get_and_cycle_barrier_semaphore_handle()
        self.assertEqual(trace_census.VIOLATIONS, [])

    def test_log_mode_logs_and_lets_the_replay_through(self):
        os.environ[trace_census.HANDLE_GUARD_FLAG] = 'log'
        self.capture('a', self.ag)
        self.ccl.index = 0
        self.capture('b', self.ag)
        ran = []
        execute = trace_census.census_execute(lambda mesh, trace, cq_id=0, blocking=True: ran.append(trace))
        execute('mesh', 'a', cq_id=0, blocking=False)
        execute('mesh', 'b', cq_id=0, blocking=False)
        self.assertEqual(ran, ['a', 'b'])
        self.assertTrue(any('unfenced same-handle collective: a replay starts' in line for line in self.lines))

    def test_off_the_twins_are_the_originals_call_and_install_binds_them_only_with_the_guard(self):
        del os.environ[trace_census.HANDLE_GUARD_FLAG]
        ran = []
        execute = trace_census.census_execute(lambda *args, **kwargs: ran.append(args) or 'r')
        self.assertEqual(execute('mesh', 't', cq_id=0, blocking=False), 'r')
        os.environ['QWEN_FAST_TP'] = '4'

        def fake():
            return SimpleNamespace(release_trace=lambda mesh, trace: None, execute_trace=lambda *a, **k: None,
                                   synchronize_device=lambda mesh: None)

        unguarded = fake()
        # Inside the image the model tree is importable, and tile_collective_tp.install would import its ccl module against
        # this bare fake ttnn (no MeshDevice): this test is about the census twins only, so the tile split is held out.
        import tile_collective_tp
        no_tile_split = patch.object(tile_collective_tp, 'install', return_value=[])
        with patch.dict(sys.modules, {'ttnn': unguarded}), no_tile_split:
            tp_addresses.install()
            self.assertTrue(hasattr(unguarded.release_trace, 'census_of'))
            self.assertFalse(hasattr(unguarded.execute_trace, 'census_of'), 'the hot-path twins stay unbound without the guard')
            self.assertFalse(hasattr(unguarded.synchronize_device, 'census_of'))
            tp_addresses.uninstall()
        os.environ[trace_census.HANDLE_GUARD_FLAG] = '1'
        guarded = fake()
        with patch.dict(sys.modules, {'ttnn': guarded}), patch.object(tile_collective_tp, 'install', return_value=[]):
            tp_addresses.install()
            self.assertTrue(hasattr(guarded.execute_trace, 'census_of') and hasattr(guarded.synchronize_device, 'census_of'))
            tp_addresses.uninstall()

    def test_releasing_a_trace_forgets_its_slots(self):
        self.capture('a', self.ag)
        trace_census.census_release(lambda mesh, trace: None)('mesh', 'a')
        self.assertNotIn('a', trace_census.TRACE_SLOTS)

    def test_the_slots_are_keyed_by_the_traces_integer_value_when_it_has_one(self):
        class TraceId:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return self.value

            def __eq__(self, other):
                return isinstance(other, TraceId) and other.value == self.value

            __hash__ = None          # defines __eq__ without __hash__: unhashable

        self.capture(TraceId(7), self.ag)
        self.assertEqual(list(trace_census.TRACE_SLOTS), [7], 'an unhashable wrapper does not raise TypeError at attach')
        self.assertEqual(trace_census.trace_key(TraceId(7)), 7)
        self.ccl.index = 0
        ran = []
        original = lambda *args, **kwargs: ran.append(args)
        replay = trace_census.census_execute(original)
        with self.assertRaises(trace_census.UnfencedHandleReuse):
            self.capture(TraceId(8), self.ag)
            replay('mesh', TraceId(7), 0, False)
            replay('mesh', TraceId(8), 0, False)
        trace_census.census_release(lambda mesh, trace: None)('mesh', TraceId(7))
        self.assertNotIn(7, trace_census.TRACE_SLOTS)

    def test_a_handle_with_neither_an_integer_value_nor_a_hash_is_keyed_by_identity(self):
        class Opaque:
            __hash__ = None

            def __eq__(self, other):
                return self is other

        handle = Opaque()
        self.assertEqual(trace_census.trace_key(handle), id(handle))
        self.assertEqual(trace_census.trace_key('t1'), 't1')
        self.assertEqual(trace_census.trace_key(5), 5)


class StallWiringTests(InstalledTwin):
    def test_with_the_stall_flag_a_sequential_step_is_a_watched_scope_and_faulthandlers_exit_is_not_armed(self):
        import stall_watch
        os.environ[stall_watch.DEADLINE_FLAG] = '120'
        os.environ[trace_census.SEQ_DEADLINE_FLAG] = '120'
        seen = []

        class Scope:
            def __enter__(self):
                seen.append('enter')

            def __exit__(self, *exc):
                seen.append('exit')
                return False

        with patch.object(stall_watch, 'scope', lambda kind, label: seen.append((kind, label)) or Scope()), \
                patch('faulthandler.dump_traceback_later') as dump:
            self.assertEqual(trace_census.watched_step('request-1', 4, lambda: 'done'), 'done')
        dump.assert_not_called()
        self.assertEqual(seen, [('step', 'sequential request=request-1 rows=4'), 'enter', 'exit'])
        self.assertTrue(trace_census.watching({stall_watch.DEADLINE_FLAG: '5'}))


if __name__ == '__main__':
    unittest.main()
