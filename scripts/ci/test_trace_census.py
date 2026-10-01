"""trace_census: the sequential-hang diagnostics. Flag-gated, host-only; the capture twin is bound at four cards only and
returns the original's result, with the original's calls, unless QWEN_FAST_TRACE_CENSUS=1."""

import hashlib
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import attention_batch
import dflash_proposal_trace
import packed_verifier
import tp_addresses
import trace_census
import verifier_engine
from test_memory_ledger import FakeOperations, FakeTensor

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_CAPTURE = attention_batch.capture_operation
ORIGINAL_NOTE = verifier_engine.note_packed_step


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
                     trace_census.CCL_LOG_FLAG, 'QWEN_FAST_TP'):
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
        self.assertEqual((trace_census.PACKED_STEPS, calls), (2, [1, 1]))


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
        operations = CaptureOperations(nodes=[alloc(0x1000, 256), alloc(0x2000, 512), free(0x1000),
                                              alloc(0x3000, 64, 'L1'), free(0x3000)])
        mesh = SimpleNamespace(get_devices=lambda: operations.devices)
        trace, output = self.twin()(operations, mesh, lambda: 'output')
        self.assertEqual((trace, output), ('trace', 'output'))
        self.assertEqual(operations.calls, [('graph_begin', 'normal'), ('begin', 0), ('end', 0), ('graph_end',)])
        self.assertRegex(self.lines[0], r'^\[PINDIAG\] trace census seq=1 site=test_trace_census\.test_with_the_flag.* '
                                        r'temporaries=2 bytes=320$')
        views = [line for line in self.lines if ' chip' in line]
        self.assertEqual(len(views), 2)
        self.assertIn('dram allocated=0.0->0.0 largest_free=', views[0])
        self.assertIn(' l1 allocated=', views[0])
        self.assertEqual([(lo, hi, kind) for lo, hi, kind in trace_census.TRACES[0]['ranges']],
                         [(0x1000, 0x1100, 'DRAM'), (0x3000, 0x3040, 'L1')])

    def test_a_graph_capture_that_raises_logs_once_and_the_capture_still_succeeds(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = CaptureOperations(graph_refuses=True)
        for _ in range(2):
            self.assertEqual(self.twin()(operations, SimpleNamespace(), lambda: 'output'), ('trace', 'output'))
        unavailable = [line for line in self.lines if 'trace census unavailable' in line]
        self.assertEqual(len(unavailable), 1)
        self.assertIn('graph capture RuntimeError', unavailable[0])
        self.assertNotIn(('graph_end',), operations.calls)
        self.assertEqual(trace_census.TRACES, [])

    def test_the_graph_part_can_be_switched_off_on_its_own(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        os.environ[trace_census.GRAPH_FLAG] = '0'
        operations = CaptureOperations()
        self.twin()(operations, SimpleNamespace(), lambda: 'output')
        self.assertEqual(operations.calls, [('begin', 0), ('end', 0)])

    def test_an_operation_that_raises_ends_the_graph_capture_and_propagates(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
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
        self.assertIn('[PINDIAG] trace census seq=1 ccl before{ag_idx=2 barrier_idx=0} after{ag_idx=5 barrier_idx=0}', self.lines)


class FreedRangeTests(unittest.TestCase):
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

    def test_a_buffer_inside_an_earlier_traces_freed_temporaries_is_reported_per_chip(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        trace_census.ENGINE_BOUNDARY = 5
        trace_census.TRACES.append(dict(seq=3, site='packed_verifier.capture:909', ranges=[(0x100000, 0x108000, 'DRAM')]))
        trace_census.census_engine('request-E', self.request(operations), operations)
        overlaps = [line for line in self.lines if 'trace overlap' in line]
        self.assertEqual(overlaps, [
            '[PINDIAG] trace overlap request=request-E rows=4 buffer=chip0@0x100000+2048 trace=packed_verifier.capture:909',
            '[PINDIAG] trace overlap request=request-E rows=4 buffer=chip1@0x100000+2048 trace=packed_verifier.capture:909'])
        extents = [line for line in self.lines if 'engine buffers' in line]
        self.assertEqual(extents[0], '[PINDIAG] engine buffers request=request-E rows=4 chip0 buffers=2 lo=0x100000 hi=0x110800')
        self.assertIn('overlaps=2', self.lines[-1])

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

    def test_without_the_flag_nothing_is_read_or_logged(self):
        operations = FakeOperations()
        trace_census.census_engine('request-E', self.request(operations), operations)
        self.assertEqual((self.lines, operations.views), ([], 0))

    def test_an_engine_that_cannot_be_walked_logs_once_and_never_raises(self):
        os.environ[trace_census.CENSUS_FLAG] = '1'
        operations = FakeOperations()
        request = SimpleNamespace(engine=SimpleNamespace(buckets=5))
        trace_census.census_engine('request-E', request, operations)
        trace_census.census_engine('request-E', request, operations)
        self.assertEqual(len([line for line in self.lines if 'unavailable' in line]), 1)

    def test_the_stage_flag_records_when_the_engine_was_built_for_the_first_replay_line(self):
        os.environ[trace_census.STAGE_LOG_FLAG] = '1'
        operations = FakeOperations()
        trace_census.PACKED_STEPS = 7
        trace_census.engine_begin()
        trace_census.PACKED_STEPS = 10
        request = self.request(operations)
        trace_census.census_engine('request-E', request, operations)
        trace_census.first_replay(request.engine, 'request-E', 4)
        self.assertEqual(self.lines, ['[PINDIAG] first replay request=request-E rows=4 built_after_packed_round=7 '
                                      'packed_rounds_since_build=3 program_cache=n/a'])


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


if __name__ == '__main__':
    unittest.main()
