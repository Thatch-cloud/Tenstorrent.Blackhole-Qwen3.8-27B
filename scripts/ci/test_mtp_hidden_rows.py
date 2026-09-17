from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from mtp_hidden_rows import MTPHiddenRows


class HiddenRowsTests(unittest.TestCase):
    def test_replay_selects_changed_rows_into_same_destination(self):
        def tensor(data, identity):
            return SimpleNamespace(data=data, shape=data.shape, identity=identity,
                dtype='bf16', layout='tile', memory_config=lambda: 'dram')
        source = tensor(torch.arange(4).reshape(1, 1, 4, 1).expand(1, 1, 4, 5120).clone(), (10, 10))
        destination = tensor(torch.zeros(1, 1, 1, 5120), (20, 20))
        traces, active = {}, []
        def begin(mesh, cq_id):
            trace = len(traces)
            traces[trace] = []
            active.append(trace)
            return trace
        def end(mesh, trace, cq_id):
            active.clear()
        temporary = []
        def record(operation):
            operation()
            if active:
                traces[active[0]].append(operation)
        def slice_row(value, start, stop, *, memory_config):
            result = tensor(torch.zeros_like(destination.data), (30 + len(temporary), 30 + len(temporary)))
            temporary.append(result)
            record(lambda: result.data.copy_(value.data[:, :, start[2]:stop[2]]))
            return result
        def copy(source, target):
            record(lambda: target.data.copy_(source.data))
            return target
        def replay(mesh, trace, cq_id, blocking):
            for operation in traces[trace]:
                operation()
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            from_torch=Mock(return_value=destination), ReplicateTensorToMesh=Mock(),
            slice=slice_row, copy=copy, begin_trace_capture=begin, end_trace_capture=end, execute_trace=replay,
            synchronize_device=Mock(), release_trace=Mock(), deallocate=Mock())
        with patch('mtp_hidden_rows.addresses', side_effect=lambda operations, value: value.identity):
            reader = MTPHiddenRows(operations, object(), [source])
            reader.prepare()
            for value, row in ((7, 3), (11, 1), (7, 3)):
                source.data[:, :, row].fill_(value)
                self.assertIs(reader(source, row), destination)
                self.assertTrue(torch.all(destination.data == value))
            operations.from_torch.assert_called_once()
            with self.assertRaises(ValueError):
                reader(source, 4)
            destination.identity = (21, 20)
            with self.assertRaises(ValueError):
                reader(source, 0)
            reader.close()
            self.assertEqual(operations.deallocate.call_count, len(temporary) + 1)
            self.assertIs(operations.deallocate.call_args.args[0], destination)
            self.assertTrue(all(call.args[0] is not source for call in operations.deallocate.call_args_list))
            self.assertEqual(operations.release_trace.call_count, 4)

    def test_single_row_copies_directly_without_slicing_or_releasing_source(self):
        source = SimpleNamespace(shape=(1, 1, 1, 5120))
        reader = MTPHiddenRows.__new__(MTPHiddenRows)
        reader.destination = object()
        reader.operations = SimpleNamespace(copy=Mock(), slice=Mock(), deallocate=Mock())
        self.assertIs(reader.execute(source, 0), reader.destination)
        reader.operations.copy.assert_called_once_with(source, reader.destination)
        reader.operations.slice.assert_not_called()
        reader.operations.deallocate.assert_not_called()
