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
        def slice_row(value, start, stop, *, output_tensor, memory_config):
            operation = lambda: output_tensor.data.copy_(value.data[:, :, start[2]:stop[2]])
            operation()
            if active:
                traces[active[0]].append(operation)
            return output_tensor
        def replay(mesh, trace, cq_id, blocking):
            for operation in traces[trace]:
                operation()
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            from_torch=Mock(return_value=destination), ReplicateTensorToMesh=Mock(),
            slice=slice_row, begin_trace_capture=begin, end_trace_capture=end, execute_trace=replay,
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
            operations.deallocate.assert_called_once_with(destination)
            self.assertEqual(operations.release_trace.call_count, 4)
