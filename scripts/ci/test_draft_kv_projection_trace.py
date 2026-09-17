from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from draft_kv_projection_trace import PreparedDraftKVProjection


class Tensor:
    def __init__(self, value):
        self.value = value.clone()
        self.shape, self.dtype, self.layout = value.shape, value.dtype, 'tile'
        self.address = id(self)

    def memory_config(self):
        return 'dram'


class ProjectionTraceTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        operations = SimpleNamespace(bfloat16=torch.bfloat16, TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            from_torch=lambda value, **kwargs: Tensor(value), ReplicateTensorToMesh=lambda mesh: mesh,
            synchronize_device=Mock(), deallocate=Mock(), release_trace=Mock(), recording=False, commands=[])
        operations.copy = Mock(side_effect=lambda source, destination: destination.value.copy_(source.value))
        def begin(mesh, **kwargs):
            operations.recording = True
            return 23
        operations.begin_trace_capture = Mock(side_effect=begin)
        operations.end_trace_capture = Mock(side_effect=lambda *args, **kwargs: setattr(operations, 'recording', False))
        operations.execute_trace = Mock(side_effect=lambda *args, **kwargs: [command() for command in operations.commands])
        def project(runtime, inputs, query, tables, retain, *, parameters):
            outputs = {name: retain(Tensor(torch.zeros((1, 4, 32, 128), dtype=torch.bfloat16))) for name in ('k', 'v')}
            def execute():
                value = inputs.value[..., :512].reshape(1, 32, 4, 128).transpose(1, 2) * parameters['scale']
                outputs['v'].value.copy_(value)
                outputs['k'].value.copy_(value + tables[0].value - tables[1].value)
            execute()
            if operations.recording:
                operations.commands.append(execute)
            return dict(q=query, **outputs)
        query = Tensor(torch.zeros((1, 1, 32, 2048), dtype=torch.bfloat16))
        with patch('draft_kv_projection_trace.project_key_value', side_effect=project) as projection, \
                patch('draft_kv_projection_trace.addresses', side_effect=lambda runtime, value: (value.address, value.address + 1)), \
                patch('draft_kv_projection_trace.release_owned', side_effect=lambda runtime, values:
                    [runtime.deallocate(value) for value in values]):
            prepared = PreparedDraftKVProjection(operations, object(), [dict(scale=1), dict(scale=2)], query)
            try:
                yield prepared, operations, projection, query
            finally:
                prepared.close()

    def inputs(self, scalar=1):
        return [Tensor(torch.full((1, 1, 32, width), scalar + index, dtype=torch.bfloat16))
            for index, width in enumerate((5120, 128, 128))]

    def test_replay_updates_features_and_positions_without_reissuing_projection(self):
        with self.fixture() as (prepared, operations, projection, query):
            bindings = list(prepared.bindings)
            for scalar in (1, 7):
                features, cosine, sine = self.inputs(scalar)
                sine.value.fill_(scalar * 2)
                outputs = prepared.project(features, (cosine, sine))
                for layer, result in enumerate(outputs):
                    self.assertTrue(torch.all(result['v'].value == scalar * (layer + 1)))
                    self.assertTrue(torch.all(result['k'].value == scalar * (layer + 1) + scalar + 1 - scalar * 2))
                self.assertEqual(prepared.bindings, bindings)
                self.assertEqual(projection.call_count, 4)
            self.assertEqual(prepared.calls, 2)
            self.assertEqual(operations.execute_trace.call_count, 3)
            prepared.close()
            prepared.close()
            operations.release_trace.assert_called_once()
            self.assertFalse(any(call.args[0] is query for call in operations.deallocate.call_args_list))

    def test_invalid_input_or_moved_binding_fails_before_copy_or_replay(self):
        with self.fixture() as (prepared, operations, projection, query):
            features, cosine, sine = self.inputs()
            operations.copy.reset_mock()
            operations.execute_trace.reset_mock()
            for values in ((features, (cosine,)), (Tensor(torch.zeros((1, 1, 31, 5120), dtype=torch.bfloat16)), (cosine, sine))):
                with self.assertRaises(ValueError):
                    prepared.project(*values)
            prepared.inputs[0].address += 100
            with self.assertRaisesRegex(AssertionError, 'bindings moved'):
                prepared.project(features, (cosine, sine))
            operations.copy.assert_not_called()
            operations.execute_trace.assert_not_called()


if __name__ == '__main__':
    unittest.main()
