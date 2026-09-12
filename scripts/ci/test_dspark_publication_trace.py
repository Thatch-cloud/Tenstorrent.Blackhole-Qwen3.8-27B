from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dspark_publication_trace import PreparedHistoryProjection


class Tensor:
    def __init__(self, width, value=0):
        self.shape, self.dtype, self.layout = (1, 1, 32, width), 'bf16', 'tile'
        self.value, self.address = value, id(self)

    def memory_config(self):
        return 'dram'


class PublicationTraceTests(unittest.TestCase):
    @contextmanager
    def fixture(self, fail_capture=False):
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            synchronize_device=Mock(), deallocate=Mock(), release_trace=Mock(), recording=False, commands=[])
        operations.clone = Mock(side_effect=lambda value, **kwargs: Tensor(value.shape[-1], value.value))
        operations.copy = Mock(side_effect=lambda source, destination: setattr(destination, 'value', source.value))
        operations.get_device_tensors = lambda tensor: (tensor, tensor)
        operations.to_torch = lambda tensor: torch.full((1, 4, 32, 128), tensor.value, dtype=torch.bfloat16)

        def begin(*args, **kwargs):
            operations.recording = True
            return 23

        operations.begin_trace_capture = Mock(side_effect=begin)
        operations.end_trace_capture = Mock(side_effect=lambda *args, **kwargs: setattr(operations, 'recording', False))
        operations.execute_trace = Mock(side_effect=lambda *args, **kwargs: [command() for command in operations.commands])

        def project(runtime, mesh, collectives, features, parameters, layers, tables, retain):
            result = tuple(tuple(retain(Tensor(128)) for name in ('k', 'v')) for layer in layers)
            if operations.recording and fail_capture:
                raise RuntimeError('capture failed')

            def execute():
                for index, pair in enumerate(result):
                    pair[0].value = sum(value.value for value in features) + tables[0].value + index
                    pair[1].value = sum(value.value for value in features) + tables[1].value + index

            execute()
            if operations.recording:
                operations.commands.append(execute)
            return result

        features = tuple(Tensor(2560, index) for index in range(5))
        tables = (Tensor(128, 10), Tensor(128, 20))
        parameters = {'weight': Tensor(128)}
        layers = tuple({'weight': Tensor(128)} for index in range(5))
        address = lambda runtime, value: (value.address, value.address + 1)
        with patch('dspark_publication_trace.project_block', side_effect=project) as projection, \
                patch('dspark_publication_trace.addresses', side_effect=address), \
                patch('dspark_history.addresses', side_effect=address):
            yield operations, features, tables, parameters, layers, projection

    def build(self, fixture, **options):
        operations, features, tables, parameters, layers, projection = fixture
        return PreparedHistoryProjection(operations, SimpleNamespace(shape=(1, 2)), None,
            parameters, layers, features, tables, **options)

    def test_audit_detects_corrupted_replay(self):
        with self.fixture() as fixture:
            operations, features, tables, parameters, layers, projection = fixture
            prepared = self.build(fixture, audit=True)
            try:
                features[0].value = 9
                prepared.project(features, tables)
                self.assertEqual(prepared.checks, [dict(tensors=20, exact=True)] * 2)
                operations.commands.append(lambda: setattr(prepared.outputs[4][1], 'value', -999))
                with self.assertRaisesRegex(AssertionError, 'differs from eager'):
                    prepared.project(features, tables)
            finally:
                prepared.close()

    def test_changed_features_and_tables_replay_without_projection_dispatch(self):
        with self.fixture() as fixture:
            operations, features, tables, parameters, layers, projection = fixture
            prepared = self.build(fixture)
            try:
                for scalar in (7, 19):
                    features[0].value = scalar
                    tables[0].value = scalar * 2
                    output = prepared.project(features, tables)
                    for index, pair in enumerate(output):
                        self.assertEqual(pair[0].value, scalar * 3 + 10 + index)
                        self.assertEqual(pair[1].value, scalar + 30 + index)
                self.assertEqual(projection.call_count, 2)
            finally:
                prepared.close()
            prepared.close()
            operations.release_trace.assert_called_once()
            borrowed = [*features, *tables, *parameters.values(), *(value for layer in layers for value in layer.values())]
            self.assertFalse(any(call.args[0] in borrowed for call in operations.deallocate.call_args_list))

    def test_invalid_shape_and_stale_weight_fail_before_copy(self):
        with self.fixture() as fixture:
            operations, features, tables, parameters, layers, projection = fixture
            prepared = self.build(fixture)
            try:
                with self.assertRaises(ValueError):
                    prepared.project(features[:4], tables)
                parameters['weight'].address += 100
                with self.assertRaisesRegex(AssertionError, 'bindings moved'):
                    prepared.project(features, tables)
                operations.copy.assert_not_called()
            finally:
                parameters['weight'].address -= 100
                prepared.close()

    def test_capture_failure_releases_trace_and_all_owned_tensors(self):
        with self.fixture(fail_capture=True) as fixture:
            operations = fixture[0]
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                self.build(fixture)
            operations.release_trace.assert_called_once()
            released = [call.args[0] for call in operations.deallocate.call_args_list]
            self.assertEqual(len(released), 27)
            self.assertEqual(len({id(value) for value in released}), 27)


if __name__ == '__main__':
    unittest.main()
