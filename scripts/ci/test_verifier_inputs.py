from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from verifier_inputs import host_inputs, stage_inputs, validate_tokens


class InputTests(unittest.TestCase):
    def test_host_metadata_shapes_and_rotary_zero(self):
        for rows in (1, 2, 4, 8, 16, 32):
            tokens, positions, cos, sin = host_inputs(list(range(rows)), 0, 64, 1000000)
            self.assertEqual(tuple(tokens.shape), (rows, 1))
            self.assertEqual(positions.tolist(), list(range(rows)))
            self.assertEqual(tuple(cos.shape), (1, rows, 1, 64))
            self.assertTrue(torch.equal(cos[:, 0], torch.ones_like(cos[:, 0])))
            self.assertTrue(torch.equal(sin[:, 0], torch.zeros_like(sin[:, 0])))
            self.assertEqual(cos.dtype, torch.bfloat16)

    def test_rejects_invalid_ticket_before_device_updates(self):
        for tokens, rows, start in (([1], 2, 0), ([True], 1, 0), ([10], 1, 0), ([1], 1, -1),
                                    ([1, 2], 2, 65535), ([1], 1, True)):
            with self.assertRaises(ValueError):
                validate_tokens(tokens, rows, start, 10, 65536)

    def fixture(self):
        operations = SimpleNamespace(uint32='uint32', int32='int32', bfloat16='BF16', ROW_MAJOR_LAYOUT='RM', TILE_LAYOUT='tile',
            from_torch=Mock(side_effect=lambda value, **kwargs: value), copy_host_to_device_tensor=Mock(),
            synchronize_device=Mock(), ReplicateTensorToMesh=Mock())
        operations.get_device_tensors = lambda value: [SimpleNamespace(buffer_address=lambda: id(value))] * 2

        def tensor(shape, dtype, layout):
            return SimpleNamespace(shape=shape, dtype=dtype, layout=layout)

        return SimpleNamespace(rows=2, operations=operations,
            model=SimpleNamespace(args=SimpleNamespace(vocab_size=100, rope_head_dim=64, rope_theta=1000000), mesh_device='mesh'),
            pages=SimpleNamespace(shape=(2, 1024)), tokens=tensor((2, 1), 'uint32', 'RM'),
            positions=tensor((2,), 'int32', 'RM'), cos=tensor((1, 2, 1, 64), 'BF16', 'tile'),
            sin=tensor((1, 2, 1, 64), 'BF16', 'tile'), singleton_positions=[tensor((1,), 'int32', 'RM') for index in range(2)])

    def test_stages_all_positions_and_keeps_host_buffers_until_sync(self):
        fixture = self.fixture()
        stage_inputs(fixture, [7, 8], 16383)
        operations = fixture.operations
        self.assertEqual(operations.copy_host_to_device_tensor.call_count, 6)
        values = [call.args[0] for call in operations.copy_host_to_device_tensor.call_args_list]
        self.assertEqual(values[0].tolist(), [[7], [8]])
        self.assertEqual(values[1].tolist(), [16383, 16384])
        self.assertEqual([value.item() for value in values[-2:]], [16383, 16384])
        self.assertTrue(all(call.kwargs['device'] is None for call in operations.from_torch.call_args_list))
        operations.synchronize_device.assert_called_once_with('mesh')

    def test_rejects_signature_or_alias_changes_before_copy(self):
        for invalid in ('signature', 'alias', 'missing'):
            fixture = self.fixture()
            if invalid == 'signature':
                fixture.cos.shape = (1, 1, 1, 64)
            elif invalid == 'alias':
                fixture.singleton_positions[1] = fixture.singleton_positions[0]
            else:
                fixture.singleton_positions.pop()
            with self.assertRaises(ValueError):
                stage_inputs(fixture, [7, 8], 16383)
            fixture.operations.copy_host_to_device_tensor.assert_not_called()

    def test_partial_upload_failure_drains_before_staging_buffers_expire(self):
        fixture = self.fixture()
        fixture.operations.copy_host_to_device_tensor.side_effect = [None, RuntimeError('copy')]
        with self.assertRaisesRegex(RuntimeError, 'copy'):
            stage_inputs(fixture, [7, 8], 16383)
        fixture.operations.synchronize_device.assert_called_once_with('mesh')

    def test_replay_position_joins_the_same_staging_transaction(self):
        fixture = self.fixture()
        fixture.replay_reader = SimpleNamespace(validate=Mock(), start=4096, failed=False,
            positions=SimpleNamespace(shape=(8,), dtype='int32', layout='RM'))
        stage_inputs(fixture, [7, 8], 4103)
        fixture.replay_reader.validate.assert_called_once_with(4103)
        self.assertEqual(fixture.operations.copy_host_to_device_tensor.call_count, 7)
        self.assertEqual(fixture.operations.copy_host_to_device_tensor.call_args.args[0].tolist(), [4103, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(fixture.replay_reader.start, 4103)
        fixture.operations.synchronize_device.assert_called_once()

    def test_replay_family_rejected_before_any_metadata_is_changed(self):
        fixture = self.fixture()
        fixture.replay_reader = SimpleNamespace(validate=Mock(side_effect=ValueError('family')))
        with self.assertRaisesRegex(ValueError, 'family'):
            stage_inputs(fixture, [7, 8], 4352)
        fixture.operations.copy_host_to_device_tensor.assert_not_called()

    def test_partial_staging_poison_includes_attention_reader(self):
        fixture = self.fixture()
        fixture.replay_reader = SimpleNamespace(validate=Mock(), start=4096, failed=False,
            positions=SimpleNamespace(shape=(8,), dtype='int32', layout='RM'))
        fixture.operations.copy_host_to_device_tensor.side_effect = [None, RuntimeError('copy')]
        with self.assertRaisesRegex(RuntimeError, 'copy'):
            stage_inputs(fixture, [7, 8], 4103)
        self.assertTrue(fixture.replay_reader.failed)
        self.assertEqual(fixture.replay_reader.start, 4096)


if __name__ == '__main__':
    unittest.main()
