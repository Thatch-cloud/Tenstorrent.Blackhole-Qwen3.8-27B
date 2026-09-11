from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import t32_sim_target as target


class SimulatorTargetTests(unittest.TestCase):
    def test_real_boundary_shapes_and_shard_axes(self):
        matrix = torch.arange(8).reshape(4, 2).bfloat16()
        reader = SimpleNamespace(tensor=Mock(return_value=matrix), check_unchanged=Mock(), manifest={})
        operations = SimpleNamespace(bfloat16='bf16', ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile',
            DRAM_MEMORY_CONFIG='dram', ShardTensorToMesh=Mock(return_value='mapper'),
            from_torch=Mock(side_effect=['embedding', 'head']), synchronize_device=Mock(), embedding=Mock())
        mesh = SimpleNamespace(shape=[1, 2])
        owned = []
        with patch.object(target, 'TargetWeights', return_value=reader), \
                patch.object(target, 'VOCABULARY', 4), patch.object(target, 'HIDDEN_WIDTH', 2):
            model, manifest = target.load(operations, mesh, '/target', owned)
        calls = operations.from_torch.call_args_list
        self.assertEqual(tuple(calls[0].args[0].shape), (1, 1, 4, 2))
        self.assertEqual(tuple(calls[1].args[0].shape), (1, 1, 2, 4))
        self.assertTrue(torch.equal(calls[1].args[0][0, 0], matrix.T))
        self.assertEqual([call.kwargs['layout'] for call in calls], ['row', 'tile'])
        self.assertTrue(all(call.kwargs['dim'] == 3 for call in operations.ShardTensorToMesh.call_args_list))
        self.assertEqual(owned, ['embedding', 'head'])
        model.embd('ids', memory_config='dram')
        operations.embedding.assert_called_once_with('ids', 'embedding', layout='tile', memory_config='dram')
        self.assertTrue(model._lmhead_vocab_sharded)

    def test_partial_upload_is_released(self):
        reader = SimpleNamespace(tensor=Mock(return_value=torch.zeros(4, 2)), manifest={})
        operations = SimpleNamespace(bfloat16='bf16', ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile',
            DRAM_MEMORY_CONFIG='dram', ShardTensorToMesh=Mock(),
            from_torch=Mock(side_effect=['embedding', RuntimeError('upload failed')]))
        owned = []
        with patch.object(target, 'TargetWeights', return_value=reader), \
                patch.object(target, 'VOCABULARY', 4), patch.object(target, 'HIDDEN_WIDTH', 2), \
                patch.object(target, 'release_owned') as release:
            with self.assertRaisesRegex(RuntimeError, 'upload failed'):
                target.load(operations, SimpleNamespace(shape=[1, 2]), '/target', owned)
        release.assert_called_once_with(operations, ['embedding'])
        self.assertEqual(owned, [])


if __name__ == '__main__':
    unittest.main()
