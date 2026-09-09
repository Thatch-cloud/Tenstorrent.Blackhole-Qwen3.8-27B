from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from dspark_markov_device import execute, validate


def tensor(shape, dtype, layout):
    return SimpleNamespace(shape=shape, dtype=dtype, layout=layout, memory_config=lambda: 'dram')


def operands(vocabulary=64, steps=3):
    operations = SimpleNamespace(uint32='uint32', float32='float32', bfloat16='bf16',
        ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram')
    values = [tensor((1, 1, 1, 1), 'uint32', 'row'), tensor((1, 1, steps, vocabulary), 'float32', 'tile'),
        tensor((1, 1, vocabulary, 256), 'bf16', 'row'), tensor((1, 1, 256, vocabulary), 'bf16', 'tile')]
    return operations, values


class DSparkMarkovDeviceTests(unittest.TestCase):
    def test_complete_vocabulary_geometries(self):
        for vocabulary in (64, 248320):
            for steps in (3, 7, 15):
                operations, values = operands(vocabulary, steps)
                self.assertEqual(validate(operations, *values), (steps, vocabulary))

    def test_reject_sharded_vocab_and_wrong_operands(self):
        for index, field, value in ((1, 'shape', (1, 1, 7, 124160)),
                (1, 'shape', (1, 1, 8, 248320)), (0, 'dtype', 'bf16'), (2, 'layout', 'tile'),
                (3, 'dtype', 'float32'), (3, 'shape', (1, 1, 248320, 256)),
                (1, 'memory_config', lambda: 'l1')):
            with self.subTest(index=index, field=field):
                operations, values = operands(248320, 7)
                setattr(values[index], field, value)
                with self.assertRaises(ValueError):
                    validate(operations, *values)

    def test_native_graph_uses_each_selected_token_as_next_predecessor(self):
        operations, values = operands(248320, 7)
        tokens = [object() for _ in range(7)]
        for name in ('embedding', 'to_layout', 'matmul', 'slice', 'add', 'untilize'):
            setattr(operations, name, MagicMock(side_effect=lambda *args, **kwargs: object()))
        operations.argmax = MagicMock(side_effect=tokens)
        operations.reshape = MagicMock(side_effect=lambda token, shape: token)
        operations.MatmulMultiCoreReuseMultiCast1DProgramConfig = MagicMock(return_value='program')
        operations.WormholeComputeKernelConfig = MagicMock(return_value='kernel')
        operations.MathFidelity = SimpleNamespace(HiFi4='hifi4')
        owned = []
        records = execute(operations, *values, owned)
        self.assertEqual([call.args[0] for call in operations.embedding.call_args_list], [values[0], *tokens[:-1]])
        self.assertTrue(all(call.args[1] is values[2] for call in operations.embedding.call_args_list))
        self.assertEqual([record['token'] for record in records], tokens)
        self.assertEqual(len(owned), 56)
        self.assertEqual([call.args[1:] for call in operations.slice.call_args_list],
            [((0, 0, step, 0), (1, 1, step + 1, 248320)) for step in range(7)])
        for call in operations.matmul.call_args_list:
            self.assertIs(call.args[1], values[3])
            self.assertEqual(call.kwargs['dtype'], operations.float32)
        config = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs
        self.assertEqual((config['compute_with_storage_grid_size'], config['in0_block_w'], config['per_core_N']),
            ((10, 10), 1, 78))
        self.assertFalse(operations.WormholeComputeKernelConfig.call_args.kwargs['packer_l1_acc'])

    def test_incompatible_input_fails_before_dispatch(self):
        operations, values = operands()
        operations.embedding = MagicMock()
        values[2].dtype = 'float32'
        with self.assertRaises(ValueError):
            execute(operations, *values, [])
        operations.embedding.assert_not_called()


if __name__ == '__main__':
    unittest.main()
