import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlp_block_stream_projection import adapt_projection, bind_stream, validate_binding


class BlockStreamProjectionTests(unittest.TestCase):
    def fixture(self):
        def tensor(shape, dtype, layout, address):
            return SimpleNamespace(shape=shape, dtype=dtype, layout=layout, address=address,
                memory_config=lambda: 'dram')

        operations = SimpleNamespace(uint32='uint32', bfloat4_b='bf4', ROW_MAJOR_LAYOUT='row',
            TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram', get_device_tensors=lambda value:
                [SimpleNamespace(buffer_address=lambda: value.address)] * 2)
        source = tensor((5120, 17408), 'bf4', 'tile', 100)
        stream = tensor((1, 1, 1820, 6912), 'uint32', 'row', 200)
        projection = SimpleNamespace(token_rows=16, pairs_per_worker=3, intermediates=False,
            math_approx_mode=True, weights=source, compute='unchanged register epilogue')
        return projection, stream, operations

    def test_adaptation_preserves_compute_and_constructor_ast(self):
        source = Path(__file__).with_name('fused_1d.py').read_text()
        changed = adapt_projection(source)
        original_ast, candidate_ast = ast.parse(source), ast.parse(changed)
        for tree in (original_ast, candidate_ast):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'FusedProjection':
                    node.body = [function for function in node.body if function.name != '__call__']
            tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom)
                and node.module in ('mlp_block_stream', 'mlp_block_stream_projection'))]
        self.assertEqual(ast.dump(original_ast), ast.dump(candidate_ast))
        self.assertIn('ttnn.generic_op([value, stream_weights, output]', changed)
        self.assertIn('validate_binding(self, ttnn)', changed)
        with self.assertRaises(ValueError):
            adapt_projection(changed)

    def test_simulator_binding_is_stable_and_does_not_take_source_ownership(self):
        projection, stream, operations = self.fixture()
        with patch.dict(os.environ, dict(QWEN_SIM_ONLY='1', TT_METAL_SIMULATOR='fixture'), clear=True):
            with self.assertRaises(ValueError):
                validate_binding(projection, operations)
            report = bind_stream(projection, stream, operations)
            self.assertTrue(report['borrowed_source'])
            self.assertTrue(report['stream_owned_by_caller'])
            self.assertFalse(report['arithmetic_changed'])
            self.assertIs(validate_binding(projection, operations), stream)
            with self.assertRaises(ValueError):
                bind_stream(projection, stream, operations)
            for target, field, changed in ((stream, 'address', 300), (projection.weights, 'address', 400),
                    (projection, 'compute', 'different arithmetic')):
                original = getattr(target, field)
                setattr(target, field, changed)
                with self.assertRaises(ValueError):
                    validate_binding(projection, operations)
                setattr(target, field, original)

    def test_hardware_partial_or_aliasing_streams_are_rejected(self):
        projection, stream, operations = self.fixture()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            bind_stream(projection, stream, operations)
        with patch.dict(os.environ, dict(QWEN_SIM_ONLY='1', TT_METAL_SIMULATOR='fixture'), clear=True):
            stream.address = 100
            with self.assertRaises(ValueError):
                bind_stream(projection, stream, operations)
            stream.address = 200
            stream.shape = (1, 1, 1, 6912)
            with self.assertRaises(ValueError):
                bind_stream(projection, stream, operations)


if __name__ == '__main__':
    unittest.main()
