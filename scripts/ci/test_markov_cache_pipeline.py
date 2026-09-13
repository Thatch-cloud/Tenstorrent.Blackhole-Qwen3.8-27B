from types import SimpleNamespace
import unittest
from unittest.mock import patch

from markov_cache_pipeline import build


class MarkovCachePipelineTests(unittest.TestCase):
    def test_bad_operand_rejected_before_program_creation(self):
        api = SimpleNamespace(uint32='uint32', float32='float32', bfloat16='bfloat16',
            ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram')
        specifications = (((1, 1, 65, 8), 'uint32', 'row'), ((1, 1, 1, 8), 'uint32', 'row'),
            ((1, 1, 1, 64), 'float32', 'tile'), ((1, 1, 64, 64), 'float32', 'row'),
            ((1, 1, 1, 64), 'float32', 'tile'), ((1, 1, 1, 1), 'bfloat16', 'row'))
        for operand in range(6):
            for field, value in (('dtype', 'wrong'), ('layout', 'wrong'), ('memory_config', lambda: 'l1')):
                tensors = [SimpleNamespace(shape=shape, dtype=dtype, layout=layout, memory_config=lambda: 'dram')
                    for shape, dtype, layout in specifications]
                setattr(tensors[operand], field, value)
                with patch.dict('sys.modules', {'ttnn': api}), self.assertRaises(ValueError):
                    build(None, *tensors)


if __name__ == '__main__':
    unittest.main()
