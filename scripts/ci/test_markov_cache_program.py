from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from markov_cache_program import build


class MarkovCacheProgramTests(unittest.TestCase):
    def test_rejects_shape_dtype_layout_and_memory(self):
        api = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram')
        def tensor(shape=(1, 1, 1, 8), dtype='uint32', layout='row', memory='dram'):
            return SimpleNamespace(shape=shape, dtype=dtype, layout=layout, memory_config=lambda: memory)
        for field, value in (('shape', (8,)), ('dtype', 'int32'), ('layout', 'tile'), ('memory', 'l1')):
            arguments = [tensor((1, 1, 65, 8)), tensor(**{field: value}), tensor(), tensor()]
            with patch.dict('sys.modules', {'ttnn': api}), self.assertRaises(ValueError):
                build(None, *arguments)

    def test_ci_is_weight_free_and_bounded(self):
        directory = Path(__file__).parent
        workflow = (directory.parents[1] / '.github/workflows/qwen-ttsim.yml').read_text()
        self.assertIn('options: [markov-cache-control,', workflow)
        self.assertIn('group: qwen-two-p150a-exclusive', workflow)
        registered = (directory.parents[1] / '.github/workflows/qwen-experiments.yml').read_text()
        self.assertEqual(registered.count('markov-cache-control-sim'), 5)
        self.assertIn("inputs.suite == 'markov-cache-control-sim' && 'markov-cache-control'", registered)
        runner = (directory / 'run-simulator.sh').read_text()
        self.assertIn('= markov-cache-control ]]; then kinds=\'\'; fi', runner)
        suite = (directory / 'simulator-suite.sh').read_text()
        branch = suite.split('= markov-cache-control ]]; then', 1)[1].split('\nfi', 1)[0]
        self.assertIn('timeout -k 15 600', branch)
        self.assertIn('exit "$status"', branch)
        self.assertNotIn('build.py', branch)

    def test_live_anchor_requires_a_uint32_scalar(self):
        api = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram')
        tensors = [SimpleNamespace(shape=shape, dtype='uint32', layout='row', memory_config=lambda: 'dram')
            for shape in ((1, 1, 65, 8), (1, 1, 1, 8), (1, 1, 1, 8), (1, 1, 1, 8))]
        anchor = SimpleNamespace(shape=(1, 1, 1, 8), dtype='uint32', layout='row', memory_config=lambda: 'dram')
        with patch.dict('sys.modules', {'ttnn': api}), self.assertRaises(ValueError):
            build(None, *tensors, anchor=anchor)


if __name__ == '__main__':
    unittest.main()
