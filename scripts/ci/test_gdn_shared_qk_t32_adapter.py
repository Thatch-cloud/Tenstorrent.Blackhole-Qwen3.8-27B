import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from gdn_shared_qk_t32_adapter import adapt_probe, payloads, require_simulator


class T32SharedQKAdapterTests(unittest.TestCase):
    def test_probe_checks_all_states_and_changing_inputs(self):
        source = Path(__file__).with_name('gdn-shared-recurrence-probe.py').read_text()
        probe = adapt_probe(source)
        self.assertIn('rows=32, norm_unchanged=True', probe)
        self.assertIn('mask.reshape(32, -1)', probe)
        self.assertIn('from shared_qk_norm_t32_scatter import build as build_pipeline', probe)
        self.assertIn("len(report['checks']) != 24", probe)
        self.assertIn("len(report['immutable_checks']) != 48", probe)
        self.assertIn("len(report['stale_controls']) != 6", probe)
        self.assertIn('torch.equal(previous, current)', probe)
        self.assertNotIn('(2, 16,', probe)
        self.assertNotIn('allocate((16,', probe)
        with self.assertRaises(ValueError):
            adapt_probe(probe)

    def test_separate_builders_change_width_not_kernel_sources(self):
        directory = Path(__file__).parent
        originals = {name: (directory / name).read_text() for name in
            ('gdn_shared_qk_program.py', 'gdn_shared_qk_pipeline.py', 'shared_qk_norm_scatter.py')}
        kernels = {name: (directory / name).read_bytes() for name in
            ('gdn_shared_qk_compute.py', 'gdn_shared_qk_dataflow.py', 'gdn_shared_qk_recurrence.py', 'gdn_norm_scatter.py')}
        sources = payloads(originals)
        program = sources['gdn_shared_qk_t32_program.py']
        pipeline = sources['gdn_shared_qk_t32_pipeline.py']
        self.assertIn('[head, 32, addresses[0]]', program)
        self.assertIn('[head, 32, *addresses[1:]]', program)
        self.assertIn("if role == 'writer' else [32]", program)
        self.assertIn("'recurrence', 32, False", pipeline)
        self.assertIn('kernels, "norm_gate", 32)', pipeline)
        self.assertIn('(32, 24, 128, 128)', pipeline)
        self.assertIn('import gdn_shared_qk_t32_pipeline as pipeline', sources['shared_qk_norm_t32_scatter.py'])
        for name, original in originals.items():
            self.assertEqual((directory / name).read_text(), original)
        for name, original in kernels.items():
            self.assertEqual((directory / name).read_bytes(), original)
        self.assertIn('token / 16', kernels['gdn_shared_qk_recurrence.py'].decode())
        self.assertIn('token % 16', kernels['gdn_shared_qk_recurrence.py'].decode())
        for name in ('gdn_shared_qk_t32_program.py', 'gdn_shared_qk_t32_pipeline.py'):
            functions = [node for node in ast.parse(sources[name]).body if isinstance(node, ast.FunctionDef)]
            for function in functions:
                self.assertIsInstance(function.body[0], ast.ImportFrom)
                self.assertEqual(function.body[0].module, 'gdn_shared_qk_t32_adapter')
                self.assertEqual(ast.unparse(function.body[1]), 'require_simulator()')

    def test_hardware_cannot_enter_candidate_builders(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            require_simulator()
        environment = dict(QWEN_SIM_ONLY='1', TT_METAL_SIMULATOR='fixture')
        with patch.dict(os.environ, environment, clear=True):
            require_simulator()
        for flag in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'):
            with patch.dict(os.environ, {**environment, flag: '1'}, clear=True), self.assertRaises(ValueError):
                require_simulator()


if __name__ == '__main__':
    unittest.main()
