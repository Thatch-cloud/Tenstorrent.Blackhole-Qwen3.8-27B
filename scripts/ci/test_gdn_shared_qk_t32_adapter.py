import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gdn_shared_qk_t32_adapter import adapt_probe, payloads, require_simulator
from gdn_shared_qk_t32_stage import stage
from frozen_recipe_context import adapt_cache_launcher


class T32SharedQKAdapterTests(unittest.TestCase):
    def test_stage_retains_baseline_builders_and_records_generated_sources(self):
        directory = Path(__file__).parent
        names = ('gdn_shared_qk_program.py', 'gdn_shared_qk_pipeline.py', 'gdn-shared-recurrence-probe.py')
        originals = {name: (directory / name).read_text() for name in names}
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            scripts = checkout / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name, source in originals.items():
                (scripts / name).write_text(source)
            from frozen_sim_assets import ASSETS

            downloads = '\n'.join(f'curl --fail --location --max-time 180 {url} -o "$assets/{name}"'
                for name, url, checksum in ASSETS)
            launcher = ("    -e \"QWEN_SIM_CASE=${QWEN_SIM_CASE:-stack}\"\n" + downloads + '\n'
                '        docker logs "$container" > experiment-results/simulator-container.log 2>&1 || true\n'
                '        docker cp "$container:/experiment/results/." experiment-results/ || true\n'
                '        docker rm -f "$container" >/dev/null || true\ncontainer=$(docker create\n')
            suite = ('if [[ "$QWEN_SIM_CASE" = dspark-native-8k-attention ]]; then\n'
                'timeout -k 30 1900 python3 -u /experiment-scripts/ci/dspark_fp32_build.py\n'
                'timeout -k 15 "$limit" python3 -u "/experiment-scripts/ci/$QWEN_SIM_CASE-probe.py"\nfi\n')
            prepared = adapt_cache_launcher({'run-simulator.sh': launcher, 'simulator-suite.sh': suite})
            (scripts / 'simulator-suite.sh').write_text(prepared['simulator-suite.sh'])
            manifest = checkout / 'candidate.json'
            with patch('gdn_shared_qk_t32_stage.subprocess.check_output',
                    side_effect=lambda command: originals[command[-1].rsplit('/', 1)[1]].encode()):
                stage(checkout, manifest)
                with self.assertRaises(ValueError):
                    stage(checkout, manifest)
            report = json.loads(manifest.read_text())
            self.assertEqual(report['rows'], 32)
            self.assertFalse(report['simulator_qualified'])
            for name in names[:2]:
                self.assertEqual((scripts / name).read_text(), originals[name])
            for name, digest in report['after'].items():
                self.assertEqual(hashlib.sha256((scripts / name).read_bytes()).hexdigest(), digest)
                if name.endswith('.py'):
                    compile((scripts / name).read_text(), name, 'exec')
            self.assertIn('frozen_sim_phase.py --phase probe --seconds 420',
                (scripts / 'simulator-suite.sh').read_text())

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
