from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import t32_score_hardware as hardware
from test_dspark_markov_device import operands, tensor


class ScoreHardwareTests(unittest.TestCase):
    def fixture(self, stack):
        mesh = SimpleNamespace(shape=[1, 2])
        links, composition = {'backend': 'hardware'}, {'component_composition_audited': True}
        installation = dict(proposal=dict(score_composition=composition), runtime=hardware.RUNTIME,
            patched=hardware.PATCHED, links=links, full_request_qualified=False)
        stack.enter_context(patch.object(hardware, 'environment', return_value=links))
        stack.enter_context(patch.object(hardware, 'composition_audit', return_value=composition))
        stack.enter_context(patch.object(hardware, 'native_sources', return_value={'native': 'unchanged'}))
        return mesh, installation

    def test_hardware_body_is_identical_to_simulator_after_admission(self):
        root = Path(__file__).parent
        hardware.sources(root)
        original = (root / 'dspark_t32_score_layout.py').read_text()
        changed = original.replace('(28, 3)', '(28, 2)')
        self.assertNotEqual(hardware.mathematical_body(original), hardware.mathematical_body(changed))
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(ValueError):
            hardware.require_active()

    def test_owned_scope_executes_all_segments_and_restores_after_failure(self):
        operations, values = operands(64, 31)
        operations.slice = lambda value, start, end: tensor((1, 1, end[2] - start[2], 64), 'float32', 'tile')
        operations.reshape = lambda value, shape: value
        steps, anchors = [], []

        def fused(ops, mesh, anchor, logits, predecessor, successor, owned, *, on_step_enqueued):
            anchors.append(anchor)
            records = []
            for step in range(logits.shape[2]):
                on_step_enqueued(step)
                records.append(dict(token=object()))
            return records

        with ExitStack() as stack:
            mesh, installation = self.fixture(stack)
            stack.enter_context(patch.object(hardware, 'fused', side_effect=fused))
            with hardware.hardware_scope('native', Path(__file__).parent, 'proposal', 'score', mesh, installation) as record:
                result = hardware.execute(operations, *values, [], mesh=mesh, on_step_enqueued=steps.append)
                self.assertEqual(len(result), 31)
                self.assertEqual(steps, list(range(31)))
                self.assertEqual(anchors, [values[0], result[6]['token'], result[13]['token'], result[20]['token'], result[27]['token']])
                with self.assertRaises(ValueError):
                    hardware.require_active(SimpleNamespace(shape=[1, 2]))
                with self.assertRaises(ValueError):
                    with hardware.hardware_scope('native', Path(__file__).parent, 'proposal', 'score', mesh, installation):
                        self.fail('Nested scope admitted')
            self.assertTrue(record['restored'])
            self.assertFalse(record['hardware_qualified'])
            with self.assertRaises(ValueError):
                hardware.require_active(mesh)
            with self.assertRaisesRegex(RuntimeError, 'request'):
                with hardware.hardware_scope('native', Path(__file__).parent, 'proposal', 'score', mesh, installation):
                    raise RuntimeError('request failed')
            self.assertIsNone(hardware._ACTIVE.get())

    def test_changed_installation_and_sources_reject(self):
        with ExitStack() as stack:
            mesh, installation = self.fixture(stack)
            with self.assertRaisesRegex(ValueError, 'Matching'):
                with hardware.hardware_scope('native', Path(__file__).parent, 'proposal', 'score', mesh,
                        dict(installation, full_request_qualified=True)):
                    self.fail('Unbound installation admitted')
            with patch.object(hardware, 'native_sources', side_effect=[{'native': 'before'}, {'native': 'after'}]), \
                    self.assertRaisesRegex(ValueError, 'changed'):
                with hardware.hardware_scope('native', Path(__file__).parent, 'proposal', 'score', mesh, installation) as record:
                    record['calls'] = 1
            self.assertIsNone(hardware._ACTIVE.get())

    def test_device_wrapper_selects_only_owned_hardware_backend(self):
        with ExitStack() as stack:
            mesh, installation = self.fixture(stack)

            def native_init(device, *args, **options):
                self.assertIs(options['fused_score_layout'], False)
                device.mesh = mesh

            stack.enter_context(patch.object(hardware.TracedDSparkDevice, '__init__', native_init))
            with self.assertRaises(ValueError):
                hardware.HardwareTracedDSparkDevice(fused_score_layout=True)
            with hardware.hardware_scope('native', Path(__file__).parent, 'proposal', 'score', mesh, installation) as record:
                device = hardware.HardwareTracedDSparkDevice(fused_score_layout=True)
                self.assertIs(device.proposal_markov.func, hardware.execute)
                self.assertIs(device.proposal_markov.keywords['mesh'], mesh)
                record['calls'] = 1


if __name__ == '__main__':
    unittest.main()
