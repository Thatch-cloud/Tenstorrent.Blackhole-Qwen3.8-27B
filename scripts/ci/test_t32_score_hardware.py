from contextlib import ExitStack
from pathlib import Path
import hashlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import t32_score_hardware as hardware
from test_dspark_markov_device import operands, tensor


class ScoreHardwareTests(unittest.TestCase):
    def test_native_source_checks_accept_string_kernel_directory_and_detect_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / hardware.KERNEL_DIRECTORY
            kernel.mkdir(parents=True)
            (root / 'runtime.so').write_bytes(b'runtime')
            (kernel / 'compute.hpp').write_bytes(b'compute')
            runtime = {'runtime.so': hashlib.sha256(b'runtime').hexdigest()}
            patched = {'compute.hpp': hashlib.sha256(b'compute').hexdigest()}
            with patch.object(hardware, 'RUNTIME', runtime), patch.object(hardware, 'PATCHED', patched):
                result = hardware.native_sources(root)
                self.assertEqual(len(result), 2)
                self.assertEqual(result[str(Path(hardware.KERNEL_DIRECTORY) / 'compute.hpp')], patched['compute.hpp'])
                (kernel / 'compute.hpp').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Installed simulator-qualified'):
                    hardware.native_sources(root)

    def test_complete_proposal_compares_native_backend_and_restores_on_failure(self):
        device = hardware.HardwareTracedDSparkDevice.__new__(hardware.HardwareTracedDSparkDevice)
        device.mesh, device.proposal_markov = object(), object()
        device.history = SimpleNamespace(position=4096)
        owner = SimpleNamespace(retain=Mock(), release=Mock())
        reference = tuple(torch.tensor([index]) for index in range(6))
        device.prepared = SimpleNamespace(audit=True, update=Mock(), output_owner=lambda: owner,
            snapshot=Mock(side_effect=[reference, tuple(value.clone() for value in reference)]),
            inputs={}, history=(), outputs={})
        state = SimpleNamespace(record=dict(native_proposal_checks=[]))
        backend = device.proposal_markov
        with patch.object(hardware, 'require_active', return_value=state), \
                patch('dspark_t32_prepared.execute', return_value={}) as execute, \
                patch.object(hardware.TracedDSparkDevice, 'propose', return_value=(17, 18)):
            self.assertEqual(device.propose(17, 2), (17, 18))
            execute.assert_called_once()
            self.assertIs(device.proposal_markov, backend)
            self.assertEqual(state.record['native_proposal_checks'][0]['tensors'], 6)
            changed = list(reference)
            changed[-1] = torch.tensor([-1])
            device.prepared.snapshot.side_effect = [reference, tuple(changed)]
            with self.assertRaisesRegex(AssertionError, 'native-score reference'):
                device.propose(18, 2)
            self.assertEqual(len(state.record['native_proposal_checks']), 1)
            execute.side_effect = RuntimeError('native failure')
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                device.propose(19, 2)
            self.assertIs(device.proposal_markov, backend)
            self.assertEqual(owner.release.call_count, 3)
            device.prepared.audit = False
            with self.assertRaisesRegex(ValueError, 'audited prepared'):
                device.propose(19, 2)
            state.timing_reference = {}
            execute.reset_mock()
            self.assertEqual(device.propose(19, 2), (17, 18))
            execute.assert_not_called()
            self.assertEqual(len(state.record['native_proposal_checks']), 1)

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
