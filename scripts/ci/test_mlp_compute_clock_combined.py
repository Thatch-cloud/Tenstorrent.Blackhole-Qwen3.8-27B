from types import SimpleNamespace
import unittest

from mlp_compute_clock_combined import CombinedComputeClockCapture
from mlp_compute_clock_hardware import SIMULATOR_SHA256


class Capture:
    def __init__(self, mesh, index):
        self.mesh, self.index = mesh, index
        self.pending = False
        self.records = []
        self.buffer = object()
        self.events = []

    def addresses(self):
        return (1024 * self.index, 1024 * self.index)

    def prepare(self):
        self.events.append('poison')
        self.pending = True

    def collect(self, label):
        if not self.pending:
            raise ValueError('missing poison')
        self.pending = False
        self.events.append('read')
        result = dict(label=label)
        self.records.append(result)
        return result


class Engine:
    def __init__(self, model, fail=False):
        self.phase = 'preparing'
        if fail:
            raise ValueError('capture failed')
        self.calls = 0
        self.phase = 'idle'

    def verify(self, ticket):
        self.calls += 1
        return ticket.position


class CombinedClockTests(unittest.TestCase):
    def fixture(self):
        mesh = object()
        model = SimpleNamespace(mesh_device=mesh, layers=[SimpleNamespace(
            feed_forward=SimpleNamespace(weights=SimpleNamespace(w_gate_up=object()))) for unused in range(64)])
        captures = [Capture(mesh, index) for index in range(64)]
        candidate = lambda mesh, weights, **kwargs: SimpleNamespace(**kwargs)
        candidate.diagnostic_evidence = dict(passed=True, report_sha256=SIMULATOR_SHA256, kernels=[dict(token_rows=16)])
        module = SimpleNamespace(FusedProjection=candidate, qualify_simulator=lambda: dict(passed=True))
        bank = CombinedComputeClockCapture(model, captures, candidate)
        return model, captures, module, bank

    def bind(self, model, module):
        return [module.FusedProjection(model.mesh_device, layer.feed_forward.weights.w_gate_up,
            token_rows=16, pairs_per_worker=3, math_approx_mode=True) for layer in model.layers]

    def test_two_complete_replays_tail_untouched_and_scope_restoration(self):
        model, captures, module, bank = self.fixture()
        original, initializer, verifier = module.FusedProjection, Engine.__init__, Engine.verify
        with bank.install(module, Engine):
            self.assertEqual(module.qualify_simulator(), bank.candidate.diagnostic_evidence)
            projections = self.bind(model, module)
            self.assertEqual([projection.compute_samples for projection in projections], [capture.buffer for capture in captures])
            engine = Engine(model)
            for rows, position in ((16, 4096), (16, 4107), (16, 4118), (4, 4129)):
                self.assertEqual(engine.verify(SimpleNamespace(tokens=[1] * rows, position=position)), position)
            self.assertEqual(engine.calls, 4)
            with self.assertRaises(ValueError):
                bank.assert_releasable()
            engine.phase = 'closed'
        self.assertIs(module.FusedProjection, original)
        self.assertIs(Engine.__init__, initializer)
        self.assertIs(Engine.verify, verifier)
        self.assertEqual(bank.summary()['sampled_verifier_replays'], 2)
        self.assertTrue(all(capture.events == ['poison', 'read', 'poison', 'read'] for capture in captures))
        with self.assertRaises(ValueError):
            with bank.install(module, Engine):
                pass

    def test_diagnostic_admission_preserves_original_gate_and_restores_it(self):
        model, captures, module, bank = self.fixture()
        def failed_original():
            raise ValueError('Original numerical qualification failed')
        module.qualify_simulator = failed_original
        with bank.install(module, Engine):
            with self.assertRaisesRegex(ValueError, 'Original numerical'):
                module.qualify_simulator()
        self.assertIs(module.qualify_simulator, failed_original)
        model, captures, module, bank = self.fixture()
        bank.candidate.diagnostic_evidence['report_sha256'] = 'unqualified'
        with self.assertRaisesRegex(ValueError, 'Source-admitted'):
            with bank.install(module, Engine):
                pass

    def test_layer_aliases_wrong_width_and_order_rejected(self):
        model, captures, module, bank = self.fixture()
        captures[1].index = 0
        with self.assertRaisesRegex(ValueError, 'aliases'):
            CombinedComputeClockCapture(model, captures, module.FusedProjection)
        with bank.install(module, Engine):
            with self.assertRaises(ValueError):
                module.FusedProjection(model.mesh_device, bank.weights[1], token_rows=16,
                    pairs_per_worker=3, math_approx_mode=True)
            with self.assertRaises(ValueError):
                module.FusedProjection(model.mesh_device, bank.weights[0], token_rows=32,
                    pairs_per_worker=3, math_approx_mode=True)

    def test_failed_engine_capture_does_not_authorize_buffer_release(self):
        model, captures, module, bank = self.fixture()
        with self.assertRaisesRegex(ValueError, 'capture failed'):
            with bank.install(module, Engine):
                self.bind(model, module)
                Engine(model, fail=True)
        self.assertTrue(bank.failed)
        with self.assertRaisesRegex(ValueError, 'Release verifier traces'):
            bank.assert_releasable()
        with self.assertRaises(ValueError):
            bank.summary()

    def test_failed_replay_discards_samples_without_retrying_execution(self):
        model, captures, module, bank = self.fixture()
        attempts = []

        def failed_execution():
            attempts.append('execute')
            raise RuntimeError('device execution failed')

        with self.assertRaisesRegex(RuntimeError, 'device execution failed'):
            with bank.install(module, Engine):
                self.bind(model, module)
                engine = Engine(model)
                bank.verify(engine, SimpleNamespace(tokens=[1] * 16, position=4096), failed_execution)
        self.assertEqual(attempts, ['execute'])
        self.assertTrue(bank.failed)
        self.assertFalse(bank.records)
        self.assertTrue(all(not capture.pending and capture.events == ['poison'] for capture in captures))
        with self.assertRaises(ValueError):
            bank.assert_releasable()
        engine.phase = 'closed'
        bank.assert_releasable()
