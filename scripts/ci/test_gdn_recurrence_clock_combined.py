import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_recurrence_clock_combined as combined
from gdn_recurrence_clock import instrument
from gdn_recurrence_clock_gate import REPORT_SHA256
from test_gdn_recurrence_clock import RecurrenceClockTests
from test_mlp_compute_clock_combined import Capture, Engine


class CombinedRecurrenceTests(unittest.TestCase):
    def setUp(self):
        self.source = RecurrenceClockTests().source()
        self.kernel = dict(control_sha256=hashlib.sha256(self.source.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(instrument(self.source).encode()).hexdigest(), token=8)
        swap = patch.object(combined, 'KERNEL', self.kernel)
        swap.start()
        self.addCleanup(swap.stop)
        self.mesh = object()
        self.model = SimpleNamespace(mesh_device=self.mesh,
            layers=[SimpleNamespace(is_full_attention=index % 4 == 3) for index in range(64)])
        self.captures = [Capture(self.mesh, index + 1) for index in range(48)]
        for capture in self.captures:
            capture.token = 8
        self.operations = SimpleNamespace(get_device_tensors=lambda value: [value, value])
        self.kernels = dict(recurrence=dict(compute=self.source, reader='reader', writer='writer'),
            norm_gate=dict(reader='qualified-prefetch', compute='unchanged-norm'))
        self.pipeline = SimpleNamespace(build_recurrence=lambda *args: object())
        self.pipeline.build = self.build
        self.candidate = SimpleNamespace(build_recurrence=self.construct)
        self.bank = combined.CombinedRecurrenceCapture(self.operations, self.model, self.captures,
            self.candidate, dict(report_sha256=REPORT_SHA256, kernel=self.kernel))

    def build(self, operations, mesh, tensors, *, root):
        program = self.pipeline.build_recurrence(operations, mesh,
            [[value, value] for value in tensors], self.kernels)
        return ((['query'], 'normalization'), (tensors, program), (tensors, 'prefetched-norm'))

    def construct(self, operations, mesh, shards, kernels, token):
        self.assertEqual(len(shards), 12)
        self.assertEqual(token, 8)
        self.assertEqual(kernels['norm_gate'], self.kernels['norm_gate'])
        self.assertEqual(kernels['recurrence']['reader'], 'reader')
        self.assertEqual(kernels['recurrence']['writer'], 'writer')
        self.assertEqual(kernels['recurrence']['compute'], instrument(self.source))
        return object()

    def bind(self, count=96):
        for index in range(count):
            tensors = [object() for unused in range(11)]
            programs = self.pipeline.build(self.operations, self.mesh, tensors, root='root')
            self.assertEqual(programs[0], (['query'], 'normalization'))
            self.assertEqual(programs[2], (tensors, 'prefetched-norm'))
            self.assertEqual(programs[1][0], tensors + [self.captures[index % 48].buffer])

    def test_two_build_rounds_two_replays_keep_norm_prefetch_and_restore(self):
        original_build, original_recurrence = self.pipeline.build, self.pipeline.build_recurrence
        initializer, verifier = Engine.__init__, Engine.verify
        with self.bank.install(self.pipeline, Engine):
            engine = Engine(self.model)
            self.bind()
            for rows, position in ((16, 4096), (16, 4107), (16, 4118), (4, 4129)):
                self.assertEqual(engine.verify(SimpleNamespace(tokens=[1] * rows, position=position)), position)
            with self.assertRaises(ValueError):
                self.bank.assert_releasable()
            engine.phase = 'closed'
        summary = self.bank.summary()
        self.assertEqual(len(summary['builds']), 96)
        self.assertEqual(summary['sampled_verifier_replays'], 2)
        self.assertEqual(summary['layers'], [index for index in range(64) if index % 4 != 3])
        self.assertTrue(all(capture.events == ['poison', 'read', 'poison', 'read'] for capture in self.captures))
        self.assertIs(self.pipeline.build, original_build)
        self.assertIs(self.pipeline.build_recurrence, original_recurrence)
        self.assertIs(Engine.__init__, initializer)
        self.assertIs(Engine.verify, verifier)
        self.assertEqual(self.kernels['recurrence']['compute'], self.source)

    def test_partial_bindings_alias_and_unqualified_admission_rejected(self):
        with self.bank.install(self.pipeline, Engine):
            engine = Engine(self.model)
            self.bind(1)
            with self.assertRaisesRegex(ValueError, 'Complete all-layer'):
                engine.verify(SimpleNamespace(tokens=[1] * 16, position=4096))
            engine.phase = 'closed'
        with self.assertRaises(ValueError):
            self.bank.summary()
        self.captures[1].index = self.captures[0].index
        with self.assertRaisesRegex(ValueError, 'alias'):
            combined.CombinedRecurrenceCapture(self.operations, self.model, self.captures, self.candidate,
                dict(report_sha256=REPORT_SHA256, kernel=self.kernel))
        with self.assertRaises(ValueError):
            combined.CombinedRecurrenceCapture(self.operations, self.model, self.captures, self.candidate, {})

    def test_failed_capture_retains_storage_and_restores_scope(self):
        original = self.pipeline.build
        with self.assertRaisesRegex(ValueError, 'capture failed'):
            with self.bank.install(self.pipeline, Engine):
                Engine(self.model, fail=True)
        self.assertIs(self.pipeline.build, original)
        self.assertTrue(self.bank.failed)
        with self.assertRaisesRegex(ValueError, 'Release verifier traces'):
            self.bank.assert_releasable()

    def test_failed_replay_never_retries_and_discards_partial_samples(self):
        attempts = []
        def fail():
            attempts.append(True)
            raise RuntimeError('replay failed')
        with self.assertRaisesRegex(RuntimeError, 'replay failed'):
            with self.bank.install(self.pipeline, Engine):
                engine = Engine(self.model)
                self.bind(48)
                self.bank.verify(engine, SimpleNamespace(tokens=[1] * 16, position=4096), fail)
        self.assertEqual(attempts, [True])
        self.assertTrue(self.bank.failed)
        self.assertFalse(self.bank.records)
        self.assertTrue(all(not capture.pending for capture in self.captures))
        engine.phase = 'closed'
        self.bank.assert_releasable()

    def test_staging_remains_a_single_audited_combined_request(self):
        from gdn_recurrence_clock_combined_stage import adapt

        sources = {
            'dspark_request_experiment.py': "def run():\n    schedule = (('publication', True), ('publication', False), ('publication', False))\n",
            'dspark-target-hardware.py': 'def run():\n    if True:\n        if True:\n            from dspark_request_experiment import run_loaded_requests\n',
            'run-dspark-hardware.sh': '    -e "QWEN_DSPARK_MODE=$mode"',
        }
        result = adapt(sources)
        self.assertIn("schedule = (('publication', True),)", result['dspark_request_experiment.py'])
        self.assertIn('QWEN_GDN_RECURRENCE_CLOCK_COMBINED', result['run-dspark-hardware.sh'])
        with self.assertRaises(ValueError):
            adapt(result)
        workflow = (Path(__file__).parents[2] / '.github/workflows/qwen-gdn-recurrence-clock-combined.yml').read_text()
        self.assertIn('35243917766', workflow)
        self.assertIn('runner_io_gate.py --attempts 4', workflow)
        self.assertIn('QWEN_DSPARK_FUSION_T16:', workflow)
        self.assertIn('group: qwen-two-p150a-exclusive', workflow)


if __name__ == '__main__':
    unittest.main()
