import gc
import unittest

from publication_diagnostics import PublicationDiagnostics


class PublicationDiagnosticsTests(unittest.TestCase):
    def test_dspark_success_abort_and_failure_keep_transaction_semantics(self):
        from dspark_request_runtime import DSparkRequestRuntime
        from test_dflash_request_runtime import DFlashRequestRuntimeTests
        fixture = DFlashRequestRuntimeTests()
        for prefix, fail in ((2, False), (0, False), (2, True)):
            unused, drafter, session, engine = fixture.fixture()
            runtime = DSparkRequestRuntime(drafter, position=170)
            runtime.bind(session, engine)
            fixture.pending(runtime, session, engine)
            if fail:
                engine.publish.side_effect = RuntimeError('target failed')
                with self.assertRaises(RuntimeError):
                    runtime.publish(prefix)
                drafter.discard_publication.assert_called_once()
                self.assertEqual(runtime.phase, 'failed')
            else:
                runtime.publish(prefix)
                self.assertEqual(runtime.position, 170 + prefix)
            records = runtime.publication_diagnostics.records
            expected = ['features', 'prepare_history', 'publish_target', 'commit_history'] if prefix else ['publish_target']
            self.assertEqual([record['stage'] for record in records], expected[:-1] if fail else expected)
            self.assertEqual(records[-1]['passed'], not fail)

    def test_records_gc_without_changing_policy_or_existing_callbacks(self):
        callbacks, enabled, thresholds = list(gc.callbacks), gc.isenabled(), gc.get_threshold()
        observer = PublicationDiagnostics()
        with observer.stage('prepare_history', 4096, 11):
            gc.collect(0)
        self.assertEqual(gc.callbacks, callbacks)
        self.assertEqual(gc.isenabled(), enabled)
        self.assertEqual(gc.get_threshold(), thresholds)
        record = observer.records[0]
        self.assertTrue(record['passed'])
        self.assertEqual((record['position'], record['prefix']), (4096, 11))
        self.assertTrue(any(pause['generation'] == 0 for pause in record['gc_pauses']))
        self.assertGreaterEqual(record['host_ms'], 0)

    def test_failure_propagates_and_removes_only_own_callback(self):
        callbacks = list(gc.callbacks)
        observer = PublicationDiagnostics()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with observer.stage('publish_target', 4096, 0):
                raise RuntimeError('injected')
        self.assertEqual(gc.callbacks, callbacks)
        self.assertFalse(observer.records[0]['passed'])
