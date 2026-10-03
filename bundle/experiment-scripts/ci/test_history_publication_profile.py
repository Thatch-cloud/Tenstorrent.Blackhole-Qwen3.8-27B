from types import SimpleNamespace
import gc
import unittest

from history_publication_profile import HistoryPublicationProfile


class HistoryProfileTests(unittest.TestCase):
    def test_gc_attribution_preserves_collection_and_callbacks(self):
        module = SimpleNamespace(project_chunks=lambda: None)

        class History:
            def prepare_publication(self, value, prefix, *, position):
                gc.collect()
                return value

            def prepare_projected(self):
                pass

        callbacks = list(gc.callbacks)
        enabled = gc.isenabled()
        profiler = HistoryPublicationProfile()
        with profiler.install(History(), module):
            profiler.active = (4096, 1)
            with profiler.stage('test'):
                gc.collect()
            profiler.active = None
        record = profiler.records[0]
        self.assertEqual(record['gc_collections'], 1)
        self.assertGreater(record['gc_ms'], 0)
        self.assertGreaterEqual(record['host_ms'], record['gc_ms'])
        self.assertGreaterEqual(record['process_cpu_ms'], 0)
        self.assertEqual(gc.callbacks, callbacks)
        self.assertEqual(gc.isenabled(), enabled)

    def test_split_records_preserve_results_and_restore_methods(self):
        module = SimpleNamespace(project_chunks=lambda value: value + 1)

        class History:
            def prepare_publication(self, value, prefix, *, position):
                return self.prepare_projected(module.project_chunks(value), prefix, position=position)

            def prepare_projected(self, value, prefix, *, position):
                return value + prefix

        history = History()
        original = module.project_chunks
        profiler = HistoryPublicationProfile()
        with profiler.install(history, module):
            self.assertEqual(history.prepare_publication(5, 3, position=4096), 9)
        self.assertIs(module.project_chunks, original)
        self.assertEqual(vars(history), {})
        self.assertEqual([record['stage'] for record in profiler.records],
            ['history_feature_projection', 'history_bank_assembly', 'history_total'])
        self.assertTrue(all(record['passed'] and record['prefix'] == 3 for record in profiler.records))

    def test_failure_restores_hooks_and_records_failure(self):
        module = SimpleNamespace(project_chunks=lambda: None)

        class History:
            def prepare_publication(self, value, prefix, *, position):
                raise RuntimeError('failed publication')

            def prepare_projected(self):
                pass

        history = History()
        profiler = HistoryPublicationProfile()
        with self.assertRaisesRegex(RuntimeError, 'failed publication'):
            with profiler.install(history, module):
                history.prepare_publication(None, 1, position=4096)
        self.assertEqual(vars(history), {})
        self.assertIsNone(profiler.active)
        self.assertFalse(profiler.records[0]['passed'])


if __name__ == '__main__':
    unittest.main()
