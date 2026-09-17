import unittest
from types import SimpleNamespace

import incremental_history_scope as module
from history_writer_profile import profile_writers
from test_incremental_history_scope import History, writer


class HistoryWriterProfileTests(unittest.TestCase):
    def test_observation_preserves_writes_sync_and_restoration(self):
        history, records, updates, syncs = History(), [], [], []
        history.operations.synchronize_device = lambda mesh: syncs.append(mesh)
        original = module.incremental_history
        prepare = History.prepare_projected
        publication_module = SimpleNamespace(prepare=object())
        added = tuple(([7] * 32, [7] * 32) for layer in range(5))
        with profile_writers(module, records):
            with module.incremental_history(History, publication_module, updates, writer_factory=writer):
                publication = history.prepare_projected(added, 3, position=31)
                history.commit_publication(publication)
        self.assertEqual(len(syncs), 1)
        self.assertEqual(history.layers[0][0][31:34], [7] * 3)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['passed'])
        self.assertEqual(len(records[0]['stages']), 21)
        self.assertFalse(records[0]['added_device_fences'])
        self.assertIs(module.incremental_history, original)
        self.assertIs(History.prepare_projected, prepare)

    def test_failure_is_recorded_and_hooks_restored(self):
        history, records = History(), []
        original = module.incremental_history
        synchronize = history.operations.synchronize_device

        def failed(*args):
            raise RuntimeError('writer failed')

        with self.assertRaisesRegex(RuntimeError, 'writer failed'):
            with profile_writers(module, records):
                with module.incremental_history(History, SimpleNamespace(prepare=object()), [], writer_factory=failed):
                    history.prepare_projected(tuple(([7] * 32, [7] * 32) for layer in range(5)), 3, position=31)
        self.assertIs(module.incremental_history, original)
        self.assertIs(history.operations.synchronize_device, synchronize)
        self.assertFalse(records[0]['passed'])
        self.assertFalse(records[0]['stages'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
