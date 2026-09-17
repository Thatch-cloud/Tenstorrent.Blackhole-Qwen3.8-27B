import unittest
from types import SimpleNamespace

from incremental_history_scope import incremental_history


class History:
    def __init__(self):
        self.position, self.capacity, self.pending = 31, 160, None
        self.mesh = object()
        self.operations = SimpleNamespace(synchronize_device=lambda mesh: None)
        self.layers = tuple(([0] * 31 + [0] * 129, [0] * 160) for layer in range(5))
        self.spare_layers = tuple(([0] * 160, [0] * 160) for layer in range(5))

    def check_prefix(self, prefix, position):
        if self.pending is not None or position != self.position or not 1 <= prefix <= 32:
            raise ValueError('Invalid prefix')

    def prepare_projected(self, *args, **kwargs):
        raise AssertionError('Full-bank fallback must not run')

    def validate_publication(self, publication):
        if publication is not self.pending or publication.status != 'prepared':
            raise ValueError('Stale transaction')

    def commit_publication(self, publication):
        self.validate_publication(publication)
        self.layers, self.spare_layers = self.spare_layers, self.layers
        self.position += publication.prefix
        publication.status, self.pending = 'committed', None

    def discard_publication(self, publication):
        if publication.owner is self and publication.status == 'committed':
            return
        self.validate_publication(publication)
        publication.status, self.pending = 'discarded', None


def writer(mesh, active, delta, spare, plan):
    if len(delta) != 32:
        raise ValueError('Fixed tile required')

    def execute():
        for row in range(plan.first_row, plan.end_row):
            spare[row] = active[row] if row < plan.position else (
                delta[row - plan.position] if row < plan.position + plan.prefix else 0)
    return execute


class IncrementalScopeTests(unittest.TestCase):
    def test_commit_discard_and_restoration(self):
        history, records = History(), []
        original = (History.prepare_projected, History.commit_publication, History.discard_publication)
        module = SimpleNamespace(prepare=object())
        original_prepare = module.prepare
        expected = [0] * 160
        projection = SimpleNamespace(operations=history.operations, mesh=history.mesh)
        with incremental_history(History, module, records, writer_factory=writer):
            for ordinal, (prefix, accepted) in enumerate(((3, True), (32, False), (1, True), (32, True))):
                delta = tuple(([ordinal + 1] * prefix + [999] * (32 - prefix),) * 2 for layer in range(5))
                projection.project = lambda features, tables: delta
                old_position = history.position
                publication = module.prepare(history, projection, (), (), prefix, position=old_position)
                candidate = expected.copy()
                candidate[old_position:old_position + prefix] = [ordinal + 1] * prefix
                for pair in history.layers:
                    for bank in pair:
                        self.assertEqual(bank, expected)
                for pair in publication.layers:
                    for bank in pair:
                        self.assertEqual(bank, candidate)
                if accepted:
                    history.commit_publication(publication)
                    history.discard_publication(publication)
                    expected = candidate
                else:
                    history.discard_publication(publication)
                with self.assertRaises(ValueError):
                    history.commit_publication(publication)
        self.assertEqual(original, (History.prepare_projected, History.commit_publication, History.discard_publication))
        self.assertIs(module.prepare, original_prepare)
        self.assertEqual((records[0]['prepared'], records[0]['committed'], records[0]['discarded']), (4, 3, 1))
        self.assertTrue(records[0]['restored'])
        self.assertFalse(records[0]['failed'])

    def test_partial_write_failure_prevents_reuse(self):
        history, records, calls = History(), [], []
        module = SimpleNamespace(prepare=object())

        def failing_writer(*args):
            operation = writer(*args)

            def execute():
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError('Device failure')
                operation()
            return execute

        with incremental_history(History, module, records, writer_factory=failing_writer):
            delta = tuple(([7] * 32,) * 2 for layer in range(5))
            with self.assertRaises(RuntimeError):
                history.prepare_projected(delta, 3, position=31)
            self.assertEqual(history.position, 31)
            self.assertIsNone(history.pending)
            self.assertTrue(all(bank == [0] * 160 for pair in history.layers for bank in pair))
            with self.assertRaises(ValueError):
                history.prepare_projected(delta, 3, position=31)
        self.assertTrue(records[0]['failed'])
        self.assertTrue(records[0]['restored'])


if __name__ == '__main__':
    unittest.main()
