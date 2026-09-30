from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

from draft_kv_slide_adapter import build_prepare


class SlideAdapterTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).with_name('draft_kv_history.py').read_text()
        self.copies, self.synchronizations = [], []
        self.active = [dict(k=object(), v=object()) for unused in range(5)]
        self.spare = [dict(k=object(), v=object()) for unused in range(5)]
        self.projected = dict(k=object(), v=object())
        @contextmanager
        def temporaries(protected):
            yield lambda value: value
        self.cache = SimpleNamespace(closed=False, pending=None, position=4096, history_rows=2048,
            operations=SimpleNamespace(synchronize_device=self.synchronizations.append), mesh=object(),
            parameters=range(5), active=self.active, spare=self.spare, projection=None, query=object(),
            temporaries=temporaries, project_inputs=lambda *args: (object(), object()))
        self.namespace = dict(SimpleNamespace=SimpleNamespace,
            project_key_value=lambda *args, **kwargs: self.projected)

    def transport(self, mesh, active, delta, spare, **shape):
        return lambda: self.copies.append((mesh, active, delta, spare, shape))

    def test_replaces_only_copy_chain_and_preserves_transaction(self):
        prepare, metadata = build_prepare(self.source, self.namespace, self.transport)
        publication = prepare(self.cache, object(), 16, position=4096)
        self.assertEqual(len(self.copies), 10)
        for index, (mesh, active, delta, spare, shape) in enumerate(self.copies):
            name = ('k', 'v')[index % 2]
            self.assertIs(mesh, self.cache.mesh)
            self.assertIs(active, self.active[index // 2][name])
            self.assertIs(delta, self.projected[name])
            self.assertIs(spare, self.spare[index // 2][name])
            self.assertEqual(shape, dict(history_rows=2048, prefix=16))
        self.assertEqual(self.synchronizations, [self.cache.mesh])
        self.assertIs(publication, self.cache.pending)
        self.assertEqual(vars(publication), dict(position=4096, prefix=16, rows=2048, status='prepared'))
        self.assertIs(self.cache.active, self.active)
        self.assertIs(self.cache.spare, self.spare)
        self.assertFalse(metadata['hardware_qualified'])

    def test_transport_failure_does_not_publish_or_swap(self):
        def fail(*args, **kwargs):
            def execute():
                raise RuntimeError('transport failed')
            return execute
        prepare, unused = build_prepare(self.source, self.namespace, fail)
        with self.assertRaisesRegex(RuntimeError, 'transport failed'):
            prepare(self.cache, object(), 16, position=4096)
        self.assertIsNone(self.cache.pending)
        self.assertIs(self.cache.active, self.active)
        self.assertIs(self.cache.spare, self.spare)

    def test_native_frontier_guard_remains(self):
        prepare, unused = build_prepare(self.source, self.namespace, self.transport)
        with self.assertRaises(ValueError):
            prepare(self.cache, object(), 16, position=4097)
        self.assertEqual(self.copies, [])

    def test_changed_copy_chain_is_not_silently_adapted(self):
        with self.assertRaises(ValueError):
            build_prepare(self.source.replace('operations.copy(padded, spare[name])', 'operations.copy(other, spare[name])'),
                self.namespace, self.transport)
