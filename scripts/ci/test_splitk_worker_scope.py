import unittest
from types import SimpleNamespace

from splitk_worker_scope import worker_scope


class WorkerScopeTests(unittest.TestCase):
    def test_only_worker_limit_changes_and_restores(self):
        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return 'result'

        module, records = SimpleNamespace(execute_folded=original), []
        inputs = (object(), object(), object())
        options = dict(key_chunk_size=256, max_cores_per_head=8, stripe_keys=False, fp32_dest_acc=True)
        with worker_scope(module, records):
            self.assertEqual(module.execute_folded(*inputs, **options), 'result')
        self.assertIs(module.execute_folded, original)
        self.assertEqual(calls, [(inputs, dict(options, max_cores_per_head=16))])
        self.assertEqual(options['max_cores_per_head'], 8)
        self.assertEqual(records[0]['selected_worker_limit'], 16)

    def test_unmatched_configuration_is_rejected(self):
        module = SimpleNamespace(execute_folded=lambda *args, **kwargs: self.fail('Must not execute'))
        with worker_scope(module, []):
            with self.assertRaises(ValueError):
                module.execute_folded(key_chunk_size=128, max_cores_per_head=8,
                    stripe_keys=False, fp32_dest_acc=True)


if __name__ == '__main__':
    unittest.main()
