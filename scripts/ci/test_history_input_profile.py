import unittest
from types import SimpleNamespace

from history_input_profile import profile_inputs


class InputProfileTests(unittest.TestCase):
    def test_preserves_operations_and_restores_on_failure(self):
        called = []

        def callback(value=None):
            called.append(value)
            return value

        operations = SimpleNamespace(**{name: callback for name in
            ('pad', 'from_torch', 'deallocate', 'get_device_tensors')})
        rotary = SimpleNamespace(tables=callback)
        host = SimpleNamespace(ones=callback, zeros=callback)

        class Arm:
            history = SimpleNamespace(operations=operations, rotary=rotary)

            def publication(self, features, prefix, *, position):
                operations.pad(features)
                rotary.tables(position)
                host.ones(prefix)
                if prefix == 0:
                    raise ValueError('expected failure')
                return operations.from_torch(features)

        records = []
        original = Arm.publication
        with profile_inputs(Arm, host, records):
            self.assertEqual(Arm().publication(7, 2, position=10), 7)
            with self.assertRaisesRegex(ValueError, 'expected failure'):
                Arm().publication(7, 0, position=10)
        self.assertIs(Arm.publication, original)
        self.assertIs(operations.pad, callback)
        self.assertIs(rotary.tables, callback)
        self.assertIs(host.ones, callback)
        self.assertEqual(called, [7, 10, 2, 7, 7, 10, 0])
        self.assertTrue(records[0]['passed'])
        self.assertFalse(records[1]['passed'])
        self.assertEqual([entry['stage'] for entry in records[0]['stages']],
            ['pad', 'rotary_tables', 'host_ones', 'from_torch'])


if __name__ == '__main__':
    unittest.main()
