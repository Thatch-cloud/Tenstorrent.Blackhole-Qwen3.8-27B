from types import SimpleNamespace
import unittest

from dspark_history import TensorScope, join_rows as original
from history_concat_lifetime import join_rows


class Operations:
    DRAM_MEMORY_CONFIG = 'dram'

    def __init__(self):
        self.values, self.peak = [], 0

    def tensor(self, rows):
        address = len(self.values) + 1
        value = SimpleNamespace(rows=tuple(rows), released=False,
            parts=tuple(SimpleNamespace(buffer_address=lambda chip=chip: address + 10000 * chip)
                for chip in (0, 1)))
        self.values.append(value)
        self.peak = max(self.peak, sum(len(value.rows) for value in self.values if not value.released))
        return value

    def get_device_tensors(self, value):
        if value.released:
            raise ValueError('Use after release')
        return value.parts

    def concat(self, values, **kwargs):
        for value in values:
            self.get_device_tensors(value)
        return self.tensor(row for value in values for row in value.rows)

    def deallocate(self, value):
        self.get_device_tensors(value)
        value.released = True


class ConcatLifetimeTests(unittest.TestCase):
    def execute(self, concatenate):
        operations = Operations()
        scope = TensorScope(operations, [])
        values = [scope.retain(operations.tensor([index])) for index in range(128)]
        output = concatenate(operations, values, scope.retain)
        self.assertEqual(output.rows, tuple(range(128)))
        scope.release(keep=[output])
        self.assertEqual([value for value in operations.values if not value.released], [output])
        return operations.peak

    def test_order_exact_and_peak_live_rows_lower(self):
        self.assertLess(self.execute(join_rows), self.execute(original))

    def test_borrowed_inputs_survive(self):
        operations = Operations()
        borrowed = operations.tensor([0])
        scope = TensorScope(operations, [borrowed])
        output = join_rows(operations, [borrowed, scope.retain(operations.tensor([1]))], scope.retain)
        scope.release(keep=[output])
        self.assertFalse(borrowed.released)
        self.assertEqual(output.rows, (0, 1))

    def test_duplicate_inputs_rejected_before_release(self):
        operations = Operations()
        scope = TensorScope(operations, [])
        value = scope.retain(operations.tensor([0]))
        with self.assertRaises(ValueError):
            join_rows(operations, [value, value], scope.retain)
        self.assertFalse(value.released)
        scope.release()

    def test_failed_concat_leaves_owned_inputs_for_scope_cleanup(self):
        operations = Operations()
        scope = TensorScope(operations, [])
        values = [scope.retain(operations.tensor([index])) for index in range(9)]
        concatenate = operations.concat
        calls = []
        def failing(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('concat')
            return concatenate(*args, **kwargs)
        operations.concat = failing
        with self.assertRaisesRegex(RuntimeError, 'concat'):
            join_rows(operations, values, scope.retain)
        scope.release()
        self.assertTrue(all(value.released for value in operations.values))


if __name__ == '__main__':
    unittest.main()
