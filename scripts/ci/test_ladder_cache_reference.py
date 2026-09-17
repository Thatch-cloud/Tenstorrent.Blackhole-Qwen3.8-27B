from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from ladder_cache_reference import snapshot


def tensor(parts, addresses):
    return SimpleNamespace(shape=parts[0].shape,
        parts=[SimpleNamespace(host=value, buffer_address=lambda address=address: address)
            for value, address in zip(parts, addresses, strict=True)])


class CacheReferenceTests(unittest.TestCase):
    def fixture(self):
        initial = torch.zeros(20, 2, 64, 32, dtype=torch.bfloat16)
        values = [initial.clone(), initial.clone()]
        for chip, value in enumerate(values):
            value[3:5] = chip + 1
            value[8] = chip + 3
        cache = tensor(values, (1, 2))
        operations = SimpleNamespace(get_device_tensors=lambda value: value.parts,
            to_torch=lambda part: part.host.clone(), deallocate=Mock())
        operations.slice = Mock(side_effect=lambda value, start, end:
            tensor([part.host[start[0]:end[0]] for part in value.parts], (3, 4)))
        return operations, cache, initial, values

    def test_all_candidate_pages_have_an_exact_expected_value(self):
        operations, cache, initial, values = self.fixture()
        expected = snapshot(operations, cache, initial, (3, 4, 8))
        self.assertTrue(all(torch.equal(actual, reference) for actual, reference in zip(expected, values)))
        self.assertEqual(operations.slice.call_count, 2)
        self.assertEqual(operations.deallocate.call_count, 2)
        candidate = values[0].clone()
        candidate[19, 0, 0, 0] = 1
        self.assertFalse(torch.equal(candidate, expected[0]))

    def test_partial_readback_failure_releases_temporary(self):
        operations, cache, initial, unused = self.fixture()
        operations.to_torch = Mock(side_effect=RuntimeError('readback'))
        with self.assertRaisesRegex(RuntimeError, 'readback'):
            snapshot(operations, cache, initial, (3, 4, 8))
        operations.deallocate.assert_called_once()

    def test_nonzero_initial_state_and_invalid_page_rejected(self):
        operations, cache, initial, unused = self.fixture()
        for pages in ((), (-1,), (20,), (True,)):
            with self.assertRaises(ValueError):
                snapshot(operations, cache, initial, pages)
        initial[0, 0, 0, 0] = 1
        with self.assertRaises(ValueError):
            snapshot(operations, cache, initial, (3,))
        operations.slice.assert_not_called()


if __name__ == '__main__':
    unittest.main()
