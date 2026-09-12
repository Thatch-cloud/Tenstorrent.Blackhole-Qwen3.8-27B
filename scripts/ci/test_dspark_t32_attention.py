from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dspark_t32_attention import append_queries, execute
from dspark_t32_inputs import fixed_mask
from test_dspark_history import HostOperations, require_tensor


class T32AttentionTests(unittest.TestCase):
    def test_append_preserves_history_and_excludes_inactive_query(self):
        operations = HostOperations()
        for position in (32, 4096, 4384):
            history = torch.arange(position).reshape(1, 1, position, 1).expand(1, 4, position, 128).bfloat16().contiguous()
            queries = torch.arange(32).reshape(1, 1, 32, 1).expand(1, 4, 32, 128).bfloat16().contiguous()
            original_history, original_queries = history.clone(), queries.clone()
            owned = []

            def retain(value):
                owned.append(value)
                return value

            with patch('dspark_t32_attention.require_tensor', require_tensor):
                output = append_queries(operations, history, queries, retain, position=position)
            self.assertTrue(torch.equal(output[:, :, :position], history))
            self.assertTrue(torch.equal(output[:, :, position:position + 31], queries[:, :, :31]))
            self.assertTrue(torch.all(output[:, :, position + 31:] == 0))
            self.assertTrue(torch.equal(history, original_history))
            self.assertTrue(torch.equal(queries, original_queries))
            self.assertIs(output, owned[-1])

    def test_native_call_retains_output_and_64_key_chunk(self):
        operations = HostOperations()
        query = torch.zeros(1, 16, 32, 128, dtype=torch.bfloat16)
        keys = torch.zeros(1, 4, 4416, 128, dtype=torch.bfloat16)
        mask = fixed_mask(4096, 4384)
        output = object()
        owned = []
        with patch('dspark_t32_attention.require_tensor', require_tensor), \
                patch('dspark_t32_attention.draft_sdpa', return_value=output) as native:
            actual = execute(operations, SimpleNamespace(), query, keys, keys, mask, owned,
                context_rows=4384, mask_validated=True)
        self.assertIs(actual, output)
        self.assertEqual(owned, [output])
        native.assert_called_once_with(operations, query, keys, keys, mask, key_chunk_size=64)
        with patch('dspark_t32_attention.require_tensor', require_tensor), \
                patch('dspark_t32_attention.draft_sdpa', return_value=output) as native:
            execute(operations, SimpleNamespace(), query, keys, keys, mask, [],
                    context_rows=4384, mask_validated=True, key_chunk_size=32)
        native.assert_called_once_with(operations, query, keys, keys, mask, key_chunk_size=32)

    def test_wrong_width_or_unvalidated_mask_never_dispatches(self):
        with patch('dspark_t32_attention.draft_sdpa') as native:
            for options in (dict(proposals=15, mask_validated=True), dict(proposals=31, mask_validated=False)):
                with self.assertRaises(ValueError):
                    execute(Mock(), object(), None, None, None, None, [], context_rows=4384, **options)
            native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
