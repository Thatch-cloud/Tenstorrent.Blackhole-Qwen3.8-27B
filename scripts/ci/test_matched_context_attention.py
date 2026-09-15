from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matched_context_attention as candidate
from matched_context_geometry import CONTEXTS, geometry


class ContextAttentionTests(unittest.TestCase):
    def test_full_storage_padding_and_unchanged_kernel_configuration(self):
        for context in CONTEXTS:
            with self.subTest(context=context):
                plan, pads, owned = geometry(context), [], []
                def pad(tensor, padding, fill):
                    shape = tuple(size + before + after for size, (before, after) in zip(tensor.shape, padding))
                    pads.append((padding, fill))
                    return SimpleNamespace(shape=shape)
                operations = SimpleNamespace(pad=pad)
                query = SimpleNamespace(shape=(1, 16, 32, 128))
                key = SimpleNamespace(shape=(1, 4, plan['storage_keys'], 128))
                mask = SimpleNamespace(shape=(1, 1, 32, plan['storage_keys']))
                with patch.object(candidate, 'validate_inputs') as validate, \
                        patch.object(candidate, 'execute_folded', return_value='output') as execute:
                    result = candidate.adapter(context)(operations, object(), query, key, key, mask, owned)
                self.assertEqual(result, 'output')
                self.assertEqual(validate.call_args.args[5:], (plan['capacity'], 15, True))
                self.assertEqual(execute.call_args.kwargs, dict(key_chunk_size=256,
                    max_cores_per_head=8, stripe_keys=False, fp32_dest_acc=True))
                self.assertEqual([value.shape[2] for value in owned[:2]], [plan['native_keys']] * 2)
                self.assertEqual(owned[2].shape[3], plan['native_keys'])
                self.assertEqual([fill for padding, fill in pads], [8192., -8192., float('-inf')])

    def test_wrong_storage_fails_before_kernel(self):
        tensor = SimpleNamespace(shape=(1, 4, 32, 128))
        with patch.object(candidate, 'validate_inputs'), patch.object(candidate, 'execute_folded') as execute:
            with self.assertRaisesRegex(ValueError, 'Complete context'):
                candidate.adapter(4096)(object(), object(), tensor, tensor, tensor, tensor, [])
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
