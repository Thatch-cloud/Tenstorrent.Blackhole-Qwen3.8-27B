from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gdn_records import retain_checkpoint_histories


class RetainedOwnershipTests(unittest.TestCase):
    def fixture(self):
        tensors = [SimpleNamespace(binding=(index, index + 100)) for index in range(8)]
        result = dict(packed_checkpoints=True, states=tensors[0], packed_conv_states=tensors[1:5], owned=tensors[:])
        return tensors, result

    def test_retains_only_histories_and_transfers_output_to_caller(self):
        tensors, result = self.fixture()
        with patch('gdn_records.addresses', side_effect=lambda operations, tensor: tensor.binding), patch('gdn_records.release_owned') as release:
            retain_checkpoint_histories(None, result, tensors[5])
        self.assertEqual(result['owned'], tensors[:5])
        release.assert_called_once_with(None, tensors[6:])

    def test_alias_view_of_output_is_not_freed(self):
        tensors, result = self.fixture()
        result['owned'][5] = SimpleNamespace(binding=tensors[5].binding)
        with patch('gdn_records.addresses', side_effect=lambda operations, tensor: tensor.binding), patch('gdn_records.release_owned') as release:
            retain_checkpoint_histories(None, result, tensors[5])
        release.assert_called_once_with(None, tensors[6:])

    def test_missing_history_or_partial_alias_fails_before_any_free(self):
        for failure in ('missing', 'partial', 'output-alias'):
            tensors, result = self.fixture()
            if failure == 'missing':
                result['owned'].pop(0)
            elif failure == 'partial':
                tensors[7].binding = (0, 999)
            else:
                tensors[5].binding = tensors[0].binding
            with patch('gdn_records.addresses', side_effect=lambda operations, tensor: tensor.binding), patch('gdn_records.release_owned') as release:
                with self.assertRaises(ValueError):
                    retain_checkpoint_histories(None, result, tensors[5])
            release.assert_not_called()

    def test_a_packed_result_retains_every_users_histories(self):
        """The combined packed result carries only the first segment's states; retaining
        those alone would free the second user's histories as scratch."""
        tensors = [SimpleNamespace(binding=(index, index + 100)) for index in range(14)]
        pieces = tuple(dict(packed_checkpoints=True, states=tensors[base], packed_conv_states=tensors[base + 1:base + 5],
                            owned=tensors[base:base + 5]) for base in (0, 5))
        result = dict(pieces[0], segments=((0, 16), (16, 32)), segment_results=pieces, owned=tensors[:])
        with patch('gdn_records.addresses', side_effect=lambda operations, tensor: tensor.binding), \
                patch('gdn_records.release_owned') as release:
            retain_checkpoint_histories(None, result, tensors[10])
        self.assertEqual(result['owned'], tensors[:10])
        release.assert_called_once_with(None, tensors[11:])
        # a user whose histories the block does not own is refused before any free
        result = dict(pieces[0], segments=((0, 16), (16, 32)), segment_results=pieces, owned=tensors[:5] + tensors[10:])
        with patch('gdn_records.addresses', side_effect=lambda operations, tensor: tensor.binding), \
                patch('gdn_records.release_owned') as release:
            with self.assertRaises(ValueError):
                retain_checkpoint_histories(None, result, tensors[10])
        release.assert_not_called()
