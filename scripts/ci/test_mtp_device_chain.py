from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from mtp_device_chain import MTPDeviceChain, feedback_embedding


class MTPDeviceChainTests(unittest.TestCase):
    def test_feedback_does_not_own_a_view_of_the_borrowed_seed(self):
        identifiers = SimpleNamespace(shape=(1, 1, 1), dtype='uint32', layout='row')
        indices = SimpleNamespace(shape=(1, 1))
        embedded = SimpleNamespace(shape=(1, 1, 2560))
        local = SimpleNamespace(shape=(1, 1, 1, 2560))
        gathered = SimpleNamespace(shape=(1, 1, 1, 5120))
        result = SimpleNamespace(shape=(1, 1, 1, 5120))
        operations = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram',
            reshape=Mock(side_effect=[indices, local, result]))
        embed, gather, owned = Mock(return_value=embedded), Mock(return_value=gathered), []
        with patch('mtp_device_chain.addresses', return_value=(100, 200)):
            self.assertIs(feedback_embedding(operations, embed, gather, identifiers, owned), result)
        self.assertFalse(any(value is indices or value is identifiers for value in owned))
        embed.assert_called_once_with(indices, memory_config='dram')
        gather.assert_called_once_with(local)
        self.assertEqual(len(owned), 4)

    def test_partial_alias_is_rejected_before_embedding(self):
        identifiers = SimpleNamespace(shape=(1, 1, 1, 1), dtype='uint32', layout='row')
        operations = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row', reshape=Mock())
        embed = Mock()
        with patch('mtp_device_chain.addresses', side_effect=[(1, 2), (1, 3)]), self.assertRaises(ValueError):
            feedback_embedding(operations, embed, Mock(), identifiers, [])
        embed.assert_not_called()

    def fixture(self):
        chain = MTPDeviceChain.__new__(MTPDeviceChain)
        chain.ready, chain.closed, chain.mesh = True, False, 'mesh'
        chain.inputs = dict(seed=object(), hidden=object())
        chain.metadata = [{name: object() for name in ('positions', 'cosine', 'sine')} for _ in range(7)]
        chain.traces = {count: count + 100 for count in range(1, 8)}
        chain.outputs = {count: torch.arange(count, dtype=torch.int32).reshape(1, 1, 1, count) for count in range(1, 8)}
        chain.step = SimpleNamespace(pages=SimpleNamespace(shape=(1, 1024)),
            model=SimpleNamespace(args=SimpleNamespace(rope_head_dim=64, rope_theta=10000)))
        chain.operations = SimpleNamespace(uint32='uint32', int32='int32', bfloat16='bf16', ROW_MAJOR_LAYOUT='row',
            TILE_LAYOUT='tile', ReplicateTensorToMesh=Mock(return_value='mapper'),
            from_torch=Mock(side_effect=lambda value, **kwargs: value), copy_host_to_device_tensor=Mock(),
            copy=Mock(), execute_trace=Mock(), get_device_tensors=Mock(side_effect=lambda value: [value, value]),
            to_torch=Mock(side_effect=lambda value: value))
        return chain

    def test_every_bounded_count_uses_one_trace_and_one_host_readback(self):
        for count in range(1, 8):
            chain = self.fixture()
            hidden = SimpleNamespace(shape=(1, 1, 1, 5120))
            self.assertEqual(chain(248319, hidden, 256, count), tuple(range(count)))
            chain.operations.execute_trace.assert_called_once_with('mesh', 100 + count, cq_id=0, blocking=True)
            chain.operations.to_torch.assert_called_once()
            self.assertEqual(chain.operations.copy_host_to_device_tensor.call_count, 1 + 3 * count)
            staged = chain.operations.copy_host_to_device_tensor.call_args_list
            self.assertEqual(int(staged[1].args[0][0]), 255)
            self.assertEqual(int(staged[1 + 3 * (count - 1)].args[0][0]), 255 + count - 1)
            chain.operations.copy.assert_called_once_with(hidden, chain.inputs['hidden'])

    def test_invalid_chain_request_fails_before_staging(self):
        for token, position, count in ((-1, 1, 1), (0, 0, 1), (0, 65536, 7), (0, 1, 0), (0, 1, True), (0, 1, 8)):
            chain = self.fixture()
            with self.assertRaises(ValueError):
                chain(token, SimpleNamespace(shape=(1, 1, 1, 5120)), position, count)
            chain.operations.from_torch.assert_not_called()
