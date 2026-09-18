from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from serving_page_binding import VerifierPageBinding, validate_initial_capture_pages


class PageBindingTests(unittest.TestCase):
    def test_capture_pages_require_real_warmup_headroom(self):
        blocks = list(range(4, 69))
        pages = torch.tensor([blocks + [4] * 3], dtype=torch.int32)
        validate_initial_capture_pages(pages, blocks, position=4096, output_budget=256)
        with self.assertRaises(ValueError):
            validate_initial_capture_pages(pages, blocks[:-1], position=4096, output_budget=256)
        for column, value in ((64, 4), (67, 99)):
            changed = pages.clone()
            changed[0, column] = value
            with self.assertRaises(ValueError):
                validate_initial_capture_pages(changed, blocks, position=4096, output_budget=256)

    def fixture(self):
        tensors = []

        def tensor(rows, columns):
            result = SimpleNamespace(shape=(rows, columns), identity=(100 + len(tensors), 200 + len(tensors)), data=None)
            tensors.append(result)
            return result

        primary, singleton, replay_pages = tensor(16, 80), tensor(1, 80), tensor(2, 68)
        reader = SimpleNamespace(metadata=[(None, replay_pages, None, None)], audit=None)
        fixture = SimpleNamespace(pages=primary, singleton_pages=singleton, replay_reader=reader,
            grouped_readers=[reader], readers=[reader, SimpleNamespace(pages=[singleton])],
            writers=[SimpleNamespace(pages=[singleton])])

        def shards(value):
            return [SimpleNamespace(buffer_address=lambda address=address: address) for address in value.identity]

        operations = SimpleNamespace(get_device_tensors=shards, int32='int32', ROW_MAJOR_LAYOUT='row-major',
            ReplicateTensorToMesh=lambda mesh: mesh, from_torch=lambda value, **kwargs: value.clone(),
            copy_host_to_device_tensor=Mock(side_effect=lambda host, target: setattr(target, 'data', host.clone())),
            synchronize_device=Mock())
        host = torch.full((1, 80), 4, dtype=torch.int32)
        host[0, :64] = torch.arange(4, 68)
        engine = SimpleNamespace(phase='idle', operations=operations, mesh='mesh', pages=host,
            buckets={16: {'fixture': fixture}})
        binding = VerifierPageBinding(engine, tuple(range(4, 68)), physical_pages=256)
        return binding, engine, tensors

    def test_append_refreshes_every_captured_table_without_replacing_buffers(self):
        binding, engine, tensors = self.fixture()
        identities = [tensor.identity for tensor in tensors]
        blocks = tuple(range(4, 69))
        self.assertTrue(binding.refresh(blocks, position=4096, rows=16))
        self.assertEqual([tensor.identity for tensor in tensors], identities)
        for tensor in tensors:
            for row in tensor.data:
                self.assertEqual(row[:65].tolist(), list(blocks))
                self.assertTrue(torch.all(row[65:] == blocks[0]))
        self.assertEqual(engine.pages[0, :65].tolist(), list(blocks))
        engine.operations.synchronize_device.assert_called_once()
        self.assertFalse(binding.refresh(blocks, position=4112, rows=16))
        self.assertEqual(engine.operations.copy_host_to_device_tensor.call_count, 3)

    def test_remap_duplicate_and_insufficient_reservation_rejected_before_copy(self):
        for blocks in (tuple(range(5, 70)), tuple(range(4, 68)), (*range(4, 68), 4), (*range(4, 68), 256)):
            binding, engine, _ = self.fixture()
            with self.assertRaises(ValueError):
                binding.refresh(blocks, position=4096, rows=16)
            engine.operations.copy_host_to_device_tensor.assert_not_called()

    def test_partial_upload_failure_poisoning_prevents_trace_reuse(self):
        binding, engine, _ = self.fixture()
        engine.operations.copy_host_to_device_tensor.side_effect = [None, RuntimeError('upload failed')]
        with self.assertRaises(RuntimeError):
            binding.refresh(tuple(range(4, 69)), position=4096, rows=16)
        self.assertEqual(engine.phase, 'failed')
        self.assertTrue(binding.failed)
        with self.assertRaises(ValueError):
            binding.refresh(tuple(range(4, 69)), position=4096, rows=16)

    def test_address_change_rejected_even_without_new_pages(self):
        binding, engine, tensors = self.fixture()
        tensors[0].identity = (999, 999)
        with self.assertRaises(ValueError):
            binding.refresh(tuple(range(4, 68)), position=4000, rows=16)
        self.assertEqual(engine.phase, 'failed')
        engine.operations.copy_host_to_device_tensor.assert_not_called()
