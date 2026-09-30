from types import SimpleNamespace
import unittest

from serving_cache_owner import ServingCacheOwner


class CacheOwnerTests(unittest.TestCase):
    def fixture(self):
        caches = [[SimpleNamespace(shape=(128, 2, 64, 256), dtype='bf8', addresses=(index * 2 + side + 1, index * 2 + side + 100))
            for side in range(2)] for index in range(16)]
        layers = [SimpleNamespace(is_full_attention=True, attention=SimpleNamespace(
            paged_k=pair[0], paged_v=pair[1], use_paged=True)) for pair in caches]
        model = SimpleNamespace(num_devices=2, args=SimpleNamespace(max_batch_size=8),
            _paged_kv_caches=caches, layers=layers)
        runner = SimpleNamespace(model=SimpleNamespace(model=[model]), kv_caches=caches)
        operations = SimpleNamespace(bfloat8_b='bf8', get_device_tensors=lambda tensor: [
            SimpleNamespace(buffer_address=lambda value=value: value) for value in tensor.addresses])
        return operations, runner, model

    def test_identical_serving_and_target_cache_admitted(self):
        owner = ServingCacheOwner(*self.fixture())
        self.assertEqual(owner.physical_pages, 128)
        owner.validate()

    def test_wrong_precision_or_shape_rejected(self):
        for field, value in (('dtype', 'bf16'), ('shape', (128, 4, 64, 256))):
            arguments = self.fixture()
            setattr(arguments[2]._paged_kv_caches[0][0], field, value)
            with self.assertRaises(ValueError):
                ServingCacheOwner(*arguments)

    def test_rebound_attention_or_serving_cache_rejected(self):
        operations, runner, model = self.fixture()
        model.layers[0].attention.paged_k = model._paged_kv_caches[1][0]
        with self.assertRaises(ValueError):
            ServingCacheOwner(operations, runner, model)

    def test_address_mutation_after_capture_rejected(self):
        arguments = self.fixture()
        owner = ServingCacheOwner(*arguments)
        arguments[2]._paged_kv_caches[0][0].addresses = (999, 999)
        with self.assertRaises(ValueError):
            owner.validate()
