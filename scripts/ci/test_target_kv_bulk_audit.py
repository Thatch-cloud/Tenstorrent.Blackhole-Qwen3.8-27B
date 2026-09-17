import hashlib
import unittest
from types import SimpleNamespace

import torch

from target_kv_bulk_audit import digest_prefix


def checksum(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def addresses(operations, value):
    return id(value)


class Operations:
    def __init__(self):
        self.created = []
        self.released = []
        self.reads = 0
        self.fail_read = None

    def slice(self, value, start, end):
        if start[0] == 0 and end[0] == value.shape[0]:
            return value
        shards = [shard[start[0]:end[0]] for shard in value.shards]
        result = SimpleNamespace(shape=shards[0].shape, shards=shards)
        self.created.append(result)
        return result

    def get_device_tensors(self, value):
        return value.shards

    def to_torch(self, shard):
        self.reads += 1
        if self.reads == self.fail_read:
            raise RuntimeError('readback failed')
        return shard.clone()

    def deallocate(self, value):
        self.released.append(id(value))


def cache(pages, seed):
    generator = torch.Generator().manual_seed(seed)
    shards = [torch.randn((pages, 2, 64, 4), generator=generator).to(torch.bfloat16)
        for chip in range(2)]
    return SimpleNamespace(shape=shards[0].shape, shards=shards)


def reference(caches, valid):
    result = []
    pages = (valid + 63) // 64
    for value in caches:
        for start in range(0, pages, 64):
            end = min(start + 64, pages)
            for tensor in value.shards:
                shard = tensor[start:end].clone()
                logical = shard.permute(1, 0, 2, 3).reshape(shard.shape[1], -1, shard.shape[3])
                result.append(checksum(logical[:, :min((end - start) * 64, valid - start * 64)]))
    return result


class BulkAuditTests(unittest.TestCase):
    def run_digest(self, operations, caches, valid, **kwargs):
        return digest_prefix(operations, caches, valid, digest=checksum, addresses=addresses, **kwargs)

    def test_exact_order_partial_pages_and_boundaries(self):
        caches = [cache(1040, 1), cache(1040, 2)]
        for valid in (1, 63, 64, 65, 4095, 4096, 4097, 16383, 16384, 16385, 65536, 65553):
            expected = reference(caches, valid)
            for pages in (64, 128, 256):
                with self.subTest(valid=valid, pages=pages):
                    operations = Operations()
                    result = self.run_digest(operations, caches, valid, read_pages=pages)
                    self.assertEqual(result, expected)
                    self.assertEqual(operations.released, [id(value) for value in operations.created])

    def test_reduces_readbacks_fourfold_without_changing_digests(self):
        caches = [cache(1040, 4)]
        legacy, bulk = Operations(), Operations()
        evidence = {}
        self.assertEqual(self.run_digest(legacy, caches, 65536, read_pages=64),
            self.run_digest(bulk, caches, 65536, evidence=evidence))
        self.assertEqual((legacy.reads, bulk.reads), (32, 8))
        self.assertEqual(evidence['digest_groups'], 16)
        self.assertEqual(evidence['peak_host_tensor_bytes'], 256 * 2 * 64 * 4 * 2)

    def test_mutations_both_shards_and_prefix_exclusion(self):
        value = cache(66, 5)
        before = self.run_digest(Operations(), [value], 4097)
        for chip in range(2):
            original = value.shards[chip][64, 0, 0, 0].clone()
            value.shards[chip][64, 0, 0, 0] = 999
            self.assertNotEqual(self.run_digest(Operations(), [value], 4097), before)
            value.shards[chip][64, 0, 0, 0] = original
            value.shards[chip][64, 0, 1, 0] = 999
        self.assertEqual(self.run_digest(Operations(), [value], 4097), before)

    def test_release_on_failure_and_no_borrowed_deallocation(self):
        operations = Operations()
        operations.fail_read = 2
        with self.assertRaisesRegex(RuntimeError, 'readback failed'):
            self.run_digest(operations, [cache(300, 6)], 16384)
        self.assertEqual(operations.released, [id(value) for value in operations.created])
        borrowed = Operations()
        self.run_digest(borrowed, [cache(1, 7)], 64)
        self.assertEqual(borrowed.released, [])

    def test_rejects_bad_shapes_prefixes_and_missing_shard(self):
        value = cache(1, 8)
        for valid in (True, 0, 65):
            with self.assertRaises(ValueError):
                self.run_digest(Operations(), [value], valid)
        with self.assertRaises(ValueError):
            self.run_digest(Operations(), [value], 64, read_pages=1024)
        value.shards.pop()
        with self.assertRaises(AssertionError):
            self.run_digest(Operations(), [value], 64)
