"""Op-fusion programme, host-gap package WPH, lever 2: the lean write_packed (QWEN_FAST_WRITE_PACKED_LEAN, audit twin QWEN_FAST_WRITE_PACKED_LEAN_AUDIT;
write_packed_lean.py, bound by verify_prestage.BlockPrestage.writer; default off, host only).

Pinned here:
  - the flags are 0 or 1, the audit needs the lever, and with the flag off the block holds no lean state and writes through packed_verifier.write_packed itself;
  - over a fake device that counts what the host asks of it, the lean write copies the SAME values into the SAME destinations in the SAME order, with the same
    uploads (values, dtype, layout), builds one mapper where the original builds one a value, reads each destination's addresses once the first time and once a round
    (never before a write, and not at all at a verify-time write), and keeps the original's errors: a pair per chip, pairwise distinct, a replaced buffer, the poison;
  - a buffer that moved is found at the next round's check even when the move happened at a verify-time write, and a verify-time write alone reads no address;
  - the audit twin runs the original's before and after reads beside the book's, reads back a rotating eight from every chip, and on a difference writes the destination
    again with the original function and latches the lever off, as it does for an exception in its own bookkeeping before the first copy;
  - on the real four-user block (and with PRESTAGE_DIFF) the device state and the predictions are the flag-off run's."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import packed_verifier
import prestage_diff
import test_packed_verifier as tpv
import test_verify_prestage as tvp
import verify_prestage
import write_packed_lean as lean

HERE = Path(__file__).resolve().parent
LEAN, AUDIT, DIFF = 'QWEN_FAST_WRITE_PACKED_LEAN', 'QWEN_FAST_WRITE_PACKED_LEAN_AUDIT', 'QWEN_FAST_PRESTAGE_DIFF'


class CountingTTNN(tpv.FakeTTNN):
    """The fake device, counting the address reads (get_device_tensors) and the mappers built."""

    def __init__(self):
        super().__init__()
        self.address_reads = 0
        self.mappers = 0
        self.uploads = []

    def get_device_tensors(self, tensor):
        self.address_reads += 1
        return super().get_device_tensors(tensor)

    def ReplicateTensorToMesh(self, mesh):
        self.mappers += 1
        return ('replicate', self.mappers)

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        self.uploads.append((value.clone(), dtype, layout))
        return super().from_torch(value, device=device, dtype=dtype, layout=layout, memory_config=memory_config, mesh_mapper=mesh_mapper)


def model():
    return SimpleNamespace(mesh_device=SimpleNamespace(name='mesh'))


def values_of(ttnn, count=6):
    destinations = [ttnn.allocate((4,), 'int32', 'row_major', torch.zeros(4, dtype=torch.int32)) for _ in range(count)]
    return [(destination, torch.arange(4, dtype=torch.int32) + 10 * index, 'int32', 'row_major') for index, destination in enumerate(destinations)]


class Clean(unittest.TestCase):
    def setUp(self):
        lean.reset()
        self.addCleanup(lean.reset)
        self.log = []
        patcher = patch.object(verify_prestage, 'log_line', side_effect=self.log.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (LEAN, AUDIT):
            os.environ.pop(name, None)


class FlagTests(Clean):
    def test_the_flags_are_zero_or_one_and_the_audit_needs_the_lever(self):
        self.assertFalse(lean.enabled({}))
        self.assertTrue(lean.enabled({LEAN: '1'}))
        self.assertFalse(lean.audit_enabled({AUDIT: '1'}))
        self.assertTrue(lean.audit_enabled({LEAN: '1', AUDIT: '1'}))
        for name in (LEAN, AUDIT):
            with self.assertRaises(ValueError):
                lean._flag(name, {name: 'on'})

    def test_engage_returns_the_config_only_for_the_lever_and_a_misset_flag_fails_the_attach(self):
        self.assertIsNone(lean.engage(None, {LEAN: '0', AUDIT: '1'}))
        self.assertFalse(lean.engage(None, {LEAN: '1'}).audit)
        self.assertTrue(lean.engage(None, {LEAN: '1', AUDIT: '1'}).audit)
        with patch.dict(os.environ, {LEAN: 'maybe'}), self.assertRaises(ValueError):
            verify_prestage.BlockPrestage(SimpleNamespace(users=4), audit=False)

    def test_off_the_block_writes_through_packed_verifier_write_packed_itself(self):
        made = verify_prestage.BlockPrestage(SimpleNamespace(users=4), audit=False)
        self.assertIsNone(made.lean)
        self.assertIs(made.writer('prestage'), packed_verifier.write_packed)
        with patch.object(packed_verifier, 'write_packed', lambda *args, **kwargs: 'patched'):
            self.assertEqual(made.writer('verify')(), 'patched', 'looked up at the call, as the call sites did')


class WriteTests(Clean):
    def setUp(self):
        super().setUp()
        self.ttnn = CountingTTNN()
        self.model = model()
        self.readers = [SimpleNamespace(failed=False)]

    def original(self, values, **options):
        ttnn = CountingTTNN()
        mirrored = [(ttnn.allocate(destination.shape, destination.dtype, destination.layout, destination.value.clone()), value, dtype, layout)
                    for destination, value, dtype, layout in values]
        packed_verifier.write_packed(ttnn, self.model, mirrored, self.readers, **options)
        return ttnn, mirrored

    def lean_write(self, values, site='prestage', **options):
        return lean.write(self.ttnn, self.model, values, self.readers, site=site, **options)

    def test_the_same_copies_uploads_and_order_as_the_original_with_fewer_reads_and_one_mapper(self):
        values = values_of(self.ttnn)
        expected, mirrored = self.original(values)
        self.lean_write(values)
        self.assertEqual([(host.value.tolist(), id(destination) is not None) for host, destination in self.ttnn.host_copies],
                         [(host.value.tolist(), True) for host, destination in expected.host_copies])
        self.assertTrue(all(torch.equal(own[0].value, mirror[0].value) for own, mirror in zip(values, mirrored)), 'every destination holds what the original left')
        self.assertEqual([(item[0].tolist(), item[1], item[2]) for item in self.ttnn.uploads], [(item[0].tolist(), item[1], item[2]) for item in expected.uploads])
        self.assertEqual((expected.mappers, self.ttnn.mappers), (len(values), 1))
        self.assertEqual(expected.address_reads, 2 * len(values))
        self.assertEqual(self.ttnn.address_reads, 2 * len(values), 'the first call captures once and checks once')
        before = self.ttnn.address_reads
        self.lean_write(values[:3])
        self.assertEqual(self.ttnn.address_reads - before, 3, 'a later pre-stage call: the after-check alone, of what it wrote')
        self.assertEqual(self.ttnn.mappers, 1)
        before = self.ttnn.address_reads
        self.lean_write(values[:2], site='verify')
        self.assertEqual(self.ttnn.address_reads - before, 0, 'a verify-time write reads no address at all')
        self.assertTrue(any('engaged site=prestage' in line for line in self.log) and any('engaged site=verify' in line for line in self.log))

    def test_the_fence_and_the_returned_host_tensors(self):
        values = values_of(self.ttnn, 3)
        synced = self.ttnn.synchronized
        staged = self.lean_write(values, fence=False)
        self.assertEqual((self.ttnn.synchronized - synced, len(staged)), (0, 3))
        staged = self.lean_write(values, fence=True)
        self.assertEqual((self.ttnn.synchronized - synced, len(staged)), (1, 3))
        self.assertEqual(self.lean_write(values, indices=[2, 0]) and len(self.ttnn.host_copies), 8)

    def test_indices_write_only_those_in_that_order(self):
        values = values_of(self.ttnn, 5)
        self.lean_write(values, indices=[3, 1])
        self.assertEqual([destination for host, destination in self.ttnn.host_copies], [values[3][0], values[1][0]])

    def test_a_pair_per_chip_and_distinct_buffers_are_the_originals_errors(self):
        values = values_of(self.ttnn, 3)
        values[1][0].shards[0].address = values[0][0].shards[0].address
        with self.assertRaises(ValueError) as raised:
            self.lean_write(values)
        with self.assertRaises(ValueError) as expected:
            packed_verifier.write_packed(self.ttnn, self.model, values, self.readers)
        self.assertEqual(str(raised.exception), str(expected.exception))
        self.assertEqual(self.ttnn.host_copies, [], 'nothing was copied')
        lean.reset()
        short = values_of(self.ttnn, 2)
        short[0][0].shards = short[0][0].shards[:1]
        with self.assertRaises(ValueError) as raised:
            self.lean_write(short)
        with self.assertRaises(ValueError) as expected:
            packed_verifier.write_packed(self.ttnn, self.model, short, self.readers)
        self.assertEqual(str(raised.exception), str(expected.exception), 'the address reader refuses a short shard list, here and in the original')

    def test_a_moved_buffer_is_found_at_the_round_check_and_a_verify_time_move_at_the_next_one(self):
        values = values_of(self.ttnn, 4)
        self.lean_write(values)
        values[2][0].shards[1].address += 4096
        with self.assertRaises(AssertionError) as raised:
            self.lean_write(values[2:3])
        self.assertEqual(str(raised.exception), lean.REPLACED)
        lean.reset()
        values = values_of(self.ttnn, 4)
        self.lean_write(values)
        self.lean_write(values[1:2], site='verify')           # written at the verify
        values[1][0].shards[0].address += 4096                 # ... moved there (a copy that replaced the buffer)
        with self.assertRaises(AssertionError):
            self.lean_write(values[3:4])                        # the next window finds it though it does not write it

    def test_a_failed_copy_poisons_the_readers_unless_asked_not_to(self):
        values = values_of(self.ttnn, 2)

        def broken(host, destination):
            raise RuntimeError('copy failed')

        self.ttnn.copy_host_to_device_tensor = broken
        with self.assertRaises(RuntimeError):
            self.lean_write(values, poison=False)
        self.assertFalse(self.readers[0].failed)
        with self.assertRaises(RuntimeError):
            self.lean_write(values, poison=True)
        self.assertTrue(self.readers[0].failed)

    def test_an_exception_in_the_bookkeeping_before_the_first_copy_latches_and_the_original_does_the_call(self):
        values = values_of(self.ttnn, 3)
        with patch.object(lean, 'mapper_for', side_effect=RuntimeError('x')):
            self.lean_write(values)
        self.assertEqual(lean.BOOK.off, 'prelude:RuntimeError')
        self.assertTrue(any(line.startswith(lean.FELL_BACK_MARKER) for line in self.log))
        self.assertEqual(len(self.ttnn.host_copies), 3, 'the original wrote them once')
        self.lean_write(values)
        self.assertEqual(len(self.ttnn.host_copies), 6)
        self.assertEqual(self.ttnn.mappers, 3 + 3, 'both calls were the original: a mapper a value')


class PerBlockTests(Clean):
    def test_each_block_has_its_own_book_and_a_rebuilt_block_may_reuse_a_dead_blocks_addresses(self):
        ttnn = CountingTTNN()
        first = lean.engage(None, {LEAN: '1'})
        second = lean.engage(None, {LEAN: '1'})
        self.assertIsNot(first.book, second.book)
        values = values_of(ttnn, 3)
        readers = [SimpleNamespace(failed=False)]
        first.writer('prestage')(ttnn, model(), values, readers)
        self.assertEqual(len(first.book.captured), 3)
        self.assertEqual(second.book.captured, {})
        # the first block is closed and a new one is built over buffers the allocator hands out again at the same addresses
        reborn = lean.engage(None, {LEAN: '1'})
        again = values_of(ttnn, 3)
        for old, new in zip(values, again):
            new[0].shards = old[0].shards
        reborn.writer('prestage')(ttnn, model(), again, readers)
        self.assertEqual(len(reborn.book.captured), 3, 'no distinctness clash with the dead block, no stale comparison')
        self.assertIsNone(reborn.book.off)

    def test_one_blocks_latch_does_not_stop_the_other(self):
        ttnn = CountingTTNN()
        first, second = lean.engage(None, {LEAN: '1'}), lean.engage(None, {LEAN: '1'})
        lean.latch('x', first.book)
        self.assertIsNotNone(first.book.off)
        self.assertIsNone(second.book.off)
        values = values_of(ttnn, 2)
        second.writer('verify')(ttnn, model(), values, [])
        self.assertEqual(ttnn.mappers, 1)


class AuditTests(Clean):
    def setUp(self):
        super().setUp()
        self.ttnn = CountingTTNN()
        self.model = model()
        self.readers = []

    def test_the_audit_passes_on_a_clean_write_and_reads_back_from_every_chip(self):
        values = values_of(self.ttnn, 12)
        lean.write(self.ttnn, self.model, values, self.readers, site='prestage', audit=True)
        lines = [line for line in self.log if line.startswith(lean.AUDIT_MARKER)]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r'^\[PINDIAG\] tp4 write packed lean audit calls=1 checked=8 destinations=12 mismatches=0 exact=True$')
        lean.write(self.ttnn, self.model, values, self.readers, site='verify', audit=True)
        self.assertEqual(len([line for line in self.log if line.startswith(lean.AUDIT_MARKER)]), 2)
        self.assertIsNone(lean.BOOK.off)

    def test_a_copy_that_lands_wrong_is_found_rewritten_by_the_original_and_latches_the_lever(self):
        values = values_of(self.ttnn, 4)
        real = self.ttnn.copy_host_to_device_tensor
        wrong = {'once': True}

        def corrupting(host, destination):
            real(host, destination)
            if destination is values[2][0] and wrong.pop('once', False):
                destination.value = destination.value + 1

        self.ttnn.copy_host_to_device_tensor = corrupting
        lean.write(self.ttnn, self.model, values, self.readers, site='prestage', audit=True)
        mismatches = [line for line in self.log if line.startswith(lean.MISMATCH_MARKER)]
        self.assertEqual(len(mismatches), 1)
        self.assertIn('exact=False', mismatches[0])
        self.assertEqual(lean.BOOK.off, 'audit_mismatch')
        self.assertTrue(torch.equal(values[2][0].value, values[2][1]), 'written again by the original')
        before = len(self.ttnn.host_copies)
        lean.write(self.ttnn, self.model, values, self.readers, site='prestage', audit=True)
        self.assertEqual(len(self.ttnn.host_copies) - before, 4)
        self.assertEqual(self.ttnn.mappers, 1 + 1 + 4, 'the lean call, the one destination the original wrote again, then every write is the original: a mapper a value')

    def test_the_audit_compares_the_books_addresses_with_the_originals_before_read(self):
        values = values_of(self.ttnn, 3)
        lean.write(self.ttnn, self.model, values, self.readers, site='prestage', audit=True)
        lean.BOOK.captured[id(values[1][0])] = (values[1][0], tuple(address + 1 for address in lean.BOOK.captured[id(values[1][0])][1]))
        lean.write(self.ttnn, self.model, values, self.readers, site='prestage', audit=True)
        self.assertTrue(any(line.startswith(lean.MISMATCH_MARKER) and 'captured_addresses_differ' in line for line in self.log))
        self.assertEqual(lean.BOOK.off, 'audit_mismatch')


class RealBlockTests(tvp.PrestageFixture):
    """The four-user block over the fake device model: the lean writer, alone and with the diff lever, leaves the device and the predictions as the flag-off run."""

    def setUp(self):
        super().setUp()
        lean.reset()
        prestage_diff.reset()
        self.addCleanup(lean.reset)
        self.addCleanup(prestage_diff.reset)
        for name in (LEAN, AUDIT, DIFF, 'QWEN_FAST_PRESTAGE_DIFF_AUDIT'):
            os.environ.pop(name, None)

    def rounds(self, block, steps=5):
        users, served = tvp.base_users(), []
        for number in range(steps):
            if number:
                self.window(block, users)
            predictions, metrics, staged = self.round(block, users, 3 * number)
            if number:
                self.assert_device_holds(block, staged)
            served.append(predictions)
            users = tvp.advanced(users, 2 + number)
        return served

    def test_the_predictions_and_the_device_are_the_flag_off_runs_alone_and_with_the_diff_lever(self):
        off = self.rounds(self.open_block())
        for flags in ({LEAN: '1'}, {LEAN: '1', AUDIT: '1'}, {LEAN: '1', DIFF: '1'}, {LEAN: '1', AUDIT: '1', DIFF: '1', 'QWEN_FAST_PRESTAGE_DIFF_AUDIT': '1'}):
            with self.subTest(flags=flags):
                lean.reset()
                prestage_diff.reset()
                self.h1a.clear()
                os.environ.update(flags)
                block = self.open_block()
                self.assertIsNotNone(block.prestaged.lean)
                self.assertEqual(self.rounds(block), off)
                self.assertTrue(any(line.startswith(lean.ENGAGED_MARKER + ' site=prestage') for line in self.h1a))
                self.assertTrue(any(line.startswith(lean.ENGAGED_MARKER + ' site=verify') for line in self.h1a))
                self.assertIsNone(block.prestaged.lean.book.off, self.h1a)
                for name in flags:
                    os.environ.pop(name)

    def test_one_mapper_for_the_block_and_the_address_reads_fall(self):
        reads = {}
        for name, flags in (('off', {}), ('lean', {LEAN: '1'})):
            lean.reset()
            os.environ.update(flags)
            block = self.open_block()
            users = tvp.base_users()
            self.round(block, users)
            users = tvp.advanced(users, 2)
            calls = []
            base = self.ttnn.get_device_tensors
            self.ttnn.get_device_tensors = lambda tensor, base=base: calls.append(1) or base(tensor)
            self.window(block, users)
            self.round(block, users)
            self.ttnn.get_device_tensors = base
            reads[name] = len(calls)
            for key in flags:
                os.environ.pop(key)
        self.assertLess(reads['lean'], reads['off'])


if __name__ == '__main__':
    unittest.main()
