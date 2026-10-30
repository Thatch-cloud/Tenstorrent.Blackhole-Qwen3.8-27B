"""The host side of the peer-to-peer KV restore (qwen_p2p_restore; docs/tp4-p2p-restore.md), on fakes: a regular file for the drive, ordinary memory for the cards' staging buffers and
their p2pmem mappings, a fake driver behind the ioctls, a fake ttnn for the device ops. No card, no drive, no root.

Held:
  * the checksum is the one INTERFACE section 4 fixes: three shared vectors with their expected values (scripts/ci/p2p_checksum_vectors.json), a pure-python reference, chunked passes;
  * the chip-major layout: the same bytes in another order, a chip's share one aligned run, round trips, per-chip checksums;
  * the window arithmetic equals the probe's (scripts/p2p/p2p_probe.py make_plan), and refuses what does not hold together; the ioctl numbers and structs equal the probe's;
  * a TtLink over a fake driver: the ioctls it sends (power, allocate until the published window comes up, configure, pin), the canary, the mapping, the cleanup;
  * the engine and its stream: reads of every card's shard into the cards' staging halves, the checksum compared with the stored one, a mismatch writing nothing, a read error, a
    posted write trailing the drive, positions without a stored object, the cap on a chunk, the halves reused only after a device sync.
Run alone: python3 -B -m unittest test_qwen_p2p_restore (python 3.11+, numpy)
"""

import ctypes
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_p2p_restore as p2p  # noqa: E402


def load_probe():
    spec = importlib.util.spec_from_file_location('p2p_probe_for_tests', str(HERE.parent / 'p2p' / 'p2p_probe.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_checksum(data):
    """INTERFACE section 4, one word at a time, in plain python integers."""
    total = 0
    for k in range(len(data) // 4):
        x = int.from_bytes(data[4 * k:4 * k + 4], 'little')
        total = (total + ((k * 2654435761 + 1) % (1 << 32)) * x) % (1 << 64)
    return total


# ================================================================================================
# Fakes the tier tests use too
# ================================================================================================
class FakeMesh(object):
    """The device-op extension and `ttnn.synchronize_device`, in memory: a staging buffer per chip (bytearray), the ops the engine calls, and a log of them. `sink(tensor, chip, block, bytes)`
    is where a write op puts a row (the fake device's KV cache). `drift` makes the first checksum of a range answer wrongly (a posted write that has not landed yet)."""

    def __init__(self, layout, sink=None, address=1 << 30, bad_checksum=None, missing=(), bdfs=None):
        self.layout = layout
        self.bdfs = list(bdfs) if bdfs else ['0000:%02x:00.0' % (0x50 + chip) for chip in range(layout.chips)]
        self.sink = sink or (lambda tensor, chip, block, data: None)
        self.address = address
        self.log = []
        self.memory = []
        self.size = 0
        self.drift = 0                     # checksum calls that answer wrongly before the first right one
        self.bad_checksum = bad_checksum   # a function (chip, index, value) -> value
        self.fail_write = False
        self.fail_checksum = False
        for name in missing:
            setattr(self, '_missing_' + name, True)

    def __getattr__(self, name):
        if name.startswith('_missing_'):
            raise AttributeError(name)
        raise AttributeError(name)

    def qwen_p2p_staging_alloc(self, mesh, size):
        self.size = size
        self.memory = [bytearray(size) for _ in range(self.layout.chips)]
        return 'staging-handle'

    def qwen_p2p_staging_info(self, handle):
        return [dict(device_id=chip, pci_bus_id=self.bdfs[chip], dram_bank=0, noc_x=17, noc_y=12, address=self.address, size=self.size) for chip in range(self.layout.chips)]

    def qwen_staging_checksum(self, handle, byte_offset, length, count=1):
        self.log.append(('checksum', byte_offset, length, count))
        if self.fail_checksum:
            raise RuntimeError('the checksum program failed')
        out = []
        for chip in range(self.layout.chips):
            row = [p2p.p2p_checksum(self.memory[chip][byte_offset + index * length:byte_offset + (index + 1) * length]) for index in range(count)]
            if self.drift > 0:
                row = [value ^ 1 for value in row]
            if self.bad_checksum is not None:
                row = [self.bad_checksum(chip, index, value) for index, value in enumerate(row)]
            out.append(row)
        if self.drift > 0:
            self.drift -= 1
        return out

    def qwen_write_blocks_from_staging(self, tensor, handle, byte_offset, blocks, row_stride=None):
        self.log.append(('write', tensor, byte_offset, tuple(blocks), row_stride))
        if self.fail_write:
            raise RuntimeError('the copy program failed')
        for chip in range(self.layout.chips):
            for index, block in enumerate(blocks):
                start = byte_offset + index * row_stride
                self.sink(tensor, chip, block, bytes(self.memory[chip][start:start + self.layout.slice_bytes]))

    def synchronize_device(self, mesh):
        self.log.append(('sync',))

    def ops(self, kind):
        return [entry for entry in self.log if entry[0] == kind]


class FakeLink(object):
    """A card's end of the path: its staging bytes are the mesh's, so the 'drive DMA' (a read into view()) is what the device ops then see."""

    created = []

    def __init__(self, mesh, position, index, bdf, staging, ordering):
        self.memory = mesh.memory[position]
        self.position, self.index, self.bdf, self.staging, self.ordering = position, index, bdf, staging, ordering
        self.map_verified = True
        self.window_id = 202
        self.closed = False
        self.selftests = 0

    def view(self, staging_offset, length):
        return memoryview(self.memory)[staging_offset:staging_offset + length]

    def selftest(self):
        self.selftests += 1

    def close(self):
        self.closed = True


def make_links(mesh):
    links = []

    def factory(position, index, bdf, staging, ordering):
        link = FakeLink(mesh, position, index, bdf, staging, ordering)
        links.append(link)
        return link

    return factory, links


class FileDrive(object):
    """A regular file standing in for the namespace: chip-major objects back to back, entries naming where they are and carrying their checksums."""

    def __init__(self, directory, layout, blocks, seed=1):
        self.layout = layout
        self.path = os.path.join(directory, 'drive.img')
        rng = np.random.default_rng(seed)
        self.blocks = [rng.integers(0, 256, layout.block_bytes, dtype=np.uint8).tobytes() for _ in range(blocks)]
        self.entries = []
        with open(self.path, 'wb') as handle:
            for index, payload in enumerate(self.blocks):
                obj = bytearray(layout.block_bytes)
                layout.to_chipmajor(payload, obj)
                handle.write(obj)
                self.entries.append(types.SimpleNamespace(offset=index * layout.block_bytes, nbytes=layout.block_bytes, aux=layout.pack_checksums(layout.checksums(obj)),
                                                          digest=b'', key=b'k%d' % index))
        self.fd = os.open(self.path, os.O_RDWR)
        self.reads = []
        self.fail_at = None
        self.lock = threading.Lock()

    def reader(self, offset, view):
        with self.lock:
            self.reads.append((offset, len(view)))
        if self.fail_at is not None and self.fail_at(offset):
            raise OSError(5, 'Input/output error')
        got = os.preadv(self.fd, [view], offset)
        if got != len(view):
            raise IOError('short read')

    def flip(self, block, chip, byte=0):
        """One byte of one chip's shard of one block changes on the 'drive'."""
        position = block * self.layout.block_bytes + chip * self.layout.shard_bytes + byte
        old = os.pread(self.fd, 1, position)
        os.pwrite(self.fd, bytes([old[0] ^ 0xFF]), position)

    def close(self):
        os.close(self.fd)


class Device(object):
    """The fake KV cache the write ops fill: block -> bytes in the host tier's layout ([cache][chip][slice])."""

    def __init__(self, layout):
        self.layout = layout
        self.blocks = {}

    def sink(self, tensor, chip, block, data):
        layout = self.layout
        buffer = self.blocks.setdefault(block, bytearray(layout.block_bytes))
        start = (tensor * layout.chips + chip) * layout.slice_bytes
        buffer[start:start + layout.slice_bytes] = data


SMALL = dict(caches=2, chips=4, slice_bytes=1024)          # a shard is 2,048 bytes: engine tests pass io_align=1


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='p2p-test-')
        self.addCleanup(shutil.rmtree, self.directory, True)


# ================================================================================================
# The checksum
# ================================================================================================
class ChecksumTests(unittest.TestCase):
    def test_the_three_shared_vectors_have_their_expected_values(self):
        document = json.loads((HERE / 'p2p_checksum_vectors.json').read_text())
        vectors = dict(p2p.checksum_vectors())
        self.assertEqual(sorted(vectors), sorted(item['name'] for item in document['vectors']))
        for item in document['vectors']:
            data = vectors[item['name']]
            self.assertEqual(len(data), item['length'], item['name'])
            self.assertEqual(p2p.p2p_checksum(data), int(item['checksum']), item['name'])

    def test_the_vectors_are_what_interface_section_4_says(self):
        vectors = dict(p2p.checksum_vectors())
        self.assertEqual(vectors['zero_4096'], bytes(4096))
        self.assertEqual(vectors['mod251_4096'], bytes(i % 251 for i in range(4096)))
        self.assertEqual(vectors['prng_1mib'], np.random.default_rng(20261030).integers(0, 256, 1 << 20, dtype=np.uint8).tobytes())
        self.assertEqual(p2p.p2p_checksum(bytes(4096)), 0)

    def test_it_equals_a_plain_python_reference(self):
        rng = np.random.default_rng(7)
        for size in (4, 8, 4096, 65536 + 4):
            data = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
            self.assertEqual(p2p.p2p_checksum(data), reference_checksum(data), size)
        self.assertEqual(p2p.p2p_checksum(dict(p2p.checksum_vectors())['mod251_4096']), reference_checksum(dict(p2p.checksum_vectors())['mod251_4096']))

    def test_the_passes_over_a_long_range_add_up(self):
        rng = np.random.default_rng(8)
        data = rng.integers(0, 256, 40000, dtype=np.uint8).tobytes()
        whole = p2p.p2p_checksum(data)
        with mock.patch.object(p2p, 'CHECKSUM_BLOCK_WORDS', 1000), mock.patch.dict(p2p._weights, clear=True):
            self.assertEqual(p2p.p2p_checksum(data), whole)
            self.assertEqual(p2p.p2p_checksum(data), reference_checksum(data))

    def test_every_buffer_kind_gives_the_same_answer(self):
        rng = np.random.default_rng(9)
        array = rng.integers(0, 256, 8192, dtype=np.uint8)
        want = p2p.p2p_checksum(array.tobytes())
        for data in (array, bytearray(array.tobytes()), memoryview(array.tobytes()), memoryview(bytearray(array.tobytes())), array.view(np.uint32)):
            self.assertEqual(p2p.p2p_checksum(data), want)

    def test_a_length_that_is_not_whole_words_is_refused(self):
        with self.assertRaises(ValueError):
            p2p.p2p_checksum(b'123')

    def test_a_changed_byte_changes_it_wherever_it_is(self):
        data = bytearray(np.random.default_rng(10).integers(0, 256, 4096, dtype=np.uint8).tobytes())
        base = p2p.p2p_checksum(data)
        for position in (0, 1, 3, 4, 2047, 4095):
            changed = bytearray(data)
            changed[position] ^= 0x40
            self.assertNotEqual(p2p.p2p_checksum(changed), base, position)

    def test_it_counts_from_the_start_of_the_range(self):
        data = np.random.default_rng(11).integers(0, 256, 8192, dtype=np.uint8).tobytes()
        self.assertEqual(p2p.p2p_checksum(data[4096:]), reference_checksum(data[4096:]))
        self.assertNotEqual(p2p.p2p_checksum(data[4096:]), p2p.p2p_checksum(data[:4096]))


# ================================================================================================
# The layout
# ================================================================================================
class LayoutTests(unittest.TestCase):
    PRODUCTION = dict(caches=32, chips=4, slice_bytes=17408)

    def test_the_production_geometry_is_aligned_per_chip(self):
        layout = p2p.ShardLayout(**self.PRODUCTION)
        self.assertEqual((layout.shard_bytes, layout.block_bytes), (557056, 2228224))
        self.assertEqual(layout.shard_bytes, 136 * 4096)
        self.assertTrue(layout.aligned())
        self.assertEqual(layout.aux_bytes, 32)

    def test_the_host_layout_is_not_aligned_per_chip(self):
        # (why the drive holds the blocks chip-major: a chip's slice of one tensor is 4.25 pages)
        self.assertNotEqual(17408 % 4096, 0)

    def test_chip_major_is_the_same_bytes_in_another_order(self):
        layout = p2p.ShardLayout(**self.PRODUCTION)
        payload = np.random.default_rng(1).integers(0, 256, layout.block_bytes, dtype=np.uint8)
        obj = bytearray(layout.block_bytes)
        layout.to_chipmajor(payload, obj)
        self.assertEqual(sorted(payload.tobytes()), sorted(bytes(obj)))
        cube = payload.reshape(32, 4, 17408)
        for chip in range(4):
            shard = np.frombuffer(layout.shard(obj, chip), dtype=np.uint8)
            self.assertEqual(shard.tobytes(), cube[:, chip, :].tobytes(), chip)
        back = bytearray(layout.block_bytes)
        layout.from_chipmajor(obj, back)
        self.assertEqual(bytes(back), payload.tobytes())

    def test_checksums_are_per_chip_and_pack_both_ways(self):
        layout = p2p.ShardLayout(**SMALL)
        obj = bytearray(np.random.default_rng(2).integers(0, 256, layout.block_bytes, dtype=np.uint8).tobytes())
        sums = layout.checksums(obj)
        self.assertEqual(len(sums), 4)
        self.assertEqual(sums[2], p2p.p2p_checksum(layout.shard(obj, 2)))
        packed = layout.pack_checksums(sums)
        self.assertEqual(len(packed), layout.aux_bytes)
        self.assertEqual(layout.unpack_checksums(packed), sums)
        for bad in (b'', None, packed[:-1], packed + b'x'):
            with self.assertRaises(ValueError):
                layout.unpack_checksums(bad)
        with self.assertRaises(ValueError):
            layout.pack_checksums(sums[:3])

    def test_a_block_of_the_wrong_size_is_refused(self):
        layout = p2p.ShardLayout(**SMALL)
        with self.assertRaises(ValueError):
            layout.to_chipmajor(bytes(10), bytearray(layout.block_bytes))
        with self.assertRaises(ValueError):
            p2p.ShardLayout(0, 4, 1024)

    def test_of_io_reads_the_models_tier_io(self):
        io = types.SimpleNamespace(caches=['a', 'b'], chips=4, slice_bytes=1024, block_bytes=8192)
        layout = p2p.ShardLayout.of_io(io)
        self.assertEqual((layout.caches, layout.chips, layout.slice_bytes), (2, 4, 1024))
        self.assertIsNone(p2p.ShardLayout.of_io(types.SimpleNamespace(block_bytes=32)))
        with self.assertRaises(ValueError):
            p2p.ShardLayout.of_io(types.SimpleNamespace(caches=['a', 'b'], chips=4, slice_bytes=1024, block_bytes=1))


# ================================================================================================
# The window arithmetic
# ================================================================================================
class AimTests(unittest.TestCase):
    GIB = 1 << 30

    def info(self, **kw):
        return dict(dict(bar=4, offset=0, size=4 * self.GIB, tlb_first=202, tlb_last=202, bus_addr=0x38000000000), **kw)

    def staging(self, **kw):
        return dict(dict(noc_x=17, noc_y=12, address=self.GIB, size=128 << 20), **kw)

    def test_it_agrees_with_the_probes_plan(self):
        probe = load_probe()
        for info, staging in ((self.info(), self.staging()),
                              (self.info(offset=self.GIB, size=2 * self.GIB), self.staging(address=self.GIB + (256 << 20))),
                              (self.info(tlb_first=203, tlb_last=203, offset=4 * self.GIB), self.staging(address=self.GIB // 2, noc_x=18, noc_y=15))):
            aim = p2p.plan_aim(info, staging)
            window_base = (info['tlb_first'] - 202) * 4 * self.GIB
            # the probe is told where in the slice to write and what NOC address the window starts at; here both come from the staging address
            slice_offset = aim['slice_offset']
            plan = probe.make_plan(info, staging['size'], slice_offset, 1 << 20, 0, staging['noc_x'], staging['noc_y'], aim['noc']['addr'], 0)
            self.assertEqual(plan['dest']['window_offset'], aim['window_offset'])
            self.assertEqual(plan['dest']['noc_addr'], staging['address'])
            self.assertEqual(plan['window'], aim['window'])
            self.assertEqual((plan['noc']['x'], plan['noc']['y'], plan['noc']['addr']), (aim['noc']['x'], aim['noc']['y'], aim['noc']['addr']))
            self.assertEqual(aim['window_offset'], staging['address'] - aim['noc']['addr'])
            self.assertEqual(info['offset'] - window_base + slice_offset, aim['window_offset'])

    def test_a_staging_buffer_in_the_published_slice_maps_byte_for_byte(self):
        aim = p2p.plan_aim(self.info(offset=self.GIB, size=2 * self.GIB), self.staging(address=2 * self.GIB, size=64 << 20))
        self.assertEqual((aim['slice_offset'], aim['window_offset'], aim['noc']['addr']), (self.GIB, 2 * self.GIB, 0))
        self.assertEqual(aim['noc']['ordering'], 0)
        self.assertEqual(p2p.plan_aim(self.info(), self.staging(), ordering='posted')['noc']['ordering'], 2)

    def test_refusals(self):
        cases = [
            (self.info(bar=0), self.staging(), 'BAR0'),
            (self.info(tlb_first=202, tlb_last=203), self.staging(), 'one 4 GiB window'),
            (self.info(tlb_first=-1, tlb_last=-1), self.staging(), 'one 4 GiB window'),
            (self.info(tlb_first=100, tlb_last=100), self.staging(), 'not one of the 4 GiB windows'),
            (self.info(), self.staging(noc_x=64), '6-bit'),
            (self.info(), self.staging(noc_y=-1), '6-bit'),
            (self.info(), self.staging(size=100), 'whole number of pages'),
            (self.info(), self.staging(address=4 * self.GIB - (32 << 20), size=128 << 20), 'crosses a 4 GiB boundary'),
            (self.info(offset=2 * self.GIB, size=self.GIB), self.staging(address=self.GIB), 'outside the published slice'),
            (self.info(size=self.GIB), self.staging(address=self.GIB, size=4096), 'outside the published slice'),
            (self.info(), self.staging(address=self.GIB + 512, size=4096), 'page-aligned'),
        ]
        for info, staging, why in cases:
            with self.assertRaisesRegex(p2p.P2PUnavailable, why):
                p2p.plan_aim(info, staging)

    def test_the_ioctl_numbers_and_structs_equal_the_probes(self):
        probe = load_probe()
        for name in ('IOCTL_ALLOCATE_TLB', 'IOCTL_FREE_TLB', 'IOCTL_CONFIGURE_TLB', 'IOCTL_SET_POWER_STATE', 'IOCTL_EXPORT_TLB_DMABUF', 'BUSY_FLAGS', 'BUSY_VALIDITY', 'FIRST_4G_TLB',
                     'WINDOW_4G', 'NOC_COORDINATE_LIMIT', 'ORDERINGS'):
            self.assertEqual(getattr(p2p, name), getattr(probe, name), name)
        for name in ('ALLOCATE_IN', 'ALLOCATE_OUT', 'CONFIGURE_IN', 'POWER_STATE', 'EXPORT_ARG'):
            self.assertEqual(getattr(p2p, name).format, getattr(probe, name).format, name)


# ================================================================================================
# Cards, found at run time
# ================================================================================================
def build_sysfs(root, cards, slice_size=16 << 20, available=None, with_p2pmem=True):
    """A fake /sys/class/tenstorrent and /sys/bus/pci/devices for `cards` = [(index, bdf)]."""
    sysfs_class = os.path.join(root, 'class')
    sysfs_pci = os.path.join(root, 'pci')
    for index, bdf in cards:
        pci = os.path.join(sysfs_pci, bdf)
        os.makedirs(os.path.join(pci, 'p2pmem'), exist_ok=True)
        cls = os.path.join(sysfs_class, 'tenstorrent!%d' % index)
        os.makedirs(cls, exist_ok=True)
        os.symlink(pci, os.path.join(cls, 'device'))
        with open(os.path.join(pci, 'resource'), 'w') as handle:
            handle.write(''.join('0x%016x 0x%016x 0x0000000000000000\n' % (0x1000 * (n + 1), 0x1000 * (n + 1) + 0xFFF) for n in range(7)))
        if with_p2pmem:
            for name, value in (('size', slice_size), ('available', slice_size if available is None else available)):
                with open(os.path.join(pci, 'p2pmem', name), 'w') as handle:
                    handle.write('%d\n' % value)
            os.makedirs(os.path.join(cls, 'p2pmem'), exist_ok=True)
            for name, value in (('bar', 4), ('offset', 0), ('size', slice_size), ('bus_addr', '0x38000000000'), ('tlb_first', 202), ('tlb_last', 202)):
                with open(os.path.join(cls, 'p2pmem', name), 'w') as handle:
                    handle.write('%s\n' % value)
    return sysfs_class, sysfs_pci


class SysfsTests(TmpCase):
    CARDS = [(2, '0000:41:00.0'), (0, '0000:42:00.0'), (3, '0000:4a:00.0'), (1, '0000:4c:00.0')]

    def test_cards_are_found_by_their_pci_address_not_by_a_remembered_number(self):
        sysfs_class, _ = build_sysfs(self.directory, self.CARDS)
        self.assertEqual(p2p.card_table(sysfs_class), {'0000:41:00.0': 2, '0000:42:00.0': 0, '0000:4a:00.0': 3, '0000:4c:00.0': 1})
        infos = [{'pci_bus_id': '0000:42:00.0'}, {'pci_bus_id': '4c:00.0'}, {'pci_bus_id': '0000:41:00.0'}, {'pci_bus_id': '0000:4A:00.0'}]
        self.assertEqual(p2p.locate_cards(infos, sysfs_class), [(0, '0000:42:00.0'), (1, '0000:4c:00.0'), (2, '0000:41:00.0'), (3, '0000:4a:00.0')])

    def test_a_record_may_name_the_device_number_instead(self):
        sysfs_class, _ = build_sysfs(self.directory, self.CARDS)
        self.assertEqual(p2p.locate_cards([{'pcie_device_id': 3}, {'tt_device_index': 0}], sysfs_class), [(3, '0000:4a:00.0'), (0, '0000:42:00.0')])

    def test_a_record_that_names_no_card_or_an_unknown_one_is_refused(self):
        sysfs_class, _ = build_sysfs(self.directory, self.CARDS)
        for infos, why in (([{'device_id': 0}], 'names neither'), ([{'pci_bus_id': '0000:aa:00.0'}], 'does not list'), ([{'pcie_device_id': 9}], 'does not list'),
                           ([{'pci_bus_id': 'zz'}], 'not a PCI address'), ([{'pci_bus_id': '41:00.0'}, {'pcie_device_id': 2}], 'same card')):
            with self.assertRaisesRegex(p2p.P2PUnavailable, why):
                p2p.locate_cards(infos, sysfs_class)

    def test_the_published_slice_is_read_from_both_sysfs_places(self):
        sysfs_class, sysfs_pci = build_sysfs(self.directory, self.CARDS[:1])
        info = p2p.read_p2pmem(2, '0000:41:00.0', sysfs_class, sysfs_pci)
        self.assertEqual((info['bar'], info['offset'], info['size'], info['tlb_first'], info['tlb_last'], info['core_size'], info['core_available']), (4, 0, 16 << 20, 202, 202, 16 << 20, 16 << 20))
        self.assertEqual(info['bus_addr'], 0x38000000000)

    def test_a_card_without_the_patched_module_is_unavailable(self):
        sysfs_class, sysfs_pci = build_sysfs(self.directory, self.CARDS[:1], with_p2pmem=False)
        with self.assertRaisesRegex(p2p.P2PUnavailable, 'patched tt-kmd'):
            p2p.read_p2pmem(2, '0000:41:00.0', sysfs_class, sysfs_pci)

    def test_pci_addresses_normalise(self):
        self.assertEqual(p2p.normalise_bdf('41:00.0'), '0000:41:00.0')
        self.assertEqual(p2p.normalise_bdf(' 0000:4A:00.0 '), '0000:4a:00.0')
        with self.assertRaises(ValueError):
            p2p.normalise_bdf('8:0.0')


# ================================================================================================
# A real link, over a fake driver
# ================================================================================================
class FakeDriver(object):
    """What the kernel does with the ioctls, as far as the link can tell: the windows it hands out (lowest free first), what was configured, the pins, the power state."""

    def __init__(self, held=(), refuse=()):
        self.held = set(held)
        self.refuse = set(refuse)
        self.windows = {}
        self.configured = {}
        self.pins = []
        self.freed = []
        self.power = None
        self.calls = []
        self.opened = []

    def ioctl(self, fd, request, buffer, mutate=True):
        self.calls.append(request)
        if request in self.refuse:
            raise OSError(22, 'Invalid argument')
        if request == p2p.IOCTL_SET_POWER_STATE:
            argsz, flags, _r, validity, power_flags = struct.unpack_from('<IIBBH', buffer, 0)
            self.power = (argsz, validity, power_flags)
        elif request == p2p.IOCTL_ALLOCATE_TLB:
            size, _ = p2p.ALLOCATE_IN.unpack_from(buffer, 0)
            assert size == p2p.WINDOW_4G
            window = 202
            while window in self.held or window in self.windows:
                window += 1
            self.windows[window] = True
            struct.pack_into('<IIQQQ', buffer, p2p.ALLOCATE_IN.size, window, 0, 0, 0, 0)
        elif request == p2p.IOCTL_FREE_TLB:
            window = struct.unpack_from('<I', buffer, 0)[0]
            self.windows.pop(window, None)
            self.freed.append(window)
        elif request == p2p.IOCTL_CONFIGURE_TLB:
            fields = p2p.CONFIGURE_IN.unpack_from(buffer, 0)
            window, _r, addr, x, y = fields[:5]
            ordering = fields[9]
            self.configured[window] = dict(addr=addr, x=x, y=y, ordering=ordering)
        elif request == p2p.IOCTL_EXPORT_TLB_DMABUF:
            argsz, flags, tlb_id, _fd, offset, size = p2p.EXPORT_ARG.unpack_from(buffer, 0)
            fd2 = os.open(os.devnull, os.O_RDONLY)
            self.pins.append((tlb_id, fd2))
            p2p.EXPORT_ARG.pack_into(buffer, 0, argsz, flags, tlb_id, fd2, offset, size)
        else:
            raise OSError(25, 'Inappropriate ioctl')
        return 0


class TtLinkTests(TmpCase):
    BDF = '0000:41:00.0'
    SLICE = 16 << 20

    def setUp(self):
        TmpCase.setUp(self)
        self.sysfs_class, self.sysfs_pci = build_sysfs(self.directory, [(2, self.BDF)], slice_size=self.SLICE)
        self.dev_dir = os.path.join(self.directory, 'dev')
        os.makedirs(self.dev_dir)
        # the card's window and its published slice are the same bytes: one file, reached by the device node and by p2pmem/allocate
        self.backing = os.path.join(self.directory, 'bar4.bin')
        with open(self.backing, 'wb') as handle:
            handle.truncate(self.SLICE)
        os.symlink(self.backing, os.path.join(self.dev_dir, '2'))
        os.symlink(self.backing, os.path.join(self.sysfs_pci, self.BDF, 'p2pmem', 'allocate'))
        self.driver = FakeDriver()
        patcher = mock.patch.object(p2p.fcntl, 'ioctl', self.driver.ioctl)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(p2p.TtLink, 'cpu_window_bytes', self.SLICE)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.links = []

    def open(self, address=4 << 20, size=2 << 20, **kwargs):
        staging = dict(noc_x=17, noc_y=12, address=address, size=size)
        link = p2p.TtLink.open(2, self.BDF, staging, settle=0, sysfs_class=self.sysfs_class, sysfs_pci=self.sysfs_pci, dev_dir=self.dev_dir, **kwargs)
        self.addCleanup(link.close)
        self.links.append(link)
        return link

    def test_the_link_takes_aims_pins_and_proves_the_window(self):
        link = self.open()
        self.assertEqual(self.driver.calls[0], p2p.IOCTL_SET_POWER_STATE)
        self.assertEqual(self.driver.power, (p2p.POWER_STATE.size, p2p.BUSY_VALIDITY, p2p.BUSY_FLAGS))
        self.assertEqual(self.driver.configured, {202: dict(addr=0, x=17, y=12, ordering=0)})
        self.assertEqual([window for window, _ in self.driver.pins], [202])
        order = [call for call in self.driver.calls if call != p2p.IOCTL_FREE_TLB]
        self.assertEqual(order, [p2p.IOCTL_SET_POWER_STATE, p2p.IOCTL_ALLOCATE_TLB, p2p.IOCTL_CONFIGURE_TLB, p2p.IOCTL_EXPORT_TLB_DMABUF])
        self.assertEqual(link.window_id, 202)
        self.assertEqual((link.aim['slice_offset'], link.aim['window_offset']), (4 << 20, 4 << 20))

    def test_it_takes_windows_until_the_published_one_comes_up_and_gives_the_others_back(self):
        # the slice is published in window 203 (BAR4 offset 4 GiB): the driver hands out 202 first
        with open(os.path.join(self.sysfs_class, 'tenstorrent!2', 'p2pmem', 'tlb_first'), 'w') as handle:
            handle.write('203\n')
        with open(os.path.join(self.sysfs_class, 'tenstorrent!2', 'p2pmem', 'tlb_last'), 'w') as handle:
            handle.write('203\n')
        with open(os.path.join(self.sysfs_class, 'tenstorrent!2', 'p2pmem', 'offset'), 'w') as handle:
            handle.write('%d\n' % (4 << 30))
        link = self.open()
        self.assertEqual(link.window_id, 203)
        self.assertEqual(self.driver.freed, [202])
        self.assertEqual(list(self.driver.configured), [203])

    def test_a_published_window_another_process_holds_is_unavailable_and_nothing_is_left_taken(self):
        self.driver.held = {202}
        with self.assertRaisesRegex(p2p.P2PUnavailable, 'held by another process'):
            self.open()
        self.assertEqual(self.driver.windows, {})                      # (203 was taken to find out whose 202 is, and given back)
        self.assertIn(203, self.driver.freed)

    def test_a_driver_that_refuses_a_step_is_unavailable_with_the_step_named(self):
        for request, what in ((p2p.IOCTL_SET_POWER_STATE, 'SET_POWER_STATE'), (p2p.IOCTL_ALLOCATE_TLB, 'ALLOCATE_TLB'), (p2p.IOCTL_CONFIGURE_TLB, 'CONFIGURE_TLB'),
                              (p2p.IOCTL_EXPORT_TLB_DMABUF, 'cannot be pinned')):
            self.driver = FakeDriver(refuse={request})
            with mock.patch.object(p2p.fcntl, 'ioctl', self.driver.ioctl):
                with self.assertRaisesRegex(p2p.P2PUnavailable, what):
                    self.open()

    def test_a_slice_somebody_already_allocated_is_the_leases_refusal(self):
        build_sysfs(self.directory, [], slice_size=self.SLICE)
        with open(os.path.join(self.sysfs_pci, self.BDF, 'p2pmem', 'available'), 'w') as handle:
            handle.write('%d\n' % (self.SLICE - 4096))
        with self.assertRaisesRegex(p2p.P2PUnavailable, 'already allocated'):
            self.open()

    def test_a_window_that_does_not_store_fails_the_canary_before_any_mapping_of_the_slice(self):
        class Dead(object):
            def __getitem__(self, key):
                return bytes(key.stop - key.start)

            def __setitem__(self, key, value):
                pass

            def close(self):
                pass

        with mock.patch.object(p2p.TtLink, '_map_cpu_window', lambda link: setattr(link, 'cpu', Dead())):
            with mock.patch.object(p2p.TtLink, '_map_p2pmem', side_effect=AssertionError('the slice was mapped before the canary')):
                with self.assertRaisesRegex(p2p.P2PUnavailable, 'TLB is not pointing at memory'):
                    self.open()

    def test_what_the_drive_writes_into_the_slice_is_what_the_window_shows(self):
        link = self.open(address=4 << 20, size=1 << 20)
        data = os.urandom(8192)
        view = link.view(4096, 8192)
        view[:] = data
        view.release()
        self.assertEqual(bytes(link.cpu[(4 << 20) + 4096:(4 << 20) + 4096 + 8192]), data)
        self.assertFalse(link.map_verified)                              # (an ordinary file has no BAR to check against)

    def test_the_mapping_check_refuses_a_mapping_that_is_not_where_the_window_points(self):
        link_cls = p2p.TtLink
        with mock.patch.object(p2p, '_pagemap_physical', lambda address, pagemap='': 0x1000 + 0x5000):
            # bar4 start in the fake resource file is 0x5000 (line 4); the first page should be at bar4 + slice offset (0) = 0x5000, but offset 0 reads 0x6000 here
            with self.assertRaisesRegex(p2p.P2PUnavailable, 'would not land where the window points'):
                self.open()
        self.assertTrue(link_cls)

    def test_the_mapping_check_passes_when_the_kernel_says_where_the_pages_are(self):
        bar4 = 0x5000

        def physical(address, pagemap=''):
            return bar4 + (address - physical.base)

        def verify(link):
            physical.base = ctypes.addressof(ctypes.c_char.from_buffer(link.p2p_map))
            return original(link)

        original = p2p.TtLink._verify_mapping
        with mock.patch.object(p2p, '_pagemap_physical', physical), mock.patch.object(p2p.TtLink, '_verify_mapping', verify):
            link = self.open()
        self.assertTrue(link.map_verified)

    def test_close_gives_back_everything_and_is_idempotent(self):
        link = self.open()
        link.close()
        link.close()
        self.assertEqual(self.driver.freed.count(202), 1)
        self.assertTrue(link.closed)
        self.assertIsNone(link.fd)

    def test_a_failed_open_leaves_nothing_behind(self):
        self.driver = FakeDriver(refuse={p2p.IOCTL_EXPORT_TLB_DMABUF})
        with mock.patch.object(p2p.fcntl, 'ioctl', self.driver.ioctl):
            with self.assertRaises(p2p.P2PUnavailable):
                self.open()
        self.assertIn(202, self.driver.freed)

    def test_a_staging_buffer_outside_the_slice_never_touches_the_driver(self):
        with self.assertRaisesRegex(p2p.P2PUnavailable, 'outside the published slice'):
            self.open(address=self.SLICE, size=1 << 20)
        self.assertEqual(self.driver.calls, [])


# ================================================================================================
# The device-op adapter
# ================================================================================================
class StagingApiTests(unittest.TestCase):
    def test_a_ttnn_without_the_extension_is_unavailable_and_says_which_op(self):
        with self.assertRaisesRegex(p2p.P2PUnavailable, 'qwen_staging_checksum'):
            p2p.StagingApi(types.SimpleNamespace(qwen_p2p_staging_alloc=lambda *a: 1, qwen_p2p_staging_info=lambda *a: [], qwen_write_blocks_from_staging=lambda *a: 1), 'mesh', [1])

    def test_checksums_are_read_whatever_container_the_op_returns(self):
        layout = p2p.ShardLayout(**SMALL)
        mesh = FakeMesh(layout)
        api = p2p.StagingApi(mesh, 'mesh', [1, 2])
        api.alloc(1 << 16)
        for returned in ([[1, 2], [3, 4], [5, 6], [7, 8]], np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint64), ((1, 2), (3, 4), (5, 6), (7, 8))):
            mesh.qwen_staging_checksum = lambda *a, returned=returned: returned
            self.assertEqual(api.checksums(0, 2048, 2, 4), [[1, 2], [3, 4], [5, 6], [7, 8]])
        mesh.qwen_staging_checksum = lambda *a: [1, 2, 3, 4]
        self.assertEqual(api.checksums(0, 2048, 1, 4), [[1], [2], [3], [4]])
        mesh.qwen_staging_checksum = lambda *a: [[1, 2]] * 3
        with self.assertRaisesRegex(p2p.P2PDeviceError, 'wanted 4 devices'):
            api.checksums(0, 2048, 2, 4)
        mesh.qwen_staging_checksum = lambda *a: 'nope'
        with self.assertRaises(p2p.P2PDeviceError):
            api.checksums(0, 2048, 1, 4)

    def test_a_raising_op_is_a_device_error_and_a_raising_alloc_is_unavailable(self):
        layout = p2p.ShardLayout(**SMALL)
        mesh = FakeMesh(layout)
        api = p2p.StagingApi(mesh, 'mesh', [1, 2])
        mesh.qwen_p2p_staging_alloc = mock.Mock(side_effect=RuntimeError('no DRAM'))
        with self.assertRaisesRegex(p2p.P2PUnavailable, 'no DRAM'):
            api.alloc(1 << 20)
        mesh.fail_write = mesh.fail_checksum = True
        mesh.qwen_p2p_staging_alloc = FakeMesh.qwen_p2p_staging_alloc.__get__(mesh)
        api.alloc(1 << 16)
        with self.assertRaises(p2p.P2PDeviceError):
            api.write(0, 0, [1], 2048)
        with self.assertRaises(p2p.P2PDeviceError):
            api.checksums(0, 2048, 1, 4)


# ================================================================================================
# The engine and its stream
# ================================================================================================
class EngineCase(TmpCase):
    def build(self, blocks=7, chunk_blocks=3, threads=4, geometry=SMALL, io_align=1, **kwargs):
        self.layout = p2p.ShardLayout(**geometry)
        self.device = Device(self.layout)
        self.mesh = FakeMesh(self.layout, self.device.sink, **kwargs)
        self.drive = FileDrive(self.directory, self.layout, blocks)
        self.addCleanup(self.drive.close)
        factory, self.links = make_links(self.mesh)
        self.api = p2p.StagingApi(self.mesh, 'mesh', list(range(self.layout.caches)))
        self.engine = p2p.P2PEngine(self.api, self.layout, self.drive.reader, threads=threads, staging_bytes=self.layout.shard_bytes * chunk_blocks * 2, link_factory=factory,
                                    io_align=io_align, sleep=lambda seconds: None)
        self.addCleanup(self.engine.close)
        return self.engine

    def restore(self, count=None, first=0, ids=None):
        entries = self.drive.entries[first:first + (count if count is not None else len(self.drive.entries))]
        ids = ids or [500 + index for index in range(len(entries))]
        self.engine.restore(entries, ids)
        return entries, ids

    def exact(self, ids, first=0):
        return all(bytes(self.device.blocks[block]) == self.drive.blocks[first + index] for index, block in enumerate(ids))


class EngineTests(EngineCase):
    def test_a_restore_puts_every_block_into_its_device_block_byte_for_byte(self):
        self.build()
        entries, ids = self.restore()
        self.assertTrue(self.exact(ids))
        self.assertEqual(self.engine.stats['blocks'], 7)
        self.assertEqual(self.engine.stats['chunks'], 3)
        self.assertEqual(self.engine.stats['bytes'], 7 * self.layout.block_bytes)

    def test_the_device_ops_are_what_the_contract_says(self):
        self.build()
        self.restore(count=3)
        checks = self.mesh.ops('checksum')
        self.assertEqual(checks, [('checksum', 0, self.layout.shard_bytes, 3)])
        writes = self.mesh.ops('write')
        self.assertEqual([(w[1], w[2], w[3], w[4]) for w in writes], [(0, 0, (500, 501, 502), self.layout.shard_bytes), (1, self.layout.slice_bytes, (500, 501, 502), self.layout.shard_bytes)])
        self.assertTrue(all(link.selftests == 0 for link in self.links))

    def test_every_card_reads_only_its_own_share_of_every_object(self):
        self.build(blocks=4, chunk_blocks=4)
        self.restore()
        self.assertEqual(len(self.drive.reads), 4 * 4)
        seen = sorted(self.drive.reads)
        want = sorted((block * self.layout.block_bytes + chip * self.layout.shard_bytes, self.layout.shard_bytes) for block in range(4) for chip in range(4))
        self.assertEqual(seen, want)

    def test_the_staging_halves_alternate_and_a_half_is_synced_before_the_drive_writes_into_it_again(self):
        self.build(blocks=8, chunk_blocks=2)
        self.restore()
        kinds = [entry[0] for entry in self.mesh.log if entry[0] in ('write', 'sync')]
        # chunk 0 (half 0) is written, chunk 1 (half 1) is written; chunk 2 reads into half 0 again only after a sync
        self.assertGreaterEqual(kinds.count('sync'), 2)
        offsets = [entry[1] for entry in self.mesh.ops('checksum')]
        self.assertEqual(offsets, [0, self.engine.half_bytes, 0, self.engine.half_bytes])
        self.assertGreater(kinds.index('sync'), 0)

    def test_a_flipped_byte_on_the_drive_is_a_mismatch_and_nothing_is_written_for_its_chunk(self):
        self.build(blocks=7, chunk_blocks=3)
        self.drive.flip(block=4, chip=2, byte=5)
        entries = self.drive.entries
        with self.assertRaises(p2p.P2PMismatch) as caught:
            self.engine.restore(entries, [500 + i for i in range(7)])
        self.assertEqual(caught.exception.bad, [(4, 2)])
        self.assertEqual(caught.exception.entries, [entries[4]])
        written = set(block for entry in self.mesh.ops('write') for block in entry[3])
        self.assertEqual(written, {500, 501, 502})            # chunk 0 only: chunk 1 (blocks 3, 4, 5) was refused whole, chunk 2 was never started
        self.assertNotIn(503, self.device.blocks)
        self.assertEqual(self.engine.stats['mismatches'], 1)
        self.assertEqual(self.engine.stats['mismatch_retries'], 1)

    def test_every_chip_of_every_block_is_checked(self):
        for chip in range(4):
            self.build(blocks=2, chunk_blocks=2)
            self.drive.flip(block=1, chip=chip, byte=self.layout.shard_bytes - 1)
            with self.assertRaises(p2p.P2PMismatch) as caught:
                self.restore()
            self.assertEqual(caught.exception.bad, [(1, chip)])
            self.assertEqual(self.mesh.ops('write'), [])
            self.engine.close()

    def test_a_checksum_that_is_late_once_is_retried_after_a_sync_and_the_restore_goes_through(self):
        self.build(blocks=3, chunk_blocks=3)
        self.mesh.drift = 1
        entries, ids = self.restore()
        self.assertTrue(self.exact(ids))
        self.assertEqual(self.engine.stats['mismatch_retries'], 1)
        self.assertEqual(self.engine.stats['mismatches'], 0)
        self.assertEqual(len(self.mesh.ops('checksum')), 2)

    def test_a_read_error_names_the_chunk_and_stops_without_writing_it(self):
        self.build(blocks=7, chunk_blocks=3)
        self.drive.fail_at = lambda offset: offset // self.layout.block_bytes == 4
        with self.assertRaises(p2p.P2PReadError) as caught:
            self.restore()
        self.assertIn('chunk 1', str(caught.exception))
        written = set(block for entry in self.mesh.ops('write') for block in entry[3])
        self.assertEqual(written, {500, 501, 502})
        self.assertEqual(self.engine.stats['read_errors'], 1)
        # the halves are back: the next restore works
        self.drive.fail_at = None
        entries, ids = self.restore(ids=[700 + i for i in range(7)])
        self.assertTrue(self.exact(ids))

    def test_an_entry_without_checksums_is_an_error_not_a_pass(self):
        self.build(blocks=2, chunk_blocks=2)
        self.drive.entries[1].aux = b''
        with self.assertRaisesRegex(p2p.P2PError, 'checksums'):
            self.restore()
        self.assertEqual(self.mesh.ops('write'), [])

    def test_a_device_op_that_raises_is_a_device_error(self):
        self.build(blocks=2, chunk_blocks=2)
        self.mesh.fail_checksum = True
        with self.assertRaises(p2p.P2PDeviceError):
            self.restore()
        self.mesh.fail_checksum = False
        self.mesh.fail_write = True
        with self.assertRaises(p2p.P2PDeviceError):
            self.restore()

    def test_positions_with_no_stored_object_need_no_read_and_keep_their_place(self):
        self.build(blocks=5, chunk_blocks=2)
        items = [self.drive.entries[0], None, self.drive.entries[1], None, None, self.drive.entries[2], self.drive.entries[3]]
        stream = self.engine.start(items)
        seen = []
        try:
            while True:
                chunk = stream.take(wait=True)
                if chunk is None:
                    break
                seen.append((chunk.first, chunk.count, [position for position, _ in chunk.pending]))
                stream.verify(chunk)
                stream.write(chunk, [900 + position for position, _ in chunk.pending])
                stream.release(chunk)
        finally:
            stream.close()
        self.assertEqual(seen, [(0, 2, [0]), (2, 2, [2]), (4, 2, [5]), (6, 1, [6])])
        for position, entry_index in ((0, 0), (2, 1), (5, 2), (6, 3)):
            self.assertEqual(bytes(self.device.blocks[900 + position]), self.drive.blocks[entry_index])
        self.assertNotIn(901, self.device.blocks)

    def test_a_chunk_of_only_gaps_is_ready_at_once_and_reads_nothing(self):
        self.build(blocks=1, chunk_blocks=2)
        stream = self.engine.start([None, None, self.drive.entries[0]])
        first = stream.take()
        self.assertEqual((first.first, first.count, first.pending), (0, 2, []))
        stream.verify(first)
        stream.write(first, [])
        stream.release(first)
        stream.close()
        self.assertEqual(self.mesh.ops('checksum'), [])

    def test_the_sliced_restores_cap_shrinks_the_chunks(self):
        self.build(blocks=7, chunk_blocks=4)
        stream = self.engine.start(self.drive.entries, 2)
        self.assertEqual([chunk.count for chunk in stream.chunks], [2, 2, 2, 1])
        stream.close()
        stream = self.engine.start(self.drive.entries, 99)
        self.assertEqual([chunk.count for chunk in stream.chunks], [4, 3])
        stream.close()

    def test_take_does_not_wait_unless_asked_and_a_ready_chunk_comes_out_in_order(self):
        self.build(blocks=3, chunk_blocks=1)
        gate = threading.Event()
        original = self.drive.reader

        def slow(offset, view):
            gate.wait(5)
            original(offset, view)

        self.engine.reader = slow
        stream = self.engine.start(self.drive.entries)
        self.assertIsNone(stream.take(wait=False))
        self.assertFalse(stream.ready())
        gate.set()
        chunk = stream.take(wait=True)
        self.assertEqual(chunk.first, 0)
        stream.release(chunk)
        stream.close()

    def test_close_waits_for_the_reads_in_flight(self):
        self.build(blocks=3, chunk_blocks=1)
        gate = threading.Event()
        original = self.drive.reader
        self.engine.reader = lambda offset, view: (gate.wait(5), original(offset, view))
        stream = self.engine.start(self.drive.entries)
        self.assertGreater(self.engine.inflight, 0)
        threading.Timer(0.05, gate.set).start()
        self.assertTrue(stream.close(timeout=5))
        self.assertEqual(self.engine.inflight, 0)

    def test_an_abandoned_stream_gives_its_halves_back(self):
        self.build(blocks=6, chunk_blocks=2)
        stream = self.engine.start(self.drive.entries)
        stream.close()
        self.assertEqual(sorted(self.engine.free_halves), [0, 1])
        entries, ids = self.restore()
        self.assertTrue(self.exact(ids))

    def test_close_does_not_wait_forever_for_a_read_the_drive_never_answers(self):
        self.build(blocks=3, chunk_blocks=1)
        gate = threading.Event()
        self.addCleanup(gate.set)
        original = self.drive.reader
        self.engine.reader = lambda offset, view: (gate.wait(30), original(offset, view))
        self.engine.close_wait_s = 0.1
        stream = self.engine.start(self.drive.entries)
        began = time.monotonic()
        self.engine.close()
        self.assertLess(time.monotonic() - began, 5.0)
        self.assertTrue(self.engine.closed)
        self.assertTrue(all(link.closed for link in self.links))
        gate.set()
        stream.close(timeout=5)

    def test_reads_queued_when_the_engine_closes_are_not_started(self):
        self.build(blocks=8, chunk_blocks=4, threads=1)
        gate = threading.Event()
        self.addCleanup(gate.set)
        started = []
        original = self.drive.reader
        self.engine.reader = lambda offset, view: (started.append(offset), gate.wait(30), original(offset, view))
        stream = self.engine.start(self.drive.entries)
        time.sleep(0.05)
        self.engine.close_wait_s = 0.05
        self.engine.close()
        gate.set()
        time.sleep(0.1)
        self.assertEqual(len(started), 1, 'one read was in flight; the rest were queued and did not start')
        stream.close(timeout=5)

    def test_a_second_stream_shares_the_halves_with_the_first(self):
        self.build(blocks=4, chunk_blocks=2)
        first = self.engine.start(self.drive.entries[:2])
        second = self.engine.start(self.drive.entries[2:])
        a = first.take(wait=True)
        b = second.take(wait=True)
        self.assertNotEqual(a.half, b.half)
        for stream, chunk, ids in ((first, a, [10, 11]), (second, b, [12, 13])):
            stream.verify(chunk)
            stream.write(chunk, ids)
            stream.release(chunk)
            stream.close()
        self.assertTrue(self.exact([10, 11, 12, 13]))

    def test_the_real_geometry_with_real_alignment(self):
        self.build(blocks=5, chunk_blocks=2, geometry=LayoutTests.PRODUCTION, io_align=4096)
        entries, ids = self.restore()
        self.assertTrue(self.exact(ids))
        self.assertEqual(self.engine.chunk_blocks, 2)
        self.assertTrue(all(offset % 4096 == 0 and length % 4096 == 0 for offset, length in self.drive.reads))

    def test_construction_refusals_leave_nothing_open(self):
        layout = p2p.ShardLayout(**SMALL)
        cases = [
            (dict(staging_bytes=1000), 'cannot hold two shards'),
            (dict(io_align=4096), 'multiple of the 4096-byte direct-I/O alignment'),
        ]
        for kwargs, why in cases:
            mesh = FakeMesh(layout)
            factory, links = make_links(mesh)
            options = dict(staging_bytes=layout.shard_bytes * 4, io_align=1)
            options.update(kwargs)
            with self.assertRaisesRegex(p2p.P2PUnavailable, why):
                p2p.P2PEngine(p2p.StagingApi(mesh, 'mesh', [0, 1]), layout, lambda o, v: None, link_factory=factory, **options)
            self.assertEqual(links, [])

    def test_the_staging_records_must_describe_every_chip_and_be_big_enough(self):
        layout = p2p.ShardLayout(**SMALL)
        for mutate, why in ((lambda infos: infos[:3], 'answers for 3 devices'),
                            (lambda infos: [dict(info, size=info['size'] // 4) for info in infos], 'staged'),
                            (lambda infos: [{key: value for key, value in info.items() if key != 'noc_x'} for info in infos], 'lacks noc_x')):
            mesh = FakeMesh(layout)
            original = mesh.qwen_p2p_staging_info
            mesh.qwen_p2p_staging_info = lambda handle, original=original, mutate=mutate: mutate(original(handle))
            factory, links = make_links(mesh)
            with self.assertRaisesRegex(p2p.P2PUnavailable, why):
                p2p.P2PEngine(p2p.StagingApi(mesh, 'mesh', [0, 1]), layout, lambda o, v: None, staging_bytes=layout.shard_bytes * 4, link_factory=factory, io_align=1)

    def test_a_link_that_fails_to_open_closes_the_ones_that_did(self):
        layout = p2p.ShardLayout(**SMALL)
        mesh = FakeMesh(layout)
        factory, links = make_links(mesh)

        def failing(position, index, bdf, staging, ordering):
            if position == 2:
                raise p2p.P2PUnavailable('window held')
            return factory(position, index, bdf, staging, ordering)

        with self.assertRaisesRegex(p2p.P2PUnavailable, 'window held'):
            p2p.P2PEngine(p2p.StagingApi(mesh, 'mesh', [0, 1]), layout, lambda o, v: None, staging_bytes=layout.shard_bytes * 4, link_factory=failing, io_align=1)
        self.assertEqual(len(links), 2)
        self.assertTrue(all(link.closed for link in links))

    def test_the_configured_cards_must_be_the_meshs_cards(self):
        layout = p2p.ShardLayout(**SMALL)
        mesh = FakeMesh(layout)
        factory, links = make_links(mesh)
        have = ['0000:%02x:00.0' % (0x50 + chip) for chip in range(4)]
        engine = p2p.P2PEngine(p2p.StagingApi(mesh, 'mesh', [0, 1]), layout, lambda o, v: None, staging_bytes=layout.shard_bytes * 4, link_factory=factory, io_align=1, expected=have)
        engine.close()
        self.assertTrue(all(link.closed for link in links))
        mesh = FakeMesh(layout)
        factory, links = make_links(mesh)
        with self.assertRaisesRegex(p2p.P2PUnavailable, 'the configuration names'):
            p2p.P2PEngine(p2p.StagingApi(mesh, 'mesh', [0, 1]), layout, lambda o, v: None, staging_bytes=layout.shard_bytes * 4, link_factory=factory, io_align=1, expected=have[:3])

    def test_close_closes_every_link_and_is_idempotent(self):
        self.build()
        self.engine.close()
        self.engine.close()
        self.assertTrue(all(link.closed for link in self.links))
        with self.assertRaises(p2p.P2PError):
            self.engine.start(self.drive.entries)

    def test_the_description_names_what_a_log_line_needs(self):
        self.build()
        info = self.engine.describe()
        self.assertEqual((info['cards'], info['chunk_blocks'], info['shard_bytes'], info['mapping_verified'], info['window']), (4, 3, 2048, 1, 202))


class ThroughTheRealPoolTests(EngineCase):
    def test_sixteen_threads_over_many_blocks_stay_exact(self):
        self.build(blocks=40, chunk_blocks=5, threads=16)
        entries, ids = self.restore()
        self.assertTrue(self.exact(ids))
        self.assertEqual(self.engine.inflight, 0)


if __name__ == '__main__':
    unittest.main()
