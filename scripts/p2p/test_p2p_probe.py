import contextlib
import ctypes
import errno
import io
import json
import os
import pathlib
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p2p_probe as probe

MIB = 1 << 20


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = probe.main(argv)
    return code, out.getvalue(), err.getvalue()


def slice_info(offset=0, size=16 * MIB, window=probe.FIRST_4G_TLB, last=None, bar=4):
    return {'bar': bar, 'offset': offset, 'size': size, 'bus_addr': 0x38000000000 + offset, 'tlb_first': window,
            'tlb_last': window if last is None else last}


# Mirrors of the driver's structures (ioctl.h), written out by hand so that the byte layout the probe packs is checked against the C layout
# and not against itself.
class NocTlbConfig(ctypes.Structure):
    _fields_ = [('addr', ctypes.c_uint64), ('x_end', ctypes.c_uint16), ('y_end', ctypes.c_uint16), ('x_start', ctypes.c_uint16),
                ('y_start', ctypes.c_uint16), ('noc', ctypes.c_uint8), ('mcast', ctypes.c_uint8), ('ordering', ctypes.c_uint8),
                ('linked', ctypes.c_uint8), ('static_vc', ctypes.c_uint8), ('reserved0', ctypes.c_uint8 * 3), ('reserved1', ctypes.c_uint32 * 2)]


class ConfigureTlb(ctypes.Structure):
    _fields_ = [('id', ctypes.c_uint32), ('reserved', ctypes.c_uint32), ('config', NocTlbConfig), ('out_reserved', ctypes.c_uint64)]


class AllocateTlb(ctypes.Structure):
    _fields_ = [('size', ctypes.c_uint64), ('in_reserved', ctypes.c_uint64), ('id', ctypes.c_uint32), ('reserved0', ctypes.c_uint32),
                ('mmap_offset_uc', ctypes.c_uint64), ('mmap_offset_wc', ctypes.c_uint64), ('reserved1', ctypes.c_uint64)]


class ExportTlbDmabuf(ctypes.Structure):
    _fields_ = [('argsz', ctypes.c_uint32), ('flags', ctypes.c_uint32), ('tlb_id', ctypes.c_uint32), ('fd', ctypes.c_int32),
                ('offset', ctypes.c_uint64), ('size', ctypes.c_uint64)]


class PowerState(ctypes.Structure):
    _fields_ = [('argsz', ctypes.c_uint32), ('flags', ctypes.c_uint32), ('reserved0', ctypes.c_uint8), ('validity', ctypes.c_uint8),
                ('power_flags', ctypes.c_uint16), ('power_settings', ctypes.c_uint16 * 14)]


class FakeIoctl(object):
    """Stands in for fcntl.ioctl: records (fd, request, bytes) and answers ALLOCATE_TLB from a list of ids."""

    def __init__(self, ids=(), fail=()):
        self.calls = []
        self.ids = list(ids)
        self.fail = set(fail)

    def __call__(self, fd, request, buf, mutate=True):
        self.calls.append((fd, request, bytes(buf)))
        if request in self.fail:
            raise OSError(errno.ENOTTY, 'refused')
        if request == probe.IOCTL_ALLOCATE_TLB:
            if not self.ids:
                raise OSError(errno.ENOMEM, 'no window')
            probe.ALLOCATE_OUT.pack_into(buf, probe.ALLOCATE_IN.size, self.ids.pop(0), 0, 0, 0x7000000000, 0)
        if request == probe.IOCTL_EXPORT_TLB_DMABUF:
            probe.EXPORT_ARG.pack_into(buf, 0, probe.EXPORT_ARG.size, 0, 0, 77, 0, 0)
        return 0

    def frees(self):
        return [struct.unpack('<I', raw)[0] for _, request, raw in self.calls if request == probe.IOCTL_FREE_TLB]


def card_for_ioctls(fd=99):
    card = probe.TtCard.__new__(probe.TtCard)
    card.fd = fd
    card.windows = {}
    return card


class ArgumentsTest(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(probe.parse_size('4096'), 4096)
        self.assertEqual(probe.parse_size('64K'), 64 << 10)
        self.assertEqual(probe.parse_size('16m'), 16 << 20)
        self.assertEqual(probe.parse_size('2G'), 2 << 30)
        self.assertEqual(probe.parse_size('0x1000'), 4096)

    def test_dram_bank_coordinates_follow_the_translated_map(self):
        self.assertEqual(probe.dram_bank_xy(0), (17, 12))
        self.assertEqual(probe.dram_bank_xy(1), (17, 15))
        self.assertEqual(probe.dram_bank_xy(3), (17, 21))
        self.assertEqual(probe.dram_bank_xy(4), (18, 12))
        self.assertEqual(probe.dram_bank_xy(7), (18, 21))
        self.assertEqual(probe.dram_bank_xy(2, port=2), (17, 20))
        for bad in ((8, 0), (-1, 0), (0, 3)):
            with self.assertRaises(ValueError):
                probe.dram_bank_xy(*bad)

    def test_the_request_numbers_are_the_drivers(self):
        # _IO(0xFA, n)
        self.assertEqual((probe.IOCTL_ALLOCATE_TLB, probe.IOCTL_FREE_TLB, probe.IOCTL_CONFIGURE_TLB, probe.IOCTL_SET_POWER_STATE,
                          probe.IOCTL_EXPORT_TLB_DMABUF), (0xFA0B, 0xFA0C, 0xFA0D, 0xFA0F, 0xFA10))

    def test_the_struct_sizes_match_c_mirrors_of_the_header(self):
        self.assertEqual(probe.ALLOCATE_IN.size + probe.ALLOCATE_OUT.size, ctypes.sizeof(AllocateTlb))
        self.assertEqual(probe.CONFIGURE_IN.size + 8, ctypes.sizeof(ConfigureTlb))
        self.assertEqual(probe.EXPORT_ARG.size, ctypes.sizeof(ExportTlbDmabuf))
        self.assertEqual(probe.POWER_STATE.size, ctypes.sizeof(PowerState))


class WireTest(unittest.TestCase):
    """What the probe sends to the driver, decoded through the C mirrors."""

    def test_configure_tlb_puts_every_field_where_the_driver_reads_it(self):
        fake = FakeIoctl()
        window = probe.TtWindow(card_for_ioctls(), 203, 0)
        with mock.patch('fcntl.ioctl', fake):
            window.configure(x=18, y=15, addr=4 << 30, ordering=2)
        fd, request, raw = fake.calls[0]
        self.assertEqual((fd, request), (99, probe.IOCTL_CONFIGURE_TLB))
        sent = ConfigureTlb.from_buffer_copy(raw)
        self.assertEqual((sent.id, sent.reserved, sent.out_reserved), (203, 0, 0))
        config = sent.config
        self.assertEqual((config.addr, config.x_end, config.y_end, config.x_start, config.y_start), (4 << 30, 18, 15, 0, 0))
        self.assertEqual((config.noc, config.mcast, config.ordering, config.linked, config.static_vc), (0, 0, 2, 0, 0))

    def test_the_power_state_is_the_one_the_umd_asks_for(self):
        fake = FakeIoctl()
        with mock.patch('fcntl.ioctl', fake), mock.patch('time.sleep') as sleep:
            card_for_ioctls().power_up(0.25)
        sent = PowerState.from_buffer_copy(fake.calls[0][2])
        self.assertEqual(fake.calls[0][1], probe.IOCTL_SET_POWER_STATE)
        self.assertEqual((sent.argsz, sent.flags, sent.reserved0, sent.validity, sent.power_flags), (40, 0, 0, 4, 0x0E))
        self.assertEqual(list(sent.power_settings), [0] * 14)
        sleep.assert_called_once_with(0.25)

    def test_a_refused_power_request_is_an_error_and_nothing_follows(self):
        with mock.patch('fcntl.ioctl', FakeIoctl(fail=[probe.IOCTL_SET_POWER_STATE])), mock.patch('time.sleep') as sleep:
            with self.assertRaisesRegex(probe.ProbeError, 'SET_POWER_STATE'):
                card_for_ioctls().power_up(0)
        sleep.assert_not_called()

    def test_the_pin_exports_the_whole_window(self):
        fake = FakeIoctl()
        window = probe.TtWindow(card_for_ioctls(), 202, 0)
        with mock.patch('fcntl.ioctl', fake):
            self.assertTrue(window.pin())
        sent = ExportTlbDmabuf.from_buffer_copy(fake.calls[0][2])
        self.assertEqual((sent.argsz, sent.flags, sent.tlb_id, sent.offset, sent.size), (32, 0, 202, 0, 0))
        self.assertEqual(window.pin_fd, 77)
        with mock.patch('fcntl.ioctl', FakeIoctl(fail=[probe.IOCTL_EXPORT_TLB_DMABUF])):
            self.assertFalse(probe.TtWindow(card_for_ioctls(), 202, 0).pin())

    def test_the_wanted_window_is_taken_and_the_ones_before_it_given_back(self):
        fake = FakeIoctl(ids=[202, 203])
        with mock.patch('fcntl.ioctl', fake):
            window = card_for_ioctls().acquire_window(203)
        self.assertEqual(window.id, 203)
        self.assertEqual(fake.frees(), [202])
        self.assertEqual(AllocateTlb.from_buffer_copy(fake.calls[0][2]).size, 4 << 30)

    def test_a_window_somebody_else_holds_is_not_replaced_by_a_later_one(self):
        fake = FakeIoctl(ids=[204])                  # 202 and 203 are taken by others; the driver hands out 204
        with mock.patch('fcntl.ioctl', fake):
            with self.assertRaisesRegex(probe.ProbeError, 'held by another process'):
                card_for_ioctls().acquire_window(203)
        self.assertEqual(fake.frees(), [204])

    def test_running_out_of_windows_is_an_error_that_gives_the_taken_ones_back(self):
        fake = FakeIoctl(ids=[202])
        with mock.patch('fcntl.ioctl', fake):
            with self.assertRaisesRegex(probe.ProbeError, 'ALLOCATE_TLB failed'):
                card_for_ioctls().acquire_window(205)
        self.assertEqual(fake.frees(), [202])

    def test_allocated_means_the_pci_cores_available_is_below_its_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = card_for_ioctls()
            card.bdf = '0000:00:00.0'
            card.class_dir = os.path.join(tmp, 'class')
            card.pci_dir = os.path.join(tmp, 'pci')
            for directory, values in ((os.path.join(card.class_dir, 'p2pmem'), dict(bar='4', offset='0', size='4294967296', tlb_first='202',
                                                                                  tlb_last='202', bus_addr='0x38000000000')),
                                      (os.path.join(card.pci_dir, 'p2pmem'), dict(size='4294967296', available='4294967296'))):
                os.makedirs(directory)
                for name, value in values.items():
                    with open(os.path.join(directory, name), 'w') as handle:
                        handle.write(value + '\n')
            info = card.p2pmem_info()
            self.assertEqual((info['in_use'], info['bar'], info['tlb_first'], info['bus_addr']), (0, 4, 202, 0x38000000000))
            with open(os.path.join(card.pci_dir, 'p2pmem', 'available'), 'w') as handle:
                handle.write('0\n')
            self.assertEqual(card.p2pmem_info()['in_use'], 1)
            os.unlink(os.path.join(card.pci_dir, 'p2pmem', 'available'))
            with self.assertRaisesRegex(probe.ProbeError, 'unreadable'):
                card.p2pmem_info()

    def test_a_card_without_the_patched_driver_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = card_for_ioctls()
            card.bdf = '0000:00:00.0'
            card.class_dir = os.path.join(tmp, 'class')
            card.pci_dir = tmp
            with self.assertRaisesRegex(probe.ProbeError, 'patched tt-kmd is not loaded'):
                card.p2pmem_info()

    def test_the_expected_physical_address_is_bar4s_start_plus_the_slice_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = card_for_ioctls()
            card.pci_dir = tmp
            lines = ['0x%016x 0x%016x 0x0000000000040200' % (0x1000 * i, 0x1000 * i + 0xfff) for i in range(4)]
            lines.append('0x0000380000000000 0x00003807ffffffff 0x000000000014220c')
            lines.append('0x0000000000000000 0x0000000000000000 0x0000000000000000')
            with open(os.path.join(tmp, 'resource'), 'w') as handle:
                handle.write('\n'.join(lines) + '\n')
            self.assertEqual(card.expected_phys({'offset': 0x40000000}), 0x380040000000)


class PlanTest(unittest.TestCase):
    def plan(self, info=None, **kw):
        args = dict(size=4 * MIB, dest_offset=0, chunk=MIB, source_offset=0, noc_x=17, noc_y=12, noc_addr=0, ordering=0)
        args.update(kw)
        return probe.make_plan(info or slice_info(), **args)

    def test_the_destination_is_where_the_window_puts_it(self):
        plan = self.plan(slice_info(offset=0, size=4 << 30), dest_offset=1 << 30, size=64 * MIB, chunk=16 * MIB)
        self.assertEqual(plan['window'], 202)
        self.assertEqual(plan['dest']['window_offset'], 1 << 30)
        self.assertEqual(plan['dest']['noc_addr'], 1 << 30)
        self.assertEqual(plan['chunks'], 4)

    def test_the_second_window_starts_four_gib_into_bar4(self):
        plan = self.plan(slice_info(offset=(4 << 30) + 256 * MIB, size=512 * MIB, window=203), dest_offset=MIB, noc_addr=0)
        self.assertEqual(plan['window_bar4_offset'], 4 << 30)
        self.assertEqual(plan['dest']['window_offset'], 256 * MIB + MIB)
        self.assertEqual(plan['dest']['noc_addr'], 256 * MIB + MIB)

    def test_a_window_aimed_at_a_later_dram_range_adds_its_base(self):
        plan = self.plan(noc_addr=4 << 30)
        self.assertEqual(plan['dest']['noc_addr'], 4 << 30)

    def test_the_chunk_never_exceeds_the_transfer(self):
        self.assertEqual(self.plan(size=MIB, chunk=16 * MIB)['chunk'], MIB)

    def test_refusals(self):
        with self.assertRaisesRegex(probe.ProbeError, 'BAR0'):
            self.plan(slice_info(bar=0))
        with self.assertRaisesRegex(probe.ProbeError, 'one 4 GiB TLB window'):
            self.plan(slice_info(window=202, last=203))
        with self.assertRaisesRegex(probe.ProbeError, 'one 4 GiB TLB window'):
            self.plan(slice_info(window=-1, last=-1))
        with self.assertRaisesRegex(probe.ProbeError, 'not one of the 4 GiB'):
            self.plan(slice_info(window=100))
        with self.assertRaisesRegex(probe.ProbeError, 'multiple of 4 GiB'):
            self.plan(noc_addr=MIB)
        with self.assertRaisesRegex(probe.ProbeError, 'multiple of 4 GiB'):
            self.plan(noc_addr=-(4 << 30))
        with self.assertRaisesRegex(probe.ProbeError, 'sector'):
            self.plan(size=4 * MIB + 1)
        with self.assertRaisesRegex(probe.ProbeError, 'sector'):
            self.plan(dest_offset=100)
        with self.assertRaisesRegex(probe.ProbeError, 'outside the published slice'):
            self.plan(size=32 * MIB)
        with self.assertRaisesRegex(probe.ProbeError, 'outside the published slice'):
            self.plan(dest_offset=15 * MIB, size=2 * MIB)

    def test_nothing_may_be_negative(self):
        for key in ('dest_offset', 'source_offset', 'chunk', 'size'):
            with self.subTest(key=key):
                with self.assertRaisesRegex(probe.ProbeError, 'negative|at least'):
                    self.plan(**{key: -4096})

    def test_the_destination_must_leave_room_for_the_self_test_at_both_ends(self):
        with self.assertRaisesRegex(probe.ProbeError, 'at least'):
            self.plan(size=4096)
        self.assertEqual(self.plan(size=8192)['dest']['size'], 8192)

    def test_noc_coordinates_must_fit_the_register(self):
        for x, y in ((64, 12), (17, 64), (-1, 12), (17, -1)):
            with self.assertRaisesRegex(probe.ProbeError, '6-bit'):
                self.plan(noc_x=x, noc_y=y)
        self.assertEqual(self.plan(noc_x=63, noc_y=63)['noc']['x'], 63)


class ExtentsTest(unittest.TestCase):
    def test_written_extents_covering_the_range_pass(self):
        probe.check_extents([(0, 100, 8192, 0), (8192, 9000, 8192, 1)], 0, 16384)
        probe.check_extents([(0, 100, 8192, 0), (8192, 9000, 8192, 1)], 4096, 8192)

    def test_what_the_kernel_would_fill_by_cpu_is_refused(self):
        for flags in (0x800, 0x4, 0x200, 0x8, 0x2):
            with self.assertRaisesRegex(probe.ProbeError, 'flags'):
                probe.check_extents([(0, 100, 8192, flags)], 0, 8192)

    def test_a_hole_is_refused(self):
        with self.assertRaisesRegex(probe.ProbeError, 'hole'):
            probe.check_extents([(0, 100, 4096, 0), (8192, 9000, 4096, 1)], 0, 12288)
        with self.assertRaisesRegex(probe.ProbeError, 'hole'):
            probe.check_extents([(4096, 100, 4096, 1)], 0, 8192)

    def test_a_short_file_is_refused(self):
        with self.assertRaisesRegex(probe.ProbeError, 'short'):
            probe.check_extents([(0, 100, 4096, 1)], 0, 8192)

    def test_a_file_with_more_extents_than_fit_is_refused_not_reported_short(self):
        def fill(fd, request, buf, mutate):
            probe.FIEMAP_HEADER.pack_into(buf, 0, 0, 0, 0, 3, 3, 0)
            for index in range(3):
                probe.FIEMAP_EXTENT.pack_into(buf, probe.FIEMAP_HEADER.size + index * probe.FIEMAP_EXTENT.size,
                                              index * 4096, 100, 4096, 0, 0, 0, 0, 0, 0)

        with mock.patch('fcntl.ioctl', fill):
            with self.assertRaisesRegex(probe.ProbeError, 'too fragmented'):
                probe.fiemap_extents(5, 0, 1 << 20, max_extents=3)


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, 's.bin')
        probe.prepare_source(self.path, 8192, seed=1)

    def test_the_refusal_of_peer_to_peer_is_named(self):
        source = probe.Source(self.path, direct=False)
        buf = bytearray(4096)
        with mock.patch('os.preadv', side_effect=OSError(errno.EREMOTEIO, 'Remote I/O error')):
            with self.assertRaisesRegex(probe.ProbeError, 'refused peer-to-peer'):
                source.read_into(memoryview(buf), 0)
        with mock.patch('os.preadv', return_value=0):
            with self.assertRaisesRegex(probe.ProbeError, 'short read'):
                source.read_into(memoryview(buf), 0)
        source.close()

    def test_a_short_pread_is_continued(self):
        source = probe.Source(self.path, direct=False)
        buf = bytearray(8192)
        real = os.preadv
        calls = []

        def short(fd, buffers, offset):
            calls.append(offset)
            return real(fd, [buffers[0][:4096]], offset)

        with mock.patch('os.preadv', short):
            source.read_into(memoryview(buf), 0)
        source.close()
        self.assertEqual(calls, [0, 4096])
        with open(self.path, 'rb') as handle:
            self.assertEqual(bytes(buf), handle.read())

    def test_the_length_of_a_file_is_its_size_and_a_range_beyond_it_is_refused(self):
        source = probe.Source(self.path, direct=False, extents=lambda fd, start, length: [(0, 1, 8192, 1)])
        self.assertEqual(source.length, 8192)
        source.check(0, 8192)
        with self.assertRaisesRegex(probe.ProbeError, 'holds 8192'):
            source.check(4096, 8192)
        source.close()

    def test_a_filesystem_that_cannot_say_is_a_refusal_not_a_guess(self):
        def broken(fd, start, length):
            raise OSError(errno.ENOTTY, 'no fiemap')

        source = probe.Source(self.path, direct=False, extents=broken)
        with self.assertRaisesRegex(probe.ProbeError, 'cannot prove'):
            source.check(0, 8192)
        source.close()


class MockRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = os.path.join(self.tmp.name, 'source.bin')
        probe.prepare_source(self.source, 12 * MIB, seed=11)

    def argv(self, *extra):
        return ['--mock', '--source', self.source, '--size', '8M', '--chunk', '1M', '--dest-offset', '0', '--json'] + list(extra)

    def parsed(self, *extra):
        args = probe.build_parser().parse_args(['--source', self.source, '--size', '8M', '--chunk', '1M', '--dest-offset', '0', '--settle', '0']
                                               + list(extra))
        probe.resolve_target(args)
        return args

    def run_with(self, card, *extra):
        source = probe.MockSource(self.source, card=card)
        try:
            return probe.run_probe(card, source, self.parsed(*extra))
        finally:
            source.close()

    def test_a_clean_run_passes_and_says_how_fast(self):
        code, out, err = run_main(self.argv('--verify', 'full', '--threads', '2', '--repeat', '2', '--baseline'))
        self.assertEqual((code, err), (0, ''))
        report = json.loads(out)
        self.assertEqual(report['result'], 'PASS')
        self.assertTrue(report['ok'])
        self.assertEqual(report['verify'], {'mode': 'full', 'bytes_compared': 8 * MIB, 'ok': True, 'mismatches': []})
        self.assertEqual(report['p2p']['bytes'], 8 * MIB)
        self.assertEqual(report['p2p']['runs'], 2)
        self.assertGreater(report['p2p']['gbps'], 0)
        self.assertTrue(report['baseline']['verify']['ok'])
        self.assertGreater(report['baseline']['bounce_gbps'], 0)
        self.assertTrue(report['pinned'])
        self.assertTrue(report['bus_addr_check'])
        self.assertEqual(report['plan']['noc'], {'x': 17, 'y': 12, 'addr': 0, 'ordering': 0})
        self.assertTrue(any('uncached' in note for note in report['notes']))

    def test_offsets_into_the_source_and_into_the_slice_are_both_honoured(self):
        # the source bytes 1M..9M must end up at slice offset 2M..10M
        card = probe.MockCard()
        report = self.run_with(card, '--source-offset', '1M', '--dest-offset', '2M', '--verify', 'full')
        self.assertEqual(report['result'], 'PASS')
        with open(self.source, 'rb') as handle:
            handle.seek(MIB)
            expected = handle.read(8 * MIB)
        self.assertEqual(bytes(card.dram[2 * MIB:10 * MIB]), expected)
        self.assertEqual(report['plan']['dest']['window_offset'], 2 * MIB)

    def test_sample_verification_reads_the_ends_and_some_middles(self):
        code, out, _ = run_main(self.argv('--samples', '3', '--span', '1M'))
        report = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(report['verify']['mode'], 'sample')
        self.assertTrue(0 < report['verify']['bytes_compared'] <= 8 * MIB)

    def test_a_byte_landing_wrong_is_found_and_located(self):
        code, out, _ = run_main(self.argv('--verify', 'full', '--mock-fault', 'flip'))
        report = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(report['verify']['mismatches'][0]['source_offset'], 512 * 1024)

    def test_data_that_never_arrives_cannot_pass_on_leftovers(self):
        code, out, _ = run_main(self.argv('--verify', 'sample', '--mock-fault', 'misroute'))
        report = json.loads(out)
        self.assertEqual(code, 1)
        self.assertFalse(report['ok'])
        self.assertEqual(report['verify']['mismatches'][0]['found'], probe.SENTINEL)

    def test_a_dma_that_stops_after_the_first_run_is_caught_by_the_sentinel_of_the_second(self):
        code, out, _ = run_main(self.argv('--verify', 'full', '--repeat', '2', '--mock-fault', 'stall'))
        report = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(report['verify']['mismatches'][0]['run'], 2)
        self.assertEqual(report['verify']['mismatches'][0]['found'], probe.SENTINEL)

    def test_nothing_checked_is_not_a_pass(self):
        code, out, _ = run_main(self.argv('--verify', 'none', '--mock-fault', 'misroute'))
        report = json.loads(out)
        self.assertEqual((code, report['result'], report['ok']), (3, 'UNVERIFIED', False))
        code, out, _ = run_main(self.argv('--verify', 'none'))
        self.assertEqual(code, 3)

    def test_the_rate_gate_makes_a_verified_run_fail_once_and_says_so_once(self):
        out = io.StringIO()
        argv = self.argv('--verify', 'full', '--min-gbps', '1000000')
        argv.remove('--json')
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = probe.main(argv)
        self.assertEqual(code, 1)
        self.assertEqual(out.getvalue().count('RESULT'), 1)
        self.assertIn('RESULT FAIL', out.getvalue())
        self.assertIn('below --min-gbps', out.getvalue())

    def test_the_text_report_ends_in_a_verdict(self):
        out = io.StringIO()
        argv = self.argv('--verify', 'full')
        argv.remove('--json')
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = probe.main(argv)
        self.assertEqual(code, 0)
        self.assertIn('RESULT PASS', out.getvalue())
        self.assertIn('pinned True', out.getvalue())
        self.assertIn('note:', out.getvalue())

    def test_the_window_is_aimed_where_it_was_asked(self):
        card = probe.MockCard()
        self.run_with(card, '--bank', '5', '--ordering', 'posted')
        self.assertEqual(card.configured, (18, 15, 0, 2))
        self.assertEqual(card.closed, [probe.FIRST_4G_TLB])

    def test_the_dram_is_woken_before_the_window_is_touched(self):
        card = probe.MockCard()
        self.run_with(card)
        self.assertEqual(card.events[:4], ['power_up', 'acquire_window', 'configure', 'pin'])

    def test_no_power_no_window_no_dma(self):
        card = probe.MockCard(power_fails=True)
        with self.assertRaisesRegex(probe.ProbeError, 'SET_POWER_STATE'):
            self.run_with(card)
        self.assertEqual(card.events, ['power_up'])

    def test_an_unpinned_window_is_refused_given_back_and_goes_ahead_only_when_allowed(self):
        card = probe.MockCard(pinnable=False)
        with self.assertRaisesRegex(probe.ProbeError, 'pinned'):
            self.run_with(card)
        self.assertEqual(card.closed, [probe.FIRST_4G_TLB])
        self.assertNotIn('p2p_buffer', card.events)
        report = self.run_with(card, '--allow-unpinned')
        self.assertFalse(report['pinned'])
        self.assertEqual(report['result'], 'PASS')

    def test_a_window_that_is_not_memory_stops_before_any_dma(self):
        card = probe.MockCard(selftest_fails=True)
        with self.assertRaisesRegex(probe.ProbeError, 'does not read back'):
            self.run_with(card)
        self.assertNotIn('p2p_buffer', card.events)
        self.assertEqual(card.closed, [probe.FIRST_4G_TLB])

    def test_the_self_test_stays_inside_the_destination(self):
        card = probe.MockCard()
        card.dram[:] = bytes([0x11]) * len(card.dram)
        self.run_with(card, '--dest-offset', '2M', '--size', '8k', '--chunk', '8k', '--verify', 'none')
        # everything outside [2M, 2M+8k) is as it was
        self.assertEqual(set(card.dram[:2 * MIB]), {0x11})
        self.assertEqual(set(card.dram[2 * MIB + 8192:]), {0x11})

    def test_a_mapping_that_is_not_where_the_slice_is_stops_before_any_dma(self):
        card = probe.MockCard(phys_offset=4096)
        source = probe.MockSource(self.source, card=card)
        with self.assertRaisesRegex(probe.ProbeError, 'physical address'):
            probe.run_probe(card, source, self.parsed())
        self.assertEqual(source.delivered, 0)
        source.close()
        self.assertEqual(card.closed, [probe.FIRST_4G_TLB])

    def test_a_slice_somebody_else_holds_is_not_touched(self):
        card = probe.MockCard(in_use=1)
        with self.assertRaisesRegex(probe.ProbeError, 'already allocated'):
            self.run_with(card)
        self.assertEqual(card.events, [])

    def test_a_card_somebody_else_has_open_is_not_touched_unless_told(self):
        card = probe.MockCard(openers=[4242])
        with self.assertRaisesRegex(probe.ProbeError, '4242'):
            self.run_with(card)
        self.assertEqual(card.events, [])
        self.assertEqual(self.run_with(card, '--allow-shared')['result'], 'PASS')

    def test_a_window_somebody_else_holds_is_not_taken(self):
        card = probe.MockCard(held_windows=[probe.FIRST_4G_TLB])
        with self.assertRaisesRegex(probe.ProbeError, 'held by another process'):
            self.run_with(card)
        self.assertNotIn('configure', card.events)

    def test_a_source_with_a_hole_is_refused_before_the_card_is_touched(self):
        card = probe.MockCard()
        source = probe.MockSource(self.source, card=card, extents=lambda fd, start, length: [(0, 1, 4096, 0)])
        with self.assertRaisesRegex(probe.ProbeError, 'ends at|hole'):
            probe.run_probe(card, source, self.parsed())
        source.close()
        self.assertEqual(card.events, [])

    def test_baseline_only_moves_nothing_peer_to_peer(self):
        card = probe.MockCard()
        report = self.run_with(card, '--baseline-only', '--verify', 'full')
        self.assertNotIn('p2p', report)
        self.assertNotIn('p2p_buffer', card.events)
        self.assertEqual(report['result'], 'PASS')
        self.assertTrue(report['baseline']['verify']['ok'])
        self.assertGreater(report['baseline']['bounce_gbps'], 0)

    def test_a_source_shorter_than_the_request_is_refused(self):
        code, _, err = run_main(self.argv('--size', '16M'))
        self.assertEqual(code, 2)
        self.assertIn('holds', err)

    def test_a_size_that_is_not_a_sector_multiple_is_refused(self):
        code, _, err = run_main(self.argv('--size', '8388609'))
        self.assertEqual(code, 2)
        self.assertIn('sector', err)

    def test_a_missing_source_is_an_error_not_a_traceback(self):
        code, _, err = run_main(['--mock', '--source', os.path.join(self.tmp.name, 'nope.bin'), '--size', '8M', '--dest-offset', '0'])
        self.assertEqual(code, 2)
        self.assertIn('nope.bin', err)

    def test_no_source_means_no_run(self):
        code, _, err = run_main(['--mock'])
        self.assertEqual(code, 2)
        self.assertIn('--source', err)


class BufferTest(unittest.TestCase):
    """P2PBuffer and the window mapping on regular files standing in for the sysfs allocate file and the character device."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.card = card_for_ioctls()
        self.card.pci_dir = self.tmp.name
        os.makedirs(os.path.join(self.tmp.name, 'p2pmem'))
        self.allocate = os.path.join(self.tmp.name, 'p2pmem', 'allocate')
        with open(self.allocate, 'wb') as handle:
            handle.truncate(2 * MIB)

    def test_the_buffer_is_the_slice_and_closes_cleanly(self):
        buffer = self.card.p2p_buffer(2 * MIB)
        buffer.view[0:4] = b'abcd'
        self.assertEqual(buffer.size, 2 * MIB)
        self.assertIsInstance(buffer.bus_addr_of(0), int)
        buffer.close()
        with open(self.allocate, 'rb') as handle:
            self.assertEqual(handle.read(4), b'abcd')

    def test_a_failed_read_leaves_no_view_behind_to_block_the_close(self):
        class Failing(object):
            def read_into(self, view, offset):
                raise probe.ProbeError('the drive said no')

        plan = {'dest': {'size': 2 * MIB}, 'chunk': MIB, 'source_offset': 0}
        buffer = self.card.p2p_buffer(2 * MIB)
        for threads in (1, 4):
            with self.assertRaisesRegex(probe.ProbeError, 'said no'):
                probe.read_all(Failing(), buffer.view, plan, threads, 0)
        buffer.close()                                     # would raise BufferError if a slice were still exported

    def test_an_unmappable_file_is_a_probe_error_and_closes_its_descriptor(self):
        os.truncate(self.allocate, 0)
        before = set(os.listdir('/proc/self/fd'))
        with self.assertRaises(probe.ProbeError):
            self.card.p2p_buffer(2 * MIB)
        self.assertEqual(set(os.listdir('/proc/self/fd')) - before, set())

    def test_a_missing_allocate_file_says_what_is_needed(self):
        os.unlink(self.allocate)
        with self.assertRaisesRegex(probe.ProbeError, 'patched tt-kmd'):
            self.card.p2p_buffer(2 * MIB)

    def test_the_window_mapping_reads_and_writes(self):
        path = os.path.join(self.tmp.name, 'chardev')
        with open(path, 'wb') as handle:
            handle.truncate(8192)
        fd = os.open(path, os.O_RDWR)
        self.addCleanup(os.close, fd)
        window = probe.TtWindow(card_for_ioctls(fd), 202, 0)
        cpu = window.cpu_window(8192)
        cpu.write(100, b'xyz')
        self.assertEqual(cpu.read(99, 5), b'\0xyz\0')
        cpu.close()


class OverwriteGuardTest(unittest.TestCase):
    def fake(self, tmp, mounts='', mdstat='', swaps='', holders=()):
        paths = {}
        for name, text in (('proc_mounts', mounts), ('proc_mdstat', mdstat), ('proc_swaps', swaps)):
            paths[name] = os.path.join(tmp, name)
            with open(paths[name], 'w') as handle:
                handle.write(text)
        sysdir = os.path.join(tmp, 'block')
        for relative in holders:
            os.makedirs(os.path.join(sysdir, relative, 'holders'))
            open(os.path.join(sysdir, relative, 'holders', 'dm-0'), 'w').close()
        os.makedirs(os.path.join(sysdir, 'sdz'), exist_ok=True)
        paths['sys_class_block'] = sysdir
        return paths

    def test_a_free_device_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(probe.block_device_in_use('/dev/sdz', **self.fake(tmp)))

    def test_a_mounted_partition_swap_a_raid_member_or_a_held_device_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('mounted', probe.block_device_in_use('/dev/sdz', **self.fake(tmp, mounts='/dev/sdz1 / ext4 rw 0 0\n')))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('swap', probe.block_device_in_use('/dev/sdz', **self.fake(tmp, swaps='Filename Type\n/dev/sdz3 partition 1 0 -2\n')))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('array', probe.block_device_in_use('/dev/sdz', **self.fake(tmp, mdstat='md0 : active raid1 sdz2[1] sdy2[0]\n')))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('held', probe.block_device_in_use('/dev/sdz', **self.fake(tmp, holders=['sdz'])))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('held', probe.block_device_in_use('/dev/sdz', **self.fake(tmp, holders=['sdz/sdz3'])))

    def test_only_a_blank_device_may_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            blank, used = os.path.join(tmp, 'blank'), os.path.join(tmp, 'used')
            with open(blank, 'wb') as handle:
                handle.write(bytes(2 * MIB))
            with open(used, 'wb') as handle:
                handle.write(bytes(5000) + b'\x55\xaa' + bytes(MIB))
            self.assertIsNone(probe.first_nonzero_byte(blank))
            self.assertEqual(probe.first_nonzero_byte(used), 5000)
            with open(used, 'wb') as handle:
                handle.write(bytes(MIB) + b'x')                       # beyond the first MiB: not looked at
            self.assertIsNone(probe.first_nonzero_byte(used))

    def test_preparing_a_block_device_needs_the_flag_a_free_device_and_a_blank_one(self):
        with mock.patch('os.path.exists', return_value=True), mock.patch('os.stat', return_value=mock.Mock(st_mode=0o060600)):
            with self.assertRaisesRegex(probe.ProbeError, 'overwrite-device'):
                probe.prepare_source('/dev/sdz', MIB)
            with mock.patch.object(probe, 'block_device_in_use', return_value='it is mounted'):
                with self.assertRaisesRegex(probe.ProbeError, 'refusing'):
                    probe.prepare_source('/dev/sdz', MIB, overwrite_device=True)
            with mock.patch.object(probe, 'block_device_in_use', return_value=None), mock.patch.object(probe, 'first_nonzero_byte', return_value=512):
                with self.assertRaisesRegex(probe.ProbeError, 'not blank'):
                    probe.prepare_source('/dev/sdz', MIB, overwrite_device=True)

    def test_a_file_with_content_is_replaced_only_when_told(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'f.bin')
            probe.prepare_source(path, 8192, seed=1)
            with self.assertRaisesRegex(probe.ProbeError, 'overwrite-file'):
                probe.prepare_source(path, 8192, seed=2)
            with open(path, 'rb') as handle:
                before = handle.read()
            probe.prepare_source(path, 8192, seed=2, overwrite_file=True)
            with open(path, 'rb') as handle:
                self.assertNotEqual(handle.read(), before)

    def test_openers_of_a_card_are_found_in_proc(self):
        with tempfile.TemporaryDirectory() as tmp:
            for pid, target in ((4242, '/dev/tenstorrent/0'), (4243, '/dev/tenstorrent/1'), (os.getpid(), '/dev/tenstorrent/0')):
                os.makedirs(os.path.join(tmp, str(pid), 'fd'))
                os.symlink(target, os.path.join(tmp, str(pid), 'fd', '3'))
            os.makedirs(os.path.join(tmp, 'self'))
            self.assertEqual(probe.other_openers(0, proc=tmp), [4242])
            self.assertEqual(probe.other_openers(1, proc=tmp), [4243])
            self.assertEqual(probe.other_openers(2, proc=tmp), [])


class SafetyTest(unittest.TestCase):
    def test_dry_run_opens_nothing(self):
        with mock.patch('os.open', side_effect=AssertionError('opened a file')):
            code, out, _ = run_main(['--dry-run', '--size', '2G', '--bank', '6'])
        self.assertEqual(code, 0)
        plan = json.loads(out)['plan']
        self.assertEqual(plan['noc']['x'], 18)
        self.assertEqual(plan['noc']['y'], 18)
        self.assertEqual(plan['dest']['size'], 2 << 30)

    def test_a_real_run_needs_root(self):
        with mock.patch('os.geteuid', return_value=1000):
            code, _, err = run_main(['--source', '/nonexistent'])
        self.assertEqual(code, 2)
        self.assertIn('root', err)

    def test_prepare_writes_a_fully_written_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'p.bin')
            code, out, _ = run_main(['--prepare', path, '--size', '1M', '--seed', '4'])
            self.assertEqual(code, 0)
            self.assertEqual(os.path.getsize(path), MIB)
            # the same seed gives the same bytes: a prepared file can be rebuilt and compared
            other = os.path.join(tmp, 'q.bin')
            run_main(['--prepare', other, '--size', '1M', '--seed', '4'])
            with open(path, 'rb') as a, open(other, 'rb') as b:
                self.assertEqual(a.read(), b.read())

    def test_fiemap_on_a_real_file_if_the_filesystem_has_it(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('P2P_TEST_DIR')) as tmp:
            path = os.path.join(tmp, 'f.bin')
            probe.prepare_source(path, 2 * MIB, seed=1)
            fd = os.open(path, os.O_RDONLY)
            try:
                try:
                    extents = probe.fiemap_extents(fd, 0, 2 * MIB)
                except OSError as error:
                    self.skipTest('no FIEMAP here: %s' % error)
                probe.check_extents(extents, 0, 2 * MIB)
            finally:
                os.close(fd)


class HygieneTest(unittest.TestCase):
    """The repository is public: no hostname, address, registry, digest or home path in what this branch adds. /dev/tenstorrent is the
    driver's own public device node, which the probe has to open, so it is the one pattern of the repository's usual list that is allowed."""

    BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|home/|zot\.')
    ROOT = pathlib.Path(__file__).resolve().parents[2]

    def files(self):
        folders = [self.ROOT / 'scripts' / 'p2p', self.ROOT / 'patches' / 'tt-kmd-p2pmem']
        return [path for folder in folders for path in sorted(folder.iterdir()) if path.is_file() and path.suffix != '.pyc']

    def test_nothing_private_and_lf_endings(self):
        found = self.files()
        self.assertTrue(any(path.name == 'p2p_probe.py' for path in found))
        for path in found:
            text = path.read_text(encoding='utf-8')
            if path.name != pathlib.Path(__file__).name:        # this file spells the patterns it looks for
                self.assertIsNone(self.BANNED.search(text), path.name)
            self.assertNotIn('\r', text, path.name)

    def test_the_patch_adds_what_the_readme_says_and_nothing_else(self):
        patch = next(path for path in self.files() if path.suffix == '.patch').read_text(encoding='utf-8')
        touched = re.findall(r'^diff --git a/(\S+)', patch, re.M)
        self.assertEqual(sorted(touched), ['Makefile', 'chardev.c', 'device.h', 'docs/sysfs-attributes.md', 'enumerate.c',
                                           'modprobe.d-tenstorrent.conf', 'p2pmem.c', 'p2pmem.h'])
        # off by default and never BAR0: the two properties the README promises
        self.assertIn('static char p2pmem_devices[256];', patch)
        self.assertIn('#define TT_P2PMEM_BAR 4', patch)
        self.assertNotIn('p2pmem_bar', patch)
        # the reset guard holds the slice for the whole reset instead of checking and letting go
        self.assertIn('tt_p2pmem_hold(tt_dev, &p2pmem_hold)', patch)
        self.assertIn('tt_p2pmem_release(tt_dev, p2pmem_hold)', patch)

    def test_the_workflow_runs_these_tests(self):
        workflow = (self.ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertIn("python -B -m unittest discover -s scripts/p2p -p 'test_*.py'", workflow)


if __name__ == '__main__':
    unittest.main()
