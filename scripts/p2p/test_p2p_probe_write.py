"""Tests of the write arm of p2p_probe.py (--mode write): the card's DRAM into a scratch range of a drive.

No device is touched. The card is a memfd with eight banks and a window that selects between them (MockBankedCard); the drive is a regular
file; the PCIe side (link speeds, AER counters, the NVMe controller state) is a tree shaped like sysfs in a temporary directory.
"""

import contextlib
import errno
import io
import json
import mmap
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p2p_probe as probe

MIB = 1 << 20
SHARD = probe.SHARD_BYTES


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = probe.main(argv)
    return code, out.getvalue(), err.getvalue()


def bump(directory, name, counter, by=1):
    """Raise one AER counter in a fake sysfs tree, as an error during the run would."""
    path = os.path.join(directory, name)
    counters = probe.parse_aer_counters(probe.read_text(path))
    counters[counter] = counters.get(counter, 0) + by
    with open(path, 'w') as handle:
        handle.write(''.join('%s %d\n' % item for item in counters.items()))


def set_text(path, text):
    with open(path, 'w') as handle:
        handle.write(text + '\n')


class WriteFixture(unittest.TestCase):
    """A mock card, a file for a drive, and the argument list for a small write run."""

    TARGET_BYTES = 64 * MIB

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, 'drive.img')
        with open(self.target, 'wb') as handle:
            handle.truncate(self.TARGET_BYTES)
        self.cards = []

    def card(self, **kw):
        kw.setdefault('bank_bytes', 32 * MIB)
        card = probe.MockBankedCard(**kw)
        self.cards.append(card)
        self.addCleanup(card.close)
        return card

    def argv(self, *extra):
        return ['--mode', 'write', '--mock', '--target', self.target, '--scratch-start', '1M', '--scratch-size', '24M', '--dest-offset', '0',
                '--region', '4M', '--sizes', str(SHARD), '--qd', '1,2', '--arm-seconds', '0', '--min-batches', '1', '--settle', '0',
                '--reaim-op-bytes', '1M', '--reaim-ops', '2', '--reaim-rounds', '8', '--soak-qd', '2', '--soak-size', '1M',
                '--soak-max-gbps', '0', '--min-write-gbps', '0'] + list(extra)

    def parsed(self, *extra):
        args = probe.build_parser().parse_args(self.argv(*extra))
        probe.resolve_target(args)
        return args

    def reset_target(self):
        with open(self.target, 'wb') as handle:
            handle.truncate(self.TARGET_BYTES)

    def run_with(self, card, *extra, **kw):
        """One run against a blank drive (unless keep=True: a test that sets the drive up itself)."""
        if not kw.get('keep'):
            self.reset_target()
        sink = probe.MockSink(self.target, card=card, fault=kw.get('fault'))
        hook = kw.get('hook')
        if hook:
            hook(sink)
        try:
            return probe.run_write_probe(card, sink, self.parsed(*extra)), sink
        finally:
            sink.close()

    def report(self, *extra, **kw):
        card = kw.pop('card', None) or self.card()
        return self.run_with(card, *extra, **kw)[0]

    def target_bytes(self, start, size):
        with open(self.target, 'rb') as handle:
            handle.seek(start)
            return handle.read(size)


class HelpersTest(unittest.TestCase):
    def test_lists_take_sizes_with_suffixes(self):
        self.assertEqual(probe.parse_int_list('1,4 8'), [1, 4, 8])
        self.assertEqual(probe.parse_int_list('1M,696320'), [MIB, 696320])
        with self.assertRaises(ValueError):
            probe.parse_int_list(' , ')

    def test_the_default_sizes_are_a_shard_an_object_and_a_sub_run(self):
        self.assertEqual(probe.WRITE_SIZES, (696320, 2785280, 22282240))
        self.assertEqual(probe.WRITE_SIZES[0] * 4, probe.WRITE_SIZES[1])
        self.assertEqual(probe.WRITE_SIZES[1] * 8, probe.WRITE_SIZES[2])
        for size in probe.WRITE_SIZES:
            self.assertEqual(size % probe.PAGE, 0)
        self.assertEqual(probe.WRITE_QDS[1:], (4, 8, 16, 32))

    def test_link_speeds_parse(self):
        self.assertEqual(probe.parse_gts('16.0 GT/s PCIe'), 16.0)
        self.assertEqual(probe.parse_gts('2.5 GT/s PCIe'), 2.5)
        self.assertEqual(probe.parse_gts('32 GT/s'), 32.0)
        self.assertIsNone(probe.parse_gts('Unknown speed'))
        self.assertIsNone(probe.parse_gts(None))

    def test_the_chain_lists_every_bridge_from_the_root_port_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp)
            chain = probe.pci_chain(tree['block_dir'])
            self.assertEqual([os.path.basename(path) for path in chain], ['0000:00:02.0', '0000:02:00.0', '0000:03:01.0', '0000:04:00.0'])
            self.assertEqual(chain, tree['drive_chain'])
            self.assertEqual(probe.pci_chain(tmp), [])

    def test_a_namespace_under_the_nvme_subsystem_is_followed_to_its_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp)
            subsystem = os.path.join(tmp, 'devices', 'virtual', 'nvme-subsystem', 'nvme-subsys0')
            os.makedirs(os.path.join(subsystem, 'nvme0n1'))
            os.symlink(os.path.dirname(tree['block_dir']), os.path.join(subsystem, 'nvme0'))
            sys_dev_block = os.path.join(tmp, 'dev-block')
            os.makedirs(sys_dev_block)
            os.symlink(os.path.join(subsystem, 'nvme0n1'), os.path.join(sys_dev_block, '259:0'))
            sink = probe.Sink.__new__(probe.Sink)
            sink.is_block, sink.rdev = True, os.makedev(259, 0)
            self.assertEqual(probe.pci_chain(os.path.join(subsystem, 'nvme0')), tree['drive_chain'])
            with mock.patch.object(probe.Sink, 'sysfs_dir', lambda self, *a: os.path.join(subsystem, 'nvme0n1')):
                self.assertEqual(sink.pci_chain(), tree['drive_chain'])

    def test_gen1_is_refused_whatever_the_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp, drive_gts=2.5)
            for allow in (False, True):
                with self.assertRaisesRegex(probe.ProbeError, 'Gen1'):
                    probe.check_links('drive', tree['drive_chain'], 0.0, allow)
            with self.assertRaisesRegex(probe.ProbeError, 'Gen1'):
                probe.check_links('drive', tree['drive_chain'], 16.0, False)

    def test_a_slow_drive_is_refused_unless_the_floor_is_lowered(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp, drive_gts=8.0)
            with self.assertRaisesRegex(probe.ProbeError, 'below the 16.0 GT/s'):
                probe.check_links('drive', tree['drive_chain'], 16.0)
            self.assertEqual(len(probe.check_links('drive', tree['drive_chain'], 0.0)), 4)

    def test_a_link_nobody_can_read_is_refused_unless_allowed_and_a_good_one_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp)
            self.assertEqual(probe.check_links('drive', tree['drive_chain'], 16.0)[-1]['speed_gts'], 16.0)
            os.unlink(os.path.join(tree['drive_chain'][1], 'current_link_speed'))
            with self.assertRaisesRegex(probe.ProbeError, 'cannot be read'):
                probe.check_links('drive', tree['drive_chain'], 16.0)
            self.assertEqual(probe.check_links('drive', tree['drive_chain'], 16.0, allow_unknown=True)[1]['speed_gts'], None)

    def test_aer_counters_are_read_per_file_and_the_growth_is_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp)
            chain = tree['drive_chain']
            before = probe.snapshot_aer(chain)
            self.assertEqual(before['0000:00:02.0']['aer_rootport_total_err_cor'], {'total': 0})
            self.assertEqual(before['0000:03:01.0']['aer_dev_correctable']['BadTLP'], 0)
            self.assertFalse(probe.aer_delta(before, before)['correctable'])
            bump(chain[2], 'aer_dev_correctable', 'BadTLP', 3)
            bump(chain[2], 'aer_dev_correctable', 'TOTAL_ERR_COR', 3)          # a total repeats the counters above it: not summed
            bump(chain[0], 'aer_rootport_total_err_cor', 'total', 0)
            delta = probe.aer_delta(before, probe.snapshot_aer(chain))
            self.assertEqual((delta['measured'], delta['correctable'], delta['uncorrectable']), (True, 3, 0))
            self.assertEqual(delta['by_function'], {'0000:03:01.0': {'aer_dev_correctable.BadTLP': 3}})
            bump(chain[3], 'aer_dev_nonfatal', 'CmpltTO')
            delta = probe.aer_delta(before, probe.snapshot_aer(chain))
            self.assertEqual((delta['correctable'], delta['uncorrectable']), (3, 1))
            self.assertEqual(delta['by_function']['0000:04:00.0'], {'aer_dev_nonfatal.CmpltTO': 1})
            bump(chain[1], 'aer_dev_fatal', 'DLP')
            self.assertEqual(probe.aer_delta(before, probe.snapshot_aer(chain))['uncorrectable'], 2)

    def test_the_monitor_samples_on_its_interval_and_keeps_the_longest_gap_from_when_the_dma_began(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp)
            monitor = probe.AerMonitor(tree['drive_chain'], tree['drive_chain'][-1])
            time.sleep(0.15)                                    # the card fill: no DMA moves, it is not a gap in watching the DMA
            monitor.start()
            self.assertIsNone(monitor.check(interval=60.0))        # not due yet
            self.assertEqual(monitor.count, 0)
            time.sleep(0.05)
            self.assertIsNotNone(monitor.check(interval=0.01))
            self.assertIsNotNone(monitor.check(force=True))
            summary = monitor.summary()
            self.assertEqual(summary['sample_count'], 2)
            self.assertTrue(0.04 < summary['max_gap_seconds'] < 0.15, summary)
            self.assertGreater(summary['mean_check_ms'], 0)
            self.assertEqual(len(monitor.samples), 2)
            bump(tree['drive_chain'][2], 'aer_dev_fatal', 'DLP')
            with self.assertRaises(probe.StopRun):
                monitor.check(force=True)
            self.assertEqual(monitor.summary()['sample_count'], 3)               # the sample that raised is counted, and its cost

    def test_no_counters_is_not_a_clean_bill(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = probe.snapshot_aer([tmp])
            self.assertEqual(snapshot, {os.path.basename(tmp): {}})
            self.assertFalse(probe.aer_delta(snapshot, snapshot)['measured'])

    def test_the_controller_state_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp)
            self.assertEqual(probe.nvme_states(tree['drive_chain'][-1]), ['live'])
            set_text(os.path.join(os.path.dirname(tree['block_dir']), 'state'), 'resetting')
            self.assertEqual(probe.nvme_states(tree['drive_chain'][-1]), ['resetting'])
            self.assertEqual(probe.nvme_states(tmp), [])

    def test_banks_are_told_apart_by_their_canary_and_their_pattern(self):
        pages = {probe.canary_page(1, bank) for bank in range(8)}
        self.assertEqual(len(pages), 8)
        self.assertEqual(len({page[:8] for page in pages}), 8)
        for page in pages:
            self.assertEqual(len(page), probe.PAGE)
        self.assertNotEqual(probe.canary_page(1, 0), probe.canary_page(2, 0))
        self.assertEqual(probe.bank_pattern(1, 3, 4096), probe.bank_pattern(1, 3, 4096))
        self.assertNotEqual(probe.bank_pattern(1, 3, 4096), probe.bank_pattern(1, 4, 4096))
        self.assertNotEqual(probe.bank_pattern(1, 3, 4096), probe.bank_pattern(2, 3, 4096))

    def test_bank_endpoints_round_trip(self):
        for bank in range(8):
            x, y = probe.dram_bank_xy(bank)
            self.assertEqual(probe.bank_of_xy(x, y), bank)
            self.assertEqual(probe.bank_of_xy(x, y + 2), bank)
        with self.assertRaises(ValueError):
            probe.bank_of_xy(1, 1)

    def test_the_bank_sequence_visits_every_bank_and_never_repeats_one_straight_away(self):
        order = probe.bank_sequence(list(range(8)), 64, 1)
        self.assertEqual(len(order), 64)
        self.assertTrue(all(a != b for a, b in zip(order, order[1:])))
        for start in range(0, 64, 8):
            self.assertEqual(sorted(order[start:start + 8]), list(range(8)))
        self.assertEqual(order, probe.bank_sequence(list(range(8)), 64, 1))
        self.assertNotEqual(order, probe.bank_sequence(list(range(8)), 64, 2))
        self.assertEqual(probe.bank_sequence([5], 3, 1), [5, 5, 5])

    def test_the_bank_sequence_never_repeats_across_a_pass_boundary_whatever_the_seed(self):
        for seed in range(60):
            for banks in (list(range(8)), [2, 5], [0, 1, 2]):
                order = probe.bank_sequence(banks, 40, seed)
                self.assertTrue(all(a != b for a, b in zip(order, order[1:])), (seed, banks))

    def test_the_scratch_range_is_handed_out_in_order_and_wraps(self):
        scratch = probe.Scratch(1000, 100)
        self.assertEqual([scratch.take(30), scratch.take(30), scratch.take(30)], [1000, 1030, 1060])
        self.assertEqual(scratch.take(30), 1000)              # 90 + 30 > 100: back to the start
        with self.assertRaises(probe.ProbeError):
            scratch.take(101)


class PlanTest(unittest.TestCase):
    def info(self, **kw):
        info = {'bar': 4, 'offset': 0, 'size': 4 << 30, 'bus_addr': 0, 'tlb_first': probe.FIRST_4G_TLB, 'tlb_last': probe.FIRST_4G_TLB}
        info.update(kw)
        return info

    def plan(self, region, dest_offset, **kw):
        return probe.make_plan(self.info(**kw), region, dest_offset, probe.PAGE, 0, 17, 12, 0, 0)

    def test_the_top_16_mib_of_a_window_is_not_dram(self):
        self.plan(64 * MIB, 1 << 30)
        self.plan(MIB, 0xFF000000 - MIB)                            # ends exactly at the limit
        with self.assertRaisesRegex(probe.ProbeError, 'not DRAM'):
            self.plan(MIB, 0xFF000000 - MIB + probe.PAGE)
        with self.assertRaisesRegex(probe.ProbeError, 'not DRAM'):
            self.plan(3 << 30, 1 << 30)                              # 3 GiB from 1 GiB reaches the top of the window
        with self.assertRaisesRegex(probe.ProbeError, 'not DRAM'):
            self.plan(MIB, 0, offset=0xFF000000, size=1 << 30, tlb_first=probe.FIRST_4G_TLB, tlb_last=probe.FIRST_4G_TLB)

    def test_the_second_window_counts_from_its_own_base(self):
        plan = self.plan(MIB, 0, offset=(4 << 30) + (1 << 30), tlb_first=203, tlb_last=203)
        self.assertEqual(plan['dest']['window_offset'], 1 << 30)

    def test_the_card_side_geometry_is_a_canary_page_then_the_data(self):
        args = probe.build_parser().parse_args(['--mode', 'write', '--dest-offset', '1G', '--reaim', '--region', '64M', '--banks', '0,3'])
        probe.resolve_target(args)
        plan, data = probe.write_plan(args, self.info())
        self.assertEqual(plan['region_bytes'], probe.PAGE + 64 * MIB)
        self.assertEqual(plan['dest']['window_offset'], 1 << 30)
        self.assertEqual(data, {0: 64 * MIB, 3: 4 * probe.OBJECT_BYTES})
        self.assertEqual(plan['banks']['3']['noc'], list(probe.dram_bank_xy(3)))


class ArgumentTest(unittest.TestCase):
    def parse(self, *extra):
        args = probe.build_parser().parse_args(['--mode', 'write', '--target', 'x'] + list(extra))
        probe.resolve_target(args)
        return args

    def refused(self, pattern, *extra):
        with self.assertRaisesRegex(probe.ProbeError, pattern):
            probe.check_write_args(self.parse(*extra))

    def test_defaults_are_the_designs_sweep(self):
        args = self.parse('--scratch-start', '1M', '--scratch-size', '1G')
        self.assertEqual(args.sizes, [696320, 2785280, 22282240])
        self.assertEqual(args.qds, [1, 4, 8, 16, 32])
        self.assertEqual((args.iov_bytes, args.cpu_map, args.min_drive_gts, args.write_verify), (696320, 'uc', 16.0, 'full'))
        probe.check_write_args(args)

    def test_a_range_is_always_required(self):
        self.refused('required')
        self.refused('required', '--scratch-start', '1M')
        self.refused('required', '--scratch-size', '1G')

    def test_the_owner_flag_needs_the_range_and_does_not_replace_it(self):
        self.refused('needs the range', '--i-own-this-scratch-range')
        self.refused('needs the range', '--i-own-this-scratch-range', '--scratch-start', '1M')
        probe.check_write_args(self.parse('--i-own-this-scratch-range', '--scratch-start', '1M', '--scratch-size', '1G'))

    def test_the_first_mib_of_a_drive_is_never_written(self):
        self.refused('first MiB', '--scratch-start', '0', '--scratch-size', '1G')
        self.refused('first MiB', '--scratch-start', '512K', '--scratch-size', '1G')
        probe.check_write_args(self.parse('--scratch-start', '1M', '--scratch-size', '1G'))

    def test_alignment(self):
        self.refused('multiples', '--scratch-start', '1048577', '--scratch-size', '1G')
        self.refused('multiples', '--scratch-start', '1M', '--scratch-size', '1073741825')
        self.refused('sector', '--scratch-start', '1M', '--scratch-size', '1G', '--sizes', '696321')
        self.refused('sector', '--scratch-start', '1M', '--scratch-size', '1G', '--iov-bytes', '1000')

    def test_the_range_must_hold_the_largest_batch(self):
        self.refused('smaller than the largest batch', '--scratch-start', '1M', '--scratch-size', '256M')       # 22282240 x 32 = 713 MB
        probe.check_write_args(self.parse('--scratch-start', '1M', '--scratch-size', '256M', '--qd', '4'))
        self.refused('smaller than the largest batch', '--scratch-start', '1M', '--scratch-size', '1M', '--no-sweep', '--soak', '1')

    def test_nonsense_is_refused(self):
        self.refused('nothing to do', '--scratch-start', '1M', '--scratch-size', '1G', '--no-sweep')
        self.refused('banks 0-7', '--scratch-start', '1M', '--scratch-size', '1G', '--bank', '8')
        self.refused('queue depth', '--scratch-start', '1M', '--scratch-size', '1G', '--qd', '0')
        self.refused('positive', '--scratch-start', '1M', '--scratch-size', '1G', '--no-sweep', '--soak', '1', '--soak-size', '0')
        self.refused('positive', '--scratch-start', '1M', '--scratch-size', '1G', '--sizes', '0')
        self.refused('iovecs', '--scratch-start', '1M', '--scratch-size', '1G', '--iov-bytes', '4096', '--sizes', '22282240', '--qd', '1')
        self.refused('region', '--scratch-start', '1M', '--scratch-size', '1G', '--region', '1M')

    def test_the_dry_run_plans_without_opening_anything(self):
        with mock.patch('os.open', side_effect=AssertionError('opened a file')), mock.patch('builtins.open', side_effect=AssertionError('opened a file')):
            code, out, _ = run_main(['--mode', 'write', '--dry-run', '--scratch-start', '1M', '--scratch-size', '1G', '--qd', '4,8', '--reaim',
                                     '--soak', '5'])
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertTrue(plan['dry_run'])
        self.assertEqual(len(plan['arms']), 6)
        self.assertEqual(plan['arms'][0], {'size': 696320, 'qd': 4, 'iovecs': 1})
        self.assertEqual(plan['arms'][-1], {'size': 22282240, 'qd': 8, 'iovecs': 32})
        self.assertEqual(plan['soak']['minutes'], 5.0)
        self.assertEqual(plan['scratch']['largest_batch'], 22282240 * 8)
        self.assertGreater(plan['estimated_writes_gib_at_most'], 0)
        self.assertEqual(plan['plan']['dest']['window_offset'], 1 << 30)

    def test_the_dry_run_needs_no_range_but_a_bad_one_is_still_refused(self):
        code, out, _ = run_main(['--mode', 'write', '--dry-run'])
        self.assertEqual(code, 0)
        self.assertIsNone(json.loads(out)['scratch']['start'])
        code, _, err = run_main(['--mode', 'write', '--dry-run', '--scratch-start', '0', '--scratch-size', '1G'])
        self.assertEqual(code, 2)
        self.assertIn('first MiB', err)

    def test_a_real_run_needs_root_and_a_target(self):
        with mock.patch('os.geteuid', return_value=1000):
            code, _, err = run_main(['--mode', 'write', '--target', '/nonexistent', '--scratch-start', '1M', '--scratch-size', '1G'])
        self.assertEqual(code, 2)
        self.assertIn('root', err)
        code, _, err = run_main(['--mode', 'write', '--mock', '--scratch-start', '1M', '--scratch-size', '1G'])
        self.assertEqual(code, 2)
        self.assertIn('--target', err)


class SinkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, 'd.img')
        with open(self.path, 'wb') as handle:
            handle.truncate(4 * MIB)
        self.sink = probe.Sink(self.path, direct=False)
        self.addCleanup(self.sink.close)

    def test_nothing_is_written_before_the_range_is_fenced_or_outside_it(self):
        buffer = bytearray(b'x' * 4096)
        with self.assertRaisesRegex(probe.ProbeError, 'fenced'):
            self.sink.write_buf(memoryview(buffer), 0)
        self.sink.allow(MIB, 2 * MIB)
        for offset in (0, MIB - 4096, 2 * MIB - 4095, 2 * MIB):
            with self.assertRaisesRegex(probe.ProbeError, 'outside the scratch range'):
                self.sink.write_buf(memoryview(buffer), offset)
        self.sink.write_buf(memoryview(buffer), MIB)
        self.sink.write_buf(memoryview(buffer), 2 * MIB - 4096)
        with open(self.path, 'rb') as handle:
            data = handle.read()
        self.assertEqual(data[:MIB], bytes(MIB))
        self.assertEqual(data[2 * MIB:], bytes(2 * MIB))
        self.assertEqual(self.sink.written, 8192)

    def test_one_pwritev_carries_the_pieces_in_order_and_a_short_write_is_continued(self):
        self.sink.allow(MIB, 2 * MIB)
        data = bytes(range(256)) * 48                                   # 12,288 bytes
        views = [memoryview(data)[0:4096], memoryview(data)[4096:8192], memoryview(data)[8192:12288]]
        real = os.pwritev
        calls = []

        def short(fd, buffers, offset):
            calls.append((offset, [len(buffer) for buffer in buffers]))
            first = bytes(buffers[0])[:3000] if len(calls) == 1 else None
            if first is not None:
                return real(fd, [first], offset)                         # 3000 bytes of 12288
            return real(fd, buffers, offset)

        with mock.patch('os.pwritev', short):
            self.assertEqual(self.sink.write_iov(views, MIB), 12288)
        self.assertEqual(calls, [(MIB, [4096, 4096, 4096]), (MIB + 3000, [1096, 4096, 4096])])
        with open(self.path, 'rb') as handle:
            handle.seek(MIB)
            self.assertEqual(handle.read(12288), data)
        for view in views:
            view.release()

    def test_a_refusal_of_peer_to_peer_is_named_and_a_zero_write_is_an_error(self):
        self.sink.allow(MIB, 2 * MIB)
        view = memoryview(bytes(4096))
        with mock.patch('os.pwritev', side_effect=OSError(errno.EREMOTEIO, 'Remote I/O error')):
            with self.assertRaisesRegex(probe.ProbeError, 'refused peer-to-peer'):
                self.sink.write_buf(view, MIB)
        with mock.patch('os.pwritev', return_value=0):
            with self.assertRaisesRegex(probe.ProbeError, 'short write'):
                self.sink.write_buf(view, MIB)
        view.release()

    def test_a_block_device_is_opened_exclusively_a_file_is_not(self):
        flags_seen = []
        real = os.open

        def spy(path, flags, *rest):
            flags_seen.append(flags)
            return real(path, flags & ~os.O_EXCL, *rest)

        with mock.patch('os.open', spy):
            probe.Sink(self.path, direct=False).close()
            with mock.patch.object(probe.stat, 'S_ISBLK', lambda mode: True):
                probe.Sink(self.path, direct=False).close()
        self.assertEqual([bool(flags & os.O_EXCL) for flags in flags_seen], [False, True])
        self.assertTrue(all(flags & os.O_RDWR for flags in flags_seen))

    def test_a_blank_range_is_blank_and_the_first_stray_byte_is_found(self):
        self.assertIsNone(probe.first_nonzero_in_range(self.sink, MIB, 2 * MIB, chunk=MIB // 2))
        with open(self.path, 'r+b') as handle:
            handle.seek(MIB + 1_500_000)
            handle.write(b'\x01')
        self.assertEqual(probe.first_nonzero_in_range(self.sink, MIB, 2 * MIB, chunk=MIB // 2), 1_500_000)
        self.assertIsNone(probe.first_nonzero_in_range(self.sink, MIB, 1_000_000 // 4096 * 4096, chunk=MIB // 2))

    def test_o_direct_on_a_real_file_if_the_filesystem_takes_it(self):
        directory = os.environ.get('P2P_TEST_DIR') or self.tmp.name
        path = os.path.join(directory, 'direct-%d.img' % os.getpid())
        with open(path, 'wb') as handle:
            handle.truncate(4 * MIB)
        self.addCleanup(os.unlink, path)
        try:
            sink = probe.Sink(path, direct=True)
        except OSError as error:
            self.skipTest('no O_DIRECT here: %s' % error)
        self.addCleanup(sink.close)
        source = mmap.mmap(-1, 2 * probe.PAGE)
        source[:] = os.urandom(len(source))
        sink.allow(MIB, 2 * MIB)
        try:
            with memoryview(source)[:probe.PAGE] as first, memoryview(source)[probe.PAGE:] as second:
                sink.write_iov([first, second], MIB)
            back = mmap.mmap(-1, 2 * probe.PAGE)
            with memoryview(back) as view:
                sink.read_into(view, MIB)
        except probe.ProbeError as error:
            if 'Invalid argument' in str(error):
                self.skipTest('O_DIRECT is not usable on this filesystem: %s' % error)
            raise
        self.assertEqual(back[:], source[:])
        back.close()
        source.close()


class WindowMappingTest(unittest.TestCase):
    """The window can be mapped uncached as well as write-combining: the tier's canary reads go through the uncached one."""

    def test_the_uncached_offset_is_the_one_mapped_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'chardev')
            with open(path, 'wb') as handle:
                handle.truncate(16384)
            fd = os.open(path, os.O_RDWR)
            self.addCleanup(os.close, fd)
            card = probe.TtCard.__new__(probe.TtCard)
            card.fd, card.windows = fd, {}
            window = probe.TtWindow(card, 202, 4096, 8192)
            uc, wc = window.cpu_window(4096, uc=True), None
            uc.write(0, b'uncached')
            uc.close()
            with open(path, 'rb') as handle:
                handle.seek(8192)
                self.assertEqual(handle.read(8), b'uncached')
            window = probe.TtWindow(card, 202, 4096, 8192)
            wc = window.cpu_window(4096)
            wc.write(0, b'combined')
            wc.close()
            with open(path, 'rb') as handle:
                handle.seek(4096)
                self.assertEqual(handle.read(8), b'combined')
            with self.assertRaisesRegex(probe.ProbeError, 'uncached'):
                probe.TtWindow(card, 202, 4096).cpu_window(4096, uc=True)

    def test_allocate_tlb_keeps_both_offsets(self):
        def ioctl(fd, request, buf, mutate=True):
            if request == probe.IOCTL_ALLOCATE_TLB:
                probe.ALLOCATE_OUT.pack_into(buf, probe.ALLOCATE_IN.size, 202, 0, 0x6000000000, 0x7000000000, 0)
            return 0

        card = probe.TtCard.__new__(probe.TtCard)
        card.fd, card.windows = 5, {}
        with mock.patch('fcntl.ioctl', ioctl):
            window = card.acquire_window(202)
        self.assertEqual((window.mmap_offset_uc, window.mmap_offset_wc), (0x6000000000, 0x7000000000))


class MockCardTest(unittest.TestCase):
    def test_the_banks_are_one_memory_file_and_the_window_selects_between_them(self):
        card = probe.MockBankedCard(bank_bytes=MIB)
        self.addCleanup(card.close)
        window = card.acquire_window(probe.FIRST_4G_TLB)
        cpu = window.cpu_window(probe.WINDOW_4G, uc=True)
        buffer = card.p2p_buffer(MIB)
        self.addCleanup(buffer.close)
        window.configure(*probe.dram_bank_xy(2), 0, 0)
        cpu.write(100, b'bank two')
        self.assertEqual(bytes(buffer.view[100:108]), b'bank two')
        window.configure(*probe.dram_bank_xy(5), 0, 0)
        self.assertEqual(cpu.read(100, 8), bytes(8))
        cpu.write(100, b'bank 5!!')
        self.assertEqual(bytes(buffer.view[100:108]), b'bank 5!!')
        window.configure(*probe.dram_bank_xy(2), 0, 0)
        self.assertEqual(bytes(buffer.view[100:108]), b'bank two')
        self.assertEqual(os.readlink('/proc/self/fd/%d' % card.memfd).startswith('/memfd:'), hasattr(os, 'memfd_create'))
        with open('/proc/self/fd/%d' % card.memfd, 'rb') as handle:               # one memory file under all eight banks
            handle.seek(2 * MIB + 100)
            self.assertEqual(handle.read(8), b'bank two')


class WriteRunTest(WriteFixture):
    def test_a_clean_run_writes_what_was_in_the_card_and_passes(self):
        report = self.report('--reaim', '--soak', '0.01')
        self.assertEqual(report['result'], 'PASS', report['verdict'])
        self.assertTrue(report['ok'])
        self.assertIsNone(report['aborted'])
        verdict = report['verdict']
        self.assertTrue(all(verdict[key] is True for key in ('byte_exact', 'no_uncorrectable_aer', 'p2w_gbps_ok', 'reaim_canary_ok', 'go')))
        self.assertEqual([(arm['size'], arm['qd']) for arm in report['sweep']], [(SHARD, 1), (SHARD, 2)])
        for arm in report['sweep']:
            self.assertEqual(arm['verify'], {'mode': 'full', 'writes': arm['writes'], 'bytes': arm['bytes'], 'ok': True})
            self.assertGreater(arm['gbps'], 0)
            self.assertEqual(arm['iovecs'], 1)
            self.assertGreaterEqual(arm['write_max_ms'], arm['write_p50_ms'])
        self.assertEqual(report['reaim']['rounds'], 8)
        self.assertEqual(report['reaim']['canary_ok'], 8)
        self.assertEqual(report['reaim']['banks'], list(range(8)))
        self.assertEqual(report['soak']['verify']['ok'], True)
        self.assertGreaterEqual(report['soak']['verify']['cycles_verified'], 1)
        self.assertEqual(report['fill'], {bank: True for bank in range(8)})
        self.assertTrue(report['pinned'])
        self.assertTrue(report['bus_addr_check'])
        self.assertTrue(report['aer']['delta']['measured'])
        self.assertEqual(report['scratch']['start_lba512'], 2048)
        self.assertEqual(report['scratch']['blank'], True)
        json.dumps(report)                                             # all of it is JSON

    def test_the_whole_write_flow_through_a_real_o_direct_sink(self):
        """The mock sink is buffered; here the paint, the pwritev of the p2pmem views, the read-back and the fence all go through O_DIRECT
        (page-aligned iovecs taken from the card's mapping, as on the rig), on a regular file. Not exercised: a block device, the loop device
        and the /sys/dev/block chain, which need root."""
        directory = os.environ.get('P2P_TEST_DIR') or self.tmp.name
        path = os.path.join(directory, 'direct-flow-%d.img' % os.getpid())
        with open(path, 'wb') as handle:
            handle.truncate(self.TARGET_BYTES)
        self.addCleanup(os.unlink, path)
        card = self.card()
        try:
            sink = probe.Sink(path, direct=True)
        except OSError as error:
            self.skipTest('no O_DIRECT here: %s' % error)
        sink.card, sink.fault = card, None
        sink.sysfs_dir = lambda *a: card.sysfs['block_dir']
        sink.pci_chain = lambda: list(card.sysfs['drive_chain'])
        self.assertTrue(sink.direct)
        args = self.parsed('--qd', '1,4', '--reaim', '--soak', '0.005', '--target', path)
        try:
            report = probe.run_write_probe(card, sink, args)
        except probe.ProbeError as error:
            if 'Invalid argument' in str(error):
                self.skipTest('O_DIRECT is not usable on this filesystem: %s' % error)
            raise
        finally:
            sink.close()
        self.assertEqual(report['result'], 'PASS', report['aborted'])
        self.assertTrue(report['verdict']['byte_exact'])
        self.assertGreater(report['bytes']['written_to_drive'], 0)
        with open(path, 'rb') as handle:                                  # the first scratch write holds the first card slot's bytes
            handle.seek(1 * MIB)
            first = handle.read(SHARD)
        self.assertEqual(first, bytes(card.banks[0][probe.PAGE:probe.PAGE + SHARD]))
        with open(path, 'rb') as handle:                                  # and nothing was written below the range
            self.assertEqual(handle.read(1 * MIB), bytes(1 * MIB))

    def test_the_drive_holds_the_cards_bytes_where_the_range_begins(self):
        report = self.report('--sizes', str(probe.OBJECT_BYTES), '--qd', '1', '--iov-bytes', str(SHARD))
        self.assertEqual(report['result'], 'PASS')
        self.assertEqual(report['sweep'][0]['iovecs'], 4)
        # the first batch is at the start of the range, from slot 0 of bank 0's pattern; the last batch wrapped nowhere (24M range, 2.7M batches)
        pattern = probe.bank_pattern(1, 0, 4 * MIB)
        batches = report['sweep'][0]['batches']
        for index in range(batches):
            offset = MIB + index * probe.OBJECT_BYTES
            self.assertEqual(self.target_bytes(offset, probe.OBJECT_BYTES), pattern[:probe.OBJECT_BYTES], index)

    def test_nothing_outside_the_range_is_touched(self):
        with open(self.target, 'r+b') as handle:
            handle.seek(0)
            handle.write(b'\x77' * MIB)
            handle.seek(25 * MIB)
            handle.write(b'\x77' * (self.TARGET_BYTES - 25 * MIB))
        report = self.report('--reaim', '--soak', '0.005', keep=True)
        self.assertEqual(report['result'], 'PASS')
        data = self.target_bytes(0, self.TARGET_BYTES)
        self.assertEqual(data[:MIB], b'\x77' * MIB)
        self.assertEqual(data[25 * MIB:], b'\x77' * (self.TARGET_BYTES - 25 * MIB))

    def test_the_bytes_written_are_accounted(self):
        report = self.report()
        self.assertEqual(report['bytes']['p2w'], sum(arm['bytes'] for arm in report['sweep']))
        self.assertGreater(report['bytes']['sentinel'], 0)
        self.assertEqual(report['bytes']['written_to_drive'], report['bytes']['p2w'] + report['bytes']['sentinel'])

    def test_a_range_that_is_not_blank_is_refused_before_the_card_is_touched(self):
        with open(self.target, 'r+b') as handle:
            handle.seek(MIB + 100)                               # inside the first write
            handle.write(b'important')
        card = self.card()
        with self.assertRaisesRegex(probe.ProbeError, 'not blank.*first non-zero byte at %d' % (MIB + 100)):
            self.run_with(card, keep=True)
        self.assertEqual(card.events, [])
        self.assertEqual(self.target_bytes(MIB + 100, 9), b'important')           # and not overwritten
        # the owner says it is theirs: it goes ahead, and the data is gone
        report = self.run_with(card, '--i-own-this-scratch-range', keep=True)[0]
        self.assertEqual(report['result'], 'PASS')
        self.assertIsNone(report['scratch']['blank'])
        self.assertTrue(report['scratch']['owner_flag'])
        self.assertNotEqual(self.target_bytes(MIB + 100, 9), b'important')

    def test_a_block_device_that_something_uses_is_refused_before_the_card_is_touched(self):
        card = self.card()
        for reason in ('it, or a partition of it, is mounted (/dev/x1)', 'it is part of a software RAID array'):
            sink = probe.MockSink(self.target, card=card)
            sink.is_block = True
            with mock.patch.object(probe, 'block_device_in_use', return_value=reason):
                with self.assertRaises(probe.ProbeError) as raised:
                    probe.run_write_probe(card, sink, self.parsed())
            self.assertIn('refusing to write to %s: %s' % (self.target, reason), str(raised.exception))
            sink.close()
        self.assertEqual(card.events, [])
        sink = probe.MockSink(self.target, card=card)
        sink.is_block = True
        with mock.patch.object(probe, 'block_device_in_use', return_value=None) as check:
            self.assertEqual(probe.run_write_probe(card, sink, self.parsed())['result'], 'PASS')
        sink.close()
        check.assert_called_once_with(self.target)

    def test_a_block_device_with_no_pci_function_above_it_is_refused_unless_allowed(self):
        card = self.card()
        sink = probe.MockSink(self.target, card=card)
        sink.is_block = True
        with mock.patch.object(probe, 'block_device_in_use', return_value=None), mock.patch.object(probe.MockSink, 'pci_chain', lambda self: []):
            with self.assertRaisesRegex(probe.ProbeError, 'no PCI function'):
                probe.run_write_probe(card, sink, self.parsed())
            self.assertEqual(card.events, [])
            report = probe.run_write_probe(card, sink, self.parsed('--allow-unknown-link'))
        sink.close()
        self.assertTrue(any('no PCI function' in note for note in report['notes']))

    def test_a_range_beyond_the_device_is_refused(self):
        card = self.card()
        with self.assertRaisesRegex(probe.ProbeError, 'beyond'):
            self.run_with(card, '--scratch-start', '48M', '--scratch-size', '24M')
        self.assertEqual(card.events, [])

    def test_a_stale_sentinel_matters(self):
        """The range already holds exactly what the first write should put there; a write that never happens must still fail, because the
        probe paints the range with a sentinel first."""
        pattern = probe.bank_pattern(1, 0, SHARD)
        with open(self.target, 'r+b') as handle:
            handle.seek(MIB)
            handle.write(pattern)
        report = self.report('--i-own-this-scratch-range', '--qd', '1', fault='drop', keep=True)
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(report['aborted']['kind'], 'mismatch')
        self.assertTrue(report['mismatch']['found_is_sentinel'])

    def test_a_write_that_lands_a_byte_wrong_is_found_and_located(self):
        report = self.report(fault='flip')
        self.assertEqual(report['result'], 'FAIL')
        self.assertFalse(report['verdict']['byte_exact'])
        self.assertFalse(report['verdict']['go'])
        mismatch = report['mismatch']
        self.assertEqual(mismatch['drive_offset'], MIB + SHARD // 2)
        self.assertEqual(mismatch['bad_pages'], 1)
        self.assertEqual(mismatch['pages'], 170)
        self.assertFalse(mismatch['found_is_sentinel'])
        self.assertEqual(mismatch['found'], mismatch['expected'] ^ 0xFF)

    def test_a_write_that_never_arrives_is_the_sentinel(self):
        report = self.report(fault='drop')
        self.assertEqual(report['result'], 'FAIL')
        self.assertTrue(report['mismatch']['found_is_sentinel'])
        self.assertEqual(report['mismatch']['bad_pages'], 170)

    def test_the_run_stops_at_the_first_difference(self):
        report = self.report('--qd', '1,2,4', fault='flip')
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(len(report['sweep']), 0)                      # the first cell is the one that failed

    def test_a_window_that_does_not_move_is_found_before_any_data_goes_in(self):
        card = self.card(fault='stale-aim')
        report, sink = self.run_with(card, '--reaim')
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(report['aborted']['kind'], 'canary')
        self.assertIn('not where it was aimed', report['aborted']['message'])
        self.assertEqual(sink.written, 0)                              # nothing reached the drive
        self.assertNotIn('fill', report)
        self.assertFalse(report['aims'][-1]['ok'])
        self.assertIn(7, report['aims'][-1]['matches_bank'])          # the last canary written is the one still showing
        self.assertFalse(report['verdict']['reaim_canary_ok'])

    def test_a_window_that_goes_stale_late_is_still_found_before_any_data_goes_into_any_bank(self):
        card = self.card(fault='stale-aim')
        card.stale_after = 10              # 1 initial configure, 8 canary writes, then the first verifying aim works and the next does not
        report, sink = self.run_with(card, '--reaim')
        self.assertEqual(report['aborted']['kind'], 'canary')
        self.assertEqual(sink.written, 0)
        self.assertNotIn('fill', report)
        for bank in card.banks:                                        # no bank holds anything but (at most) a canary page
            self.assertEqual(bytes(bank[probe.PAGE:probe.PAGE + 4096]), bytes(4096))

    def test_a_bank_that_does_not_keep_what_is_written_into_it_is_found_before_the_drive_is_written(self):
        card = self.card()
        card.write_limit[3] = probe.PAGE + 8192       # bank 3 keeps the canary page and two data pages, the rest of its region is lost
        report, sink = self.run_with(card, '--reaim')
        self.assertEqual(report['aborted']['kind'], 'fill')
        self.assertIn('bank 3 does not read back', report['aborted']['message'])
        self.assertEqual(sink.written, 0)
        self.assertEqual(report['result'], 'FAIL')

    def test_the_cpu_mapping_asked_for_is_the_one_in_the_flag(self):
        card = self.card()
        self.run_with(card)
        self.assertEqual(card.cpu_maps, ['uc'])
        card = self.card()
        self.run_with(card, '--cpu-map', 'wc')
        self.assertEqual(card.cpu_maps, ['wc'])

    def test_a_dma_that_reads_another_bank_than_the_window_shows_is_found_by_the_bytes(self):
        report = self.report('--reaim', fault=None, card=self.card(fault='wrong-bank'))
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(report['aborted']['kind'], 'mismatch')
        self.assertNotEqual(report['mismatch']['bank'], 0)             # bank 0 is where it reads from, so only the others differ
        self.assertTrue(all(arm['verify']['ok'] for arm in report['sweep']))

    def test_every_bank_is_visited_and_written_from(self):
        report = self.report('--no-sweep', '--reaim', '--reaim-rounds', '16')
        self.assertEqual(report['result'], 'PASS')
        self.assertEqual(report['reaim']['rounds_per_bank'], {str(bank): 2 for bank in range(8)})
        self.assertEqual(report['reaim']['writes_per_round'], 2)
        self.assertGreaterEqual(len(report['aims']), 16)
        self.assertEqual(report['verdict']['reaim_canary_ok'], True)

    def test_banks_can_be_chosen(self):
        report = self.report('--no-sweep', '--reaim', '--banks', '1,6', '--reaim-rounds', '4')
        self.assertEqual(report['reaim']['banks'], [1, 6])
        self.assertEqual(report['plan']['banks'].keys(), {'0', '1', '6'})

    def test_writes_from_the_same_card_slots_are_verified_against_the_right_pattern(self):
        # 8 writes in flight from a card region that holds 4 slots of the write size: slots are reused modulo 4
        report = self.report('--sizes', str(MIB), '--qd', '8', '--region', '4M')
        self.assertEqual(report['sweep'][0]['card_slots'], 4)
        self.assertEqual(report['result'], 'PASS')

    def test_the_writes_of_a_batch_run_concurrently(self):
        state = {'now': 0, 'peak': 0}
        lock = threading.Lock()

        def hook(sink):
            real = sink.write_iov

            def slow(views, offset):
                if len(views[0]) == SHARD:                              # not the sentinel painting
                    with lock:
                        state['now'] += 1
                        state['peak'] = max(state['peak'], state['now'])
                    time.sleep(0.05)
                    with lock:
                        state['now'] -= 1
                return real(views, offset)
            sink.write_iov = slow

        report = self.report('--qd', '4', hook=hook)
        self.assertEqual(report['result'], 'PASS')
        self.assertEqual(state['peak'], 4)

    def test_a_write_is_split_into_the_iovecs_asked_for(self):
        seen = []

        def hook(sink):
            real = sink.write_iov

            def spy(views, offset):
                if len(views) > 1:
                    seen.append([len(view) for view in views])
                return real(views, offset)
            sink.write_iov = spy

        report = self.report('--sizes', str(probe.SUBRUN_BYTES), '--qd', '1', '--scratch-size', '24M', '--region', '24M', hook=hook,
                             card=self.card(bank_bytes=32 * MIB))
        self.assertEqual(report['result'], 'PASS')
        self.assertEqual(report['sweep'][0]['iovecs'], 32)
        self.assertEqual(seen[0], [SHARD] * 32)
        with_one = self.report('--sizes', str(SHARD * 4), '--qd', '1', '--iov-bytes', '0')
        self.assertEqual(with_one['sweep'][0]['iovecs'], 1)

    def test_an_uncorrectable_aer_error_mid_run_stops_the_dma(self):
        card = self.card()
        calls = {'n': 0}

        def hook(sink):
            real = sink.write_iov

            def writer(views, offset):
                calls['n'] += 1
                done = real(views, offset)
                if calls['n'] == 3:
                    bump(card.sysfs['drive_chain'][2], 'aer_dev_nonfatal', 'CmpltTO')
                return done
            sink.write_iov = writer

        report, sink = self.run_with(card, '--qd', '1,2,4', '--aer-interval', '0', hook=hook)
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(report['aborted']['kind'], 'aer')
        self.assertIn('CmpltTO', report['aborted']['message'])
        self.assertFalse(report['verdict']['no_uncorrectable_aer'])
        self.assertEqual(report['aer']['delta']['uncorrectable'], 1)
        self.assertLessEqual(calls['n'], 5)                             # and no more was sent

    def test_correctable_errors_are_recorded_and_put_to_the_owner_not_failed(self):
        card = self.card()

        def hook(sink):
            real = sink.write_iov

            def writer(views, offset):
                done = real(views, offset)
                bump(card.sysfs['card_chain'][1], 'aer_dev_correctable', 'BadTLP')
                return done
            sink.write_iov = writer

        report, _ = self.run_with(card, hook=hook)
        self.assertEqual(report['result'], 'PASS')
        self.assertGreater(report['aer']['delta']['correctable'], 0)
        self.assertEqual(report['aer']['delta']['uncorrectable'], 0)
        self.assertIn('0000:01:00.0', report['aer']['delta']['by_function'])

    def test_a_system_without_aer_counters_cannot_pass(self):
        card = self.card()
        for chain in card.sysfs.values():
            for directory in (chain if isinstance(chain, list) else []):
                for name in probe.AER_DEV_FILES + probe.AER_ROOT_FILES:
                    if os.path.exists(os.path.join(directory, name)):
                        os.unlink(os.path.join(directory, name))
        report, _ = self.run_with(card)
        self.assertEqual(report['result'], 'UNVERIFIED')
        self.assertIsNone(report['verdict']['no_uncorrectable_aer'])
        self.assertFalse(report['verdict']['go'])

    def test_a_drive_that_goes_away_stops_the_dma(self):
        card = self.card()
        calls = {'n': 0}

        def hook(sink):
            real = sink.write_iov

            def writer(views, offset):
                calls['n'] += 1
                if calls['n'] == 4:
                    set_text(os.path.join(os.path.dirname(card.sysfs['block_dir']), 'state'), 'resetting')
                return real(views, offset)
            sink.write_iov = writer

        report, _ = self.run_with(card, '--qd', '1,2,4', '--aer-interval', '0', hook=hook)
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(report['aborted']['kind'], 'drive')
        self.assertIn('resetting', report['aborted']['message'])

    def test_a_drive_that_is_not_live_is_not_started_on(self):
        card = self.card()
        set_text(os.path.join(os.path.dirname(card.sysfs['block_dir']), 'state'), 'dead')
        with self.assertRaisesRegex(probe.ProbeError, 'not live'):
            self.run_with(card)
        self.assertEqual(card.events, [])

    def test_the_write_budget_is_a_hard_stop(self):
        report = self.report('--qd', '1,2,4', '--max-write-gib', '0.001')
        self.assertEqual(report['result'], 'FAIL')
        self.assertEqual(report['aborted']['kind'], 'budget')
        self.assertLessEqual(report['bytes']['written_to_drive'], 0.001 * (1 << 30))

    def test_without_verification_the_run_is_unverified_and_nothing_is_painted(self):
        report = self.report('--write-verify', 'none')
        self.assertEqual(report['result'], 'UNVERIFIED')
        self.assertIsNone(report['verdict']['byte_exact'])
        self.assertEqual(report['bytes']['sentinel'], 0)
        report = self.report('--write-verify', 'none', fault='drop')
        self.assertEqual(report['result'], 'UNVERIFIED')                 # a write that never happened is not seen: that is what verifying is for

    def test_the_rate_gate(self):
        report = self.report('--min-write-gbps', '1000000')
        self.assertEqual(report['result'], 'FAIL')
        self.assertTrue(report['below_min_gbps'])
        self.assertFalse(report['verdict']['p2w_gbps_ok'])
        self.assertIsNone(report['aborted'])

    def test_the_drain_bound_is_one_sub_run_on_its_own(self):
        def slow(delay):
            def hook(sink):
                real = sink.write_iov

                def writer(views, offset):
                    if len(views) == 32:
                        time.sleep(delay)
                    return real(views, offset)
                sink.write_iov = writer
            return hook

        extra = ('--sizes', str(probe.SUBRUN_BYTES), '--qd', '1', '--region', '24M', '--drain-ms', '50')
        report = self.report(*extra, hook=slow(0.08))
        self.assertEqual(report['result'], 'FAIL')
        self.assertFalse(report['verdict']['subrun_drain_ok'])
        self.assertGreater(report['verdict']['subrun_drain_ms'], 50)
        self.assertGreaterEqual(report['verdict']['subrun_drain_max_ms'], report['verdict']['subrun_drain_p99_ms'])
        report = self.report(*extra[:-1], '5000', hook=slow(0.0))
        self.assertEqual(report['result'], 'PASS')
        self.assertTrue(report['verdict']['subrun_drain_ok'])
        # without that cell there is no drain figure
        self.assertNotIn('subrun_drain_ok', self.report()['verdict'])

    def test_a_single_slow_write_is_information_the_drain_gate_is_bytes_over_the_cells_rate(self):
        calls = {'n': 0}

        def hook(sink):
            real = sink.write_iov

            def writer(views, offset):
                if len(views) == 32:                               # a sub-run write, not a sentinel paint or a read-back
                    calls['n'] += 1
                    if calls['n'] == 1:
                        time.sleep(0.3)                            # the cold first write, or one garbage-collection pause
                return real(views, offset)
            sink.write_iov = writer

        report = self.report('--sizes', str(probe.SUBRUN_BYTES), '--qd', '1', '--region', '24M', '--min-batches', '30', '--drain-ms', '100', hook=hook)
        verdict = report['verdict']
        self.assertGreater(verdict['subrun_drain_max_ms'], 300)               # reported ...
        self.assertLess(verdict['subrun_drain_ms'], 100)                       # ... not gated on: 22 MB over the rate of all 30 writes
        self.assertTrue(verdict['subrun_drain_ok'])
        self.assertEqual(report['result'], 'PASS')

    def test_the_soak_is_paced_verified_and_ends_on_a_verified_cycle(self):
        report = self.report('--no-sweep', '--soak', '0.03', '--soak-max-gbps', '0.01', '--soak-verify-every', '3', '--aer-interval', '0.1')
        soak = report['soak']
        self.assertEqual(report['result'], 'PASS')
        self.assertEqual(soak['paced_to_gbps'], 0.01)
        self.assertLess(soak['cycles'], 200)
        self.assertGreaterEqual(soak['verify']['cycles_verified'], soak['cycles'] // 3)
        self.assertGreaterEqual(soak['effective_gbps'] * 1e9 * soak['wall_seconds'], soak['bytes'] * 0.99)
        self.assertLess(soak['effective_gbps'], 0.05)
        self.assertGreater(soak['gbps'], soak['effective_gbps'])
        self.assertGreaterEqual(len(report['aer']['samples']), 2)
        self.assertEqual(len(soak['bucket_gbps']), 1)

    def test_a_soak_with_no_verified_cycle_does_not_pass_and_the_last_cycle_is_always_verified(self):
        report = self.report('--no-sweep', '--soak', '0.005', '--soak-verify-every', '100000')
        soak = report['soak']
        self.assertEqual(soak['verify']['cycles_verified'], 1)               # only the last
        self.assertEqual(soak['verify']['ok'], True)
        self.assertEqual(report['result'], 'PASS')
        report = self.report('--no-sweep', '--soak', '0.005', '--write-verify', 'none')
        self.assertIsNone(report['soak']['verify']['ok'])
        self.assertEqual(report['result'], 'UNVERIFIED')

    def test_the_soak_rate_is_the_verdicts_when_there_is_one(self):
        report = self.report('--soak', '0.005', '--min-write-gbps', '0.0001')
        self.assertEqual(report['verdict']['p2w_sustained_gbps'], report['soak']['gbps'])

    def test_cpu_time_is_reported_for_the_process_and_the_workers(self):
        report = self.report('--qd', '4')
        cpu = report['sweep'][0]['cpu']
        self.assertGreaterEqual(cpu['process_seconds'], 0)
        self.assertGreaterEqual(cpu['worker_seconds'], 0)
        self.assertGreaterEqual(cpu['busiest_worker_seconds'], 0)
        self.assertLessEqual(cpu['busiest_worker_seconds'], cpu['worker_seconds'] + 1e-9)

    def burn_in_verify(self, seconds):
        """Make every read-back compare spend `seconds` of CPU, as a large verify does; returns the total burned (a list cell)."""
        burned = [0.0]
        real = probe.WriteRun.verify

        def verify(run, *args, **kw):
            end = time.process_time() + seconds
            while time.process_time() < end:
                pass
            burned[0] += seconds
            return real(run, *args, **kw)
        patcher = mock.patch.object(probe.WriteRun, 'verify', verify)
        patcher.start()
        self.addCleanup(patcher.stop)
        return burned

    def test_cpu_counts_the_writes_only_not_the_verification(self):
        burned = self.burn_in_verify(0.1)
        report = self.report('--qd', '4,8')
        for arm in report['sweep']:
            cpu = arm['cpu']
            self.assertLess(cpu['process_seconds'], 0.5 * 0.1 * arm['batches'], arm)                     # the 0.1 s per batch is not in it
            self.assertGreaterEqual(cpu['non_write_process_seconds'], 0.8 * 0.1 * arm['batches'], arm)    # it is here
            self.assertAlmostEqual(cpu['process_fraction_of_write_time'], cpu['process_seconds'] / arm['seconds'], places=6)
        self.assertGreater(burned[0], 0)

    def test_the_soaks_cpu_counts_the_writes_only_too(self):
        self.burn_in_verify(0.05)
        report = self.report('--no-sweep', '--soak', '0.01', '--soak-verify-every', '1')
        soak = report['soak']
        self.assertLess(soak['cpu']['process_seconds'], 0.5 * 0.05 * soak['verify']['cycles_verified'])
        self.assertGreaterEqual(soak['cpu']['non_write_process_seconds'], 0.8 * 0.05 * soak['verify']['cycles_verified'])
        self.assertAlmostEqual(soak['cpu']['process_fraction_of_write_time'], soak['cpu']['process_seconds'] / soak['write_seconds'], places=6)

    def test_the_cpu_of_the_writes_is_what_the_workers_spent_in_them(self):
        def hook(sink):
            real = sink.write_iov

            def writer(views, offset):
                end = time.thread_time() + 0.02
                while time.thread_time() < end:
                    pass
                return real(views, offset)
            sink.write_iov = writer
        report = self.report('--qd', '4', hook=hook)
        arm = report['sweep'][0]
        self.assertGreaterEqual(arm['cpu']['worker_seconds'], 0.8 * 0.02 * arm['writes'])
        self.assertGreaterEqual(arm['cpu']['process_seconds'], 0.8 * 0.02 * arm['writes'])               # the same CPU, seen from the process
        self.assertLess(arm['cpu']['busiest_worker_seconds'], arm['cpu']['worker_seconds'])

    def test_the_report_says_what_cpu_and_queue_depth_mean(self):
        notes = ' '.join(self.report()['notes'])
        for text in ('pwritev batches', 'batch-synchronous', 'count the same error'):
            self.assertIn(text, notes)

    def test_aer_is_sampled_often_enough_to_act_within_about_a_second(self):
        self.assertLessEqual(probe.build_parser().parse_args(['--mode', 'write']).aer_interval, 0.5)
        report = self.report('--qd', '1,4')
        self.assertGreaterEqual(report['aer']['sample_count'], 2)               # at least once per cell
        self.assertEqual(report['aer']['interval_seconds'], 0.5)
        self.assertGreater(report['aer']['mean_check_ms'], 0)
        self.assertLess(report['aer']['max_gap_seconds'], 5.0)
        self.assertLessEqual(len(report['aer']['samples']), report['aer']['sample_count'])

    def test_an_uncorrectable_error_in_a_long_cell_stops_the_dma_within_the_interval(self):
        card = self.card()
        seen = {'writes': 0, 'bumped': None}

        def hook(sink):
            real = sink.write_iov

            def writer(views, offset):
                seen['writes'] += 1
                time.sleep(0.01)
                done = real(views, offset)
                if seen['writes'] == 3:
                    bump(card.sysfs['drive_chain'][1], 'aer_dev_nonfatal', 'CmpltTO')
                    seen['bumped'] = time.monotonic()
                return done
            sink.write_iov = writer

        # one cell that would run for 5 s of write time; the default interval is not used here, a small one keeps the test short
        report, _ = self.run_with(card, '--sizes', str(SHARD), '--qd', '1', '--arm-seconds', '5', '--aer-interval', '0.1', hook=hook)
        stopped = time.monotonic() - seen['bumped']
        self.assertEqual(report['aborted']['kind'], 'aer')
        self.assertLess(stopped, 1.0)
        self.assertLess(seen['writes'] - 3, 30)                                   # about 0.1 s of writes of about 10 ms, not 5 s of them

    def test_the_aer_gap_is_counted_from_the_first_dma_not_from_the_start_of_the_run(self):
        real = probe.fill_banks

        def slow_fill(*args, **kw):
            time.sleep(0.4)                                       # a slow CPU fill through an uncached window
            return real(*args, **kw)

        with mock.patch.object(probe, 'fill_banks', slow_fill):
            report = self.report('--qd', '1,4')
        self.assertGreaterEqual(report['fill_stats']['seconds'], 0.4)
        self.assertLess(report['aer']['max_gap_seconds'], 0.3)

    def test_the_card_fill_is_timed(self):
        fill = self.report()['fill_stats']
        self.assertEqual(fill['cpu_map'], 'uc')
        self.assertGreater(fill['bytes'], 4 * MIB)
        self.assertGreater(fill['seconds'], 0)
        self.assertGreater(fill['mbps'], 0)

    def test_the_card_must_be_free(self):
        card = self.card(in_use=1)
        with self.assertRaisesRegex(probe.ProbeError, 'already allocated'):
            self.run_with(card)
        self.assertEqual(card.events, [])
        card = self.card(openers=[4242])
        with self.assertRaisesRegex(probe.ProbeError, '4242'):
            self.run_with(card)
        self.assertEqual(card.events, [])
        self.assertEqual(self.run_with(card, '--allow-shared')[0]['result'], 'PASS')

    def test_the_window_is_given_back(self):
        card = self.card()
        self.report(card=card)
        self.assertEqual(card.closed, [probe.FIRST_4G_TLB])
        card = self.card(fault='stale-aim')
        self.run_with(card)
        self.assertEqual(card.closed, [probe.FIRST_4G_TLB])

    def test_an_unpinned_window_is_refused_unless_allowed(self):
        card = self.card()
        card.pinnable = False
        with self.assertRaisesRegex(probe.ProbeError, 'pinned'):
            self.run_with(card)
        self.assertNotIn('p2p_buffer', card.events)
        self.assertFalse(self.run_with(card, '--allow-unpinned')[0]['pinned'])

    def test_a_destination_in_the_top_of_the_window_is_refused_before_anything_happens(self):
        card = self.card(bank_bytes=32 * MIB)
        with self.assertRaisesRegex(probe.ProbeError, 'outside the published slice|not DRAM'):
            self.run_with(card, '--dest-offset', '31M', '--region', '4M')
        self.assertEqual(card.events, [])


class VerdictTest(unittest.TestCase):
    """finish_write on reports made by hand: every condition of the go/no-go on its own."""

    def args(self, *extra):
        args = probe.build_parser().parse_args(['--mode', 'write', '--target', 'x'] + list(extra))
        probe.resolve_target(args)
        return args

    def arm(self, size=SHARD, qd=1, gbps=1.0, max_ms=1.0, ok=True, p99_ms=None):
        return {'size': size, 'qd': qd, 'gbps': gbps, 'write_max_ms': max_ms, 'write_p99_ms': max_ms if p99_ms is None else p99_ms, 'verify': {'ok': ok}}

    def report(self, sweep=(), reaim=None, soak=None, aborted=None, measured=True, uncorrectable=0):
        return {'sweep': list(sweep), 'reaim': reaim, 'soak': soak, 'aborted': aborted,
                'aer': {'delta': {'measured': measured, 'uncorrectable': uncorrectable, 'correctable': 0, 'by_function': {}}}}

    def verdict(self, report, *extra):
        out = probe.finish_write(report, self.args(*extra))
        return out['result'], out['verdict']

    def test_everything_holding_is_a_pass(self):
        result, verdict = self.verdict(self.report([self.arm(), self.arm(probe.SUBRUN_BYTES, 1, 2.0, 30.0)]))
        self.assertEqual(result, 'PASS')
        self.assertTrue(verdict['go'])
        self.assertAlmostEqual(verdict['subrun_drain_ms'], probe.SUBRUN_BYTES / 2.0e9 * 1e3, places=6)         # bytes / the cell's rate, not the slowest write
        self.assertEqual((verdict['subrun_drain_ok'], verdict['p2w_sustained_gbps']), (True, 1.0))
        self.assertEqual((verdict['subrun_drain_max_ms'], verdict['subrun_drain_p99_ms']), (30.0, 30.0))

    def test_a_verify_that_failed_without_stopping_is_a_fail(self):
        result, verdict = self.verdict(self.report([self.arm(), self.arm(ok=False)]))
        self.assertEqual((result, verdict['byte_exact'], verdict['go']), ('FAIL', False, False))

    def test_nothing_verified_is_unverified_not_a_pass(self):
        result, verdict = self.verdict(self.report([self.arm(ok=None)]), '--write-verify', 'none')
        self.assertEqual((result, verdict['byte_exact']), ('UNVERIFIED', None))
        self.assertEqual(self.verdict(self.report([]))[0], 'UNVERIFIED')

    def test_aer(self):
        self.assertEqual(self.verdict(self.report([self.arm()], measured=False)), ('UNVERIFIED', mock.ANY))
        result, verdict = self.verdict(self.report([self.arm()], uncorrectable=1))
        self.assertEqual((result, verdict['no_uncorrectable_aer']), ('FAIL', False))

    def test_the_rate_is_the_soaks_when_there_is_one_and_the_slowest_cell_otherwise(self):
        slow, fast = [self.arm(gbps=0.1), self.arm(gbps=5.0)], {'gbps': 3.0, 'verify': {'ok': True}}
        result, verdict = self.verdict(self.report(slow))
        self.assertEqual((result, verdict['p2w_sustained_gbps'], verdict['p2w_gbps_ok']), ('FAIL', 0.1, False))
        result, verdict = self.verdict(self.report(slow, soak=fast))
        self.assertEqual((result, verdict['p2w_sustained_gbps']), ('PASS', 3.0))
        self.assertEqual(self.verdict(self.report(slow, soak={'gbps': 0.1, 'verify': {'ok': True}}))[0], 'FAIL')
        self.assertEqual(self.verdict(self.report(slow), '--min-write-gbps', '0.05')[0], 'PASS')

    def test_a_slow_queue_depth_one_cell_is_the_drain_figure_not_the_sustained_rate(self):
        cells = [self.arm(qd=1, gbps=0.05), self.arm(qd=4, gbps=2.0), self.arm(qd=32, gbps=4.0)]
        result, verdict = self.verdict(self.report(cells))
        self.assertEqual((result, verdict['p2w_sustained_gbps']), ('PASS', 2.0))
        result, verdict = self.verdict(self.report([self.arm(qd=1, gbps=0.05)]))
        self.assertEqual((result, verdict['p2w_sustained_gbps']), ('FAIL', 0.05))               # nothing deeper was run: the only figure there is

    def test_the_drain_bound_is_the_sub_run_on_its_own_at_the_rate_it_measured(self):
        slow = self.arm(probe.SUBRUN_BYTES, 1, 0.3, 80.0)                      # 22,282,240 B at 0.3 GB/s is 74 ms
        result, verdict = self.verdict(self.report([slow]))
        self.assertEqual((result, verdict['subrun_drain_ok']), ('FAIL', False))
        self.assertAlmostEqual(verdict['subrun_drain_ms'], 74.27, places=1)
        self.assertEqual(self.verdict(self.report([slow]), '--drain-ms', '100')[0], 'PASS')
        for cell in (self.arm(probe.SUBRUN_BYTES, 4, 2.0, 600.0), self.arm(probe.OBJECT_BYTES, 1, 2.0, 600.0)):
            self.assertNotIn('subrun_drain_ok', self.verdict(self.report([cell]))[1])

    def test_one_slow_write_among_many_is_reported_and_not_gated(self):
        # a hiccup of 600 ms in a cell that ran at 2 GB/s (11 ms per sub-run): the design's bound is bytes / rate, which holds
        cell = self.arm(probe.SUBRUN_BYTES, 1, 2.0, max_ms=600.0, p99_ms=400.0)
        result, verdict = self.verdict(self.report([cell]))
        self.assertEqual((result, verdict['subrun_drain_ok'], verdict['go']), ('PASS', True, True))
        self.assertEqual((verdict['subrun_drain_max_ms'], verdict['subrun_drain_p99_ms']), (600.0, 400.0))

    def test_the_canary_after_every_re_aim(self):
        reaim = {'rounds': 8, 'canary_ok': 8, 'verify': {'ok': True}}
        self.assertEqual(self.verdict(self.report([self.arm()], reaim=reaim), '--reaim')[0], 'PASS')
        self.assertEqual(self.verdict(self.report([self.arm()], reaim=dict(reaim, canary_ok=7)), '--reaim')[0], 'FAIL')
        self.assertEqual(self.verdict(self.report([self.arm()]), '--reaim')[0], 'UNVERIFIED')       # the arm never ran
        stopped = {'kind': 'canary', 'message': 'x'}
        self.assertEqual(self.verdict(self.report([self.arm()], aborted=stopped), '--reaim')[1]['reaim_canary_ok'], False)

    def test_a_stopped_run_never_passes(self):
        for kind in ('mismatch', 'canary', 'aer', 'drive', 'budget', 'error', 'fill'):
            result, verdict = self.verdict(self.report([self.arm()], aborted={'kind': kind, 'message': 'x'}))
            self.assertEqual((result, verdict['go']), ('FAIL', False), kind)


class LinkRefusalTest(WriteFixture):
    def test_a_drive_at_gen1_is_refused_before_the_card_is_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = self.card(sysfs=probe.make_fake_sysfs(tmp, drive_gts=2.5))
            with self.assertRaisesRegex(probe.ProbeError, 'Gen1'):
                self.run_with(card)
            self.assertEqual(card.events, [])
            with self.assertRaisesRegex(probe.ProbeError, 'Gen1'):
                self.run_with(card, '--allow-unknown-link', '--min-drive-gts', '0')

    def test_a_drive_below_the_floor_is_refused_and_the_floor_can_be_lowered(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = self.card(sysfs=probe.make_fake_sysfs(tmp, drive_gts=8.0))
            with self.assertRaisesRegex(probe.ProbeError, 'below the'):
                self.run_with(card)
            self.assertEqual(self.run_with(card, '--min-drive-gts', '8')[0]['result'], 'PASS')

    def test_a_card_at_gen1_is_refused_after_it_is_woken_and_before_its_window_is_taken(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = self.card(sysfs=probe.make_fake_sysfs(tmp, card_gts=2.5))
            with self.assertRaisesRegex(probe.ProbeError, 'Gen1'):
                self.run_with(card)
            self.assertEqual(card.events, ['power_up'])

    def test_a_link_in_the_middle_of_the_chain_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = probe.make_fake_sysfs(tmp)
            set_text(os.path.join(tree['drive_chain'][1], 'current_link_speed'), '2.5 GT/s PCIe')
            card = self.card(sysfs=tree)
            with self.assertRaisesRegex(probe.ProbeError, '0000:02:00.0'):
                self.run_with(card)

    def test_links_are_in_the_report(self):
        report = self.report()
        self.assertEqual([link['speed_gts'] for link in report['links']['drive']], [16.0] * 4)
        self.assertEqual([link['speed_gts'] for link in report['links']['card']], [32.0] * 2)


class CommandLineTest(WriteFixture):
    def test_the_mock_run_through_main_prints_json_and_writes_the_report_file(self):
        out_path = os.path.join(self.tmp.name, 'report.json')
        code, out, err = run_main(self.argv('--json', '--json-out', out_path))
        self.assertEqual((code, err), (0, ''))
        report = json.loads(out)
        self.assertEqual(report['result'], 'PASS')
        with open(out_path) as handle:
            self.assertEqual(json.load(handle), report)

    def test_the_report_file_goes_to_the_user_who_ran_sudo(self):
        path = os.path.join(self.tmp.name, 'owned.json')
        env = {'SUDO_UID': '1234', 'SUDO_GID': '5678'}
        with mock.patch.dict(os.environ, env), mock.patch.object(os, 'geteuid', return_value=0), mock.patch.object(os, 'fchown') as chown:
            probe.write_json_report(path, {'a': 1})
        self.assertEqual(chown.call_count, 1)
        self.assertEqual(chown.call_args[0][1:], (1234, 5678))
        with open(path) as handle:
            self.assertEqual(json.load(handle), {'a': 1})

    def test_the_report_file_is_left_alone_when_not_root_or_not_through_sudo_or_when_chown_fails(self):
        path = os.path.join(self.tmp.name, 'plain.json')
        env = {'SUDO_UID': '1234', 'SUDO_GID': '5678'}
        with mock.patch.dict(os.environ, env), mock.patch.object(os, 'geteuid', return_value=1000), mock.patch.object(os, 'fchown') as chown:
            probe.write_json_report(path, {'a': 1})
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(os, 'geteuid', return_value=0), mock.patch.object(os, 'fchown') as chown2:
            probe.write_json_report(path, {'a': 2})
        self.assertEqual((chown.call_count, chown2.call_count), (0, 0))
        with mock.patch.dict(os.environ, env), mock.patch.object(os, 'geteuid', return_value=0), mock.patch.object(os, 'fchown', side_effect=PermissionError):
            probe.write_json_report(path, {'a': 3})
        with open(path) as handle:
            self.assertEqual(json.load(handle), {'a': 3})

    def test_the_text_report_ends_in_a_verdict_and_names_the_range(self):
        code, out, _ = run_main(self.argv('--reaim', '--soak', '0.005'))
        self.assertEqual(code, 0)
        for text in ('write arm:', 'scratch:', 'links drive', 're-aim 8 rounds', 'soak 0.0', 'aer: measured', 'aer sampling:', 'card fill:', 'verdict:', 'RESULT PASS'):
            self.assertIn(text, out)
        self.assertEqual(out.count('RESULT'), 1)

    def test_exit_codes(self):
        self.assertEqual(run_main(self.argv('--mock-fault', 'flip'))[0], 1)
        with open(self.target, 'wb') as handle:
            handle.truncate(self.TARGET_BYTES)
        self.assertEqual(run_main(self.argv('--write-verify', 'none'))[0], 3)
        code, _, err = run_main(self.argv())
        self.assertEqual(code, 2)                                          # the previous run left data in the range
        self.assertIn('not blank', err)

    def test_a_gate_missed_on_a_clean_run_is_the_only_thing_the_neutral_gates_forgive(self):
        """The W1 kit runs the probe with --min-write-gbps 0 --drain-ms 1e9 and applies the gates itself in its summary, so a slow drive costs
        no data: such a run exits 0. Every safety failure still ends the run (exit 1) or is not a pass (exit 3)."""
        slow = ('--sizes', str(probe.SUBRUN_BYTES), '--qd', '1', '--region', '24M')
        gated, neutral = slow + ('--min-write-gbps', '0.25', '--drain-ms', '50'), slow + ('--min-write-gbps', '0', '--drain-ms', '1e9')
        real = probe.MockSink.write_iov
        state = {'bump': False}

        def slow_write(sink, views, offset):
            done = real(sink, views, offset)
            if len(views) == 32:
                time.sleep(0.1)                                   # 22 MB in 0.1 s: 0.22 GB/s and a 100 ms sub-run
                if state['bump']:
                    bump(sink.card.sysfs['card_chain'][1], 'aer_dev_nonfatal', 'CmpltTO')
            return done

        with mock.patch.object(probe.MockSink, 'write_iov', slow_write):
            self.reset_target()
            self.assertEqual(run_main(self.argv(*gated, '--json'))[0], 1)                      # both gates missed: FAIL
            self.reset_target()
            code, out, _ = run_main(self.argv(*neutral, '--json'))
            report = json.loads(out)
            self.assertEqual(code, 0)
            self.assertEqual(report['result'], 'PASS')
            self.assertGreater(report['verdict']['subrun_drain_ms'], 50)                       # the figures the summary gates on are all there
            self.assertLess(report['verdict']['p2w_sustained_gbps'], 0.25)
            only_gates = ('--min-write-gbps', '0', '--drain-ms', '1e9')
            self.reset_target()
            self.assertEqual(run_main(self.argv(*only_gates, '--mock-fault', 'flip'))[0], 1)      # bytes that differ
            self.reset_target()
            self.assertEqual(run_main(self.argv(*only_gates, '--mock-fault', 'drop'))[0], 1)      # a write that never happened
            self.reset_target()
            self.assertEqual(run_main(self.argv(*only_gates, '--write-verify', 'none'))[0], 3)    # nothing compared
            self.reset_target()
            state['bump'] = True
            code, out, _ = run_main(self.argv(*neutral, '--json'))                                # an uncorrectable AER error
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)['aborted']['kind'], 'aer')

    def test_a_missing_target_is_an_error_not_a_traceback(self):
        code, _, err = run_main(['--mode', 'write', '--mock', '--target', os.path.join(self.tmp.name, 'nope.img'), '--scratch-start', '1M',
                                 '--scratch-size', '1M', '--dest-offset', '0', '--sizes', '4096', '--qd', '1'])
        self.assertEqual(code, 2)
        self.assertIn('nope.img', err)

    def test_the_read_arm_is_still_the_default(self):
        self.assertEqual(probe.build_parser().parse_args([]).mode, 'read')


if __name__ == '__main__':
    unittest.main()
