"""CardLink against a fake tt-kmd (``fake_card``): the open sequence, the canaries (written for every bank, then proved before any data), every
re-aim verified, the slot views and their bounds, the rules that keep a DMA from meeting a moving window, and the ways a card can disagree with its
staging record. No vLLM, no numpy, no device: stdlib only."""

import errno
import gc
import inspect
import logging
import mmap
import os
import re
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.environ.get('TTKVP2P_PACKAGE_ROOT') or os.path.dirname(HERE)
for entry in (HERE, PACKAGE_ROOT):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fake_card as fc  # noqa: E402
from ttkvp2p import cardlink as cl  # noqa: E402

logging.getLogger('ttkvp2p').addHandler(logging.NullHandler())

PAGE = 4096


class RigCase(unittest.TestCase):
    def rig(self, **kwargs):
        rig = fc.Rig(**kwargs)
        self.addCleanup(rig.close)
        return rig


class OpenTests(RigCase):
    def test_the_open_sequence_and_what_it_leaves(self):
        rig = self.rig()
        links = rig.open_links(nonce=b'n' * 16)
        for d, link in enumerate(links):
            board = rig.boards[d]
            self.assertTrue(link.proved)
            self.assertEqual(link.shard, d)
            self.assertEqual(link.index, board.index)
            self.assertEqual(link.banks, tuple(range(8)))
            self.assertEqual(link.slot_objects, fc.SMALL_SLOT)
            self.assertEqual(board.window_id, 202)
            self.assertEqual(board.pinned, {202})
            self.assertEqual(board.window_kind, 'uc', 'the CPU reaches the window through the uncached mapping')
            self.assertTrue(board.powered)
            self.assertTrue(board.slice_mapped)
            self.assertEqual(board.available, 0, 'mapping the whole slice leases it')
            self.assertEqual(link.aimed_bank, 7)
        calls = [name for name, shard in rig.kmd.log if shard == 0]
        self.assertEqual(calls[:4], ['power', 'allocate', 'configure', 'pin'], 'power state, window, a first aim, then the pin, before any mapping or canary')
        self.assertEqual(calls[4:], ['configure'] * 16, 'one aim per bank to write the canaries and one per bank to prove them')

    def test_every_banks_canary_page_is_written_distinct_and_proved_before_any_data(self):
        rig = self.rig()
        nonce = b'\x07' * 16
        rig.open_links(nonce=nonce)
        for board in rig.boards:
            seen = set()
            for bank in range(8):
                page = board.content(bank)[rig.canary_offset(bank):rig.canary_offset(bank) + PAGE]
                self.assertEqual(page, cl.canary_page(nonce, board.bdf, bank))
                seen.add(page[:8])
            self.assertEqual(len(seen), 8, 'no two banks share a canary')
            # nothing but canaries: the slot area of every bank is still zero
            for bank in range(8):
                data = board.content(bank)[rig.data_offset(bank, 0):rig.data_offset(bank, rig.slot_objects)]
                self.assertEqual(data, bytes(len(data)))

    def test_the_canary_depends_on_the_nonce_the_card_and_the_bank(self):
        base = cl.canary_page(b'a', 'x', 1)
        self.assertEqual(len(base), PAGE)
        self.assertNotEqual(base, cl.canary_page(b'b', 'x', 1))
        self.assertNotEqual(base, cl.canary_page(b'a', 'y', 1))
        self.assertNotEqual(base, cl.canary_page(b'a', 'x', 2))
        rig = self.rig()
        first = rig.make_link(0)
        second = rig.make_link(0)
        self.assertNotEqual(first._nonce, second._nonce, 'a fresh nonce per link: a canary left by an earlier process cannot satisfy the read-back')

    def test_a_lower_window_taken_on_the_way_is_given_back(self):
        rig = self.rig(window_id=203)
        links = rig.open_links()
        for board in rig.boards:
            self.assertEqual(board.taken, {203}, 'window 202 came up first and was freed again')
        self.assertTrue(all(link.proved for link in links))

    def test_a_window_held_by_another_process_is_refused_and_nothing_is_left_open(self):
        rig = self.rig(held_windows=(202,))
        link = rig.make_link(0)
        with self.assertRaises(cl.CardLinkUnavailable) as caught:
            link.open(rig.staging_info())
        self.assertIn('held by another process', str(caught.exception))
        self.assertEqual(rig.boards[0].taken, set(), 'the window that did come up was freed')
        self.assertEqual(rig.kmd.open_fds, 0)
        self.assertTrue(link.closed)

    def test_a_failure_at_any_device_step_closes_everything_again(self):
        for step, code in (('open', errno.EACCES), ('power', errno.EIO), ('configure', errno.EIO), ('pin', errno.ENOTTY), ('map_window', errno.EINVAL)):
            with self.subTest(step=step):
                rig = fc.Rig()
                try:
                    rig.kmd.fail[step] = code
                    link = rig.make_link(1)
                    with self.assertRaises(cl.CardLinkUnavailable):
                        link.open(rig.staging_info())
                    self.assertEqual(rig.kmd.open_fds, 0, 'no descriptor left')
                    self.assertEqual(rig.boards[1].taken, set(), 'no window left')
                    self.assertTrue(link.closed)
                finally:
                    rig.close()

    def test_a_second_mapping_of_the_slice_fails_which_is_the_lease(self):
        rig = self.rig()
        first = rig.make_link(0)
        first.open(rig.staging_info())
        rig.boards[0].taken.clear()          # (the second link is the same card in a second process: it gets its own window)
        rig.boards[0].window_id = None
        second = rig.make_link(0)
        with self.assertRaises(cl.CardLinkUnavailable) as caught:
            second.open(rig.staging_info())
        self.assertIn('already allocated', str(caught.exception), 'sysfs says so before anything is mapped')

    def test_a_canary_the_window_cannot_prove_stops_the_open(self):
        rig = self.rig()
        rig.boards[2].ignore_configure = True
        link = rig.make_link(2)
        with self.assertRaises(cl.CanaryMismatch):
            link.open(rig.staging_info())
        self.assertFalse(link.proved)
        self.assertEqual(rig.kmd.open_fds, 0)

    def test_two_banks_that_land_on_the_same_dram_are_found_while_only_canaries_are_written(self):
        rig = self.rig()
        rig.boards[0].alias = {3: 2}
        link = rig.make_link(0)
        with self.assertRaises(cl.CanaryMismatch) as caught:
            link.open(rig.staging_info())
        self.assertIn('bank 2', str(caught.exception))
        data = rig.boards[0].content(2)[rig.data_offset(2, 0):]
        self.assertEqual(data, bytes(len(data)), 'no data had been written')

    def test_a_write_that_never_reached_its_bank_is_not_taken_for_a_stale_canary_of_an_earlier_open(self):
        rig = self.rig()
        first = rig.make_link(0)
        first.open(rig.staging_info())
        first.close()
        rig.boards[0].ignore_banks = {5}      # the second open's aim at bank 5 does nothing: its canary write lands in bank 4
        second = rig.make_link(0)
        with self.assertRaises(cl.CanaryMismatch):
            second.open(rig.staging_info())


class HostCheckTests(RigCase):
    """Everything checked before the first device call: a disagreement leaves no ioctl in the log and no descriptor open."""

    def refused(self, rig, text, info=None, shard=0, **kwargs):
        link = rig.make_link(shard, **kwargs)
        with self.assertRaises(cl.CardLinkUnavailable) as caught:
            link.open(info if info is not None else rig.staging_info())
        self.assertIn(text, str(caught.exception))
        self.assertEqual(rig.kmd.log, [], 'no device call was made')
        self.assertEqual(rig.kmd.open_fds, 0)
        return caught.exception

    def test_the_chardev_must_be_the_card_at_the_given_pci_address(self):
        rig = self.rig()
        rig.kmd.links['%s/tenstorrent!%d/device' % (fc.CLASS_ROOT, rig.boards[0].index)] = '%s/%s' % (fc.PCI_ROOT, fc.BDFS[3])
        self.refused(rig, 'not %s' % fc.BDFS[0])

    def test_a_slice_that_is_already_allocated_is_refused(self):
        rig = self.rig()
        rig.kmd.files['%s/%s/p2pmem/available' % (fc.PCI_ROOT, fc.BDFS[0])] = '4096'
        self.refused(rig, 'already allocated')

    def test_sysfs_records_that_do_not_hold_together_are_refused(self):
        cases = {
            'bar': ('5', 'BAR5'),
            'tlb_first': ('203', 'windows 203'),
            'tlb_last': ('203', 'windows 202..203'),
            'size': ('4097', 'whole number of pages'),
        }
        for name, (value, text) in cases.items():
            with self.subTest(name=name):
                rig = fc.Rig()
                try:
                    rig.kmd.files['%s/tenstorrent!%d/p2pmem/%s' % (fc.CLASS_ROOT, rig.boards[0].index, name)] = value
                    if name == 'size':
                        rig.kmd.files['%s/%s/p2pmem/size' % (fc.PCI_ROOT, fc.BDFS[0])] = value
                        rig.kmd.files['%s/%s/p2pmem/available' % (fc.PCI_ROOT, fc.BDFS[0])] = value
                    self.refused(rig, text)
                finally:
                    rig.close()

    def test_a_published_slice_that_does_not_start_on_a_page_boundary_is_refused(self):
        rig = self.rig()
        path = '%s/tenstorrent!%d/p2pmem/offset' % (fc.CLASS_ROOT, rig.boards[0].index)
        rig.kmd.files[path] = str(int(rig.kmd.files[path]) + 64)
        self.refused(rig, '4 KiB boundary')

    def test_the_pci_cores_size_must_equal_the_drivers(self):
        rig = self.rig()
        rig.kmd.files['%s/%s/p2pmem/size' % (fc.PCI_ROOT, fc.BDFS[0])] = str(rig.region_size - PAGE)
        self.refused(rig, 'differs from the driver')

    def test_a_missing_sysfs_file_is_a_clean_refusal(self):
        rig = self.rig()
        del rig.kmd.files['%s/tenstorrent!%d/p2pmem/offset' % (fc.CLASS_ROOT, rig.boards[0].index)]
        self.refused(rig, 'patched tt-kmd is not loaded')

    def test_a_slice_that_does_not_cover_the_staging_banks_is_refused(self):
        rig = self.rig()
        smaller = rig.region_size - 2 * PAGE
        rig.kmd.files['%s/tenstorrent!%d/p2pmem/size' % (fc.CLASS_ROOT, rig.boards[0].index)] = str(smaller)
        rig.kmd.files['%s/%s/p2pmem/size' % (fc.PCI_ROOT, fc.BDFS[0])] = str(smaller)
        rig.kmd.files['%s/%s/p2pmem/available' % (fc.PCI_ROOT, fc.BDFS[0])] = str(smaller)
        self.refused(rig, 'outside the published slice')

    def test_the_top_16_mib_of_the_window_is_never_used(self):
        rig = self.rig(bank_address=0xFF000000 - 8 * PAGE + 64, region_base=0xFF000000 - 8 * PAGE)
        self.refused(rig, 'not DRAM')

    def test_a_staging_range_that_is_not_page_aligned_is_refused(self):
        rig = self.rig()
        info = rig.staging_info()
        info['aligned_4k_offset'] = 1
        self.refused(rig, 'not 4 KiB aligned', info=info)

    def test_a_bank_without_noc_coordinates_or_with_bad_ones_is_refused(self):
        rig = self.rig()
        info = rig.staging_info()
        info['banks'][3]['noc_x'] = None
        self.refused(rig, 'no NOC coordinates', info=info)
        info = rig.staging_info()
        info['banks'][3]['noc_y'] = 64
        self.refused(rig, '6-bit', info=info)
        info = rig.staging_info()
        info['banks'][3]['noc_x'], info['banks'][3]['noc_y'] = info['banks'][2]['noc_x'], info['banks'][2]['noc_y']
        self.refused(rig, 'share NOC coordinates', info=info)

    def test_a_bank_too_small_for_one_object_is_refused(self):
        rig = self.rig()
        info = rig.staging_info()
        info['aligned_4k_size'] = PAGE + fc.SMALL_SHARD - PAGE
        self.refused(rig, 'cannot hold', info=info)

    def test_the_shard_comes_from_the_staging_record_and_must_be_exactly_one(self):
        rig = self.rig()
        info = rig.staging_info()
        info['devices'][1]['pci_bus_id'] = fc.BDFS[0]                # two entries now name card 0 and none names card 1
        self.refused(rig, 'exactly one', info=info, shard=0)
        info = rig.staging_info()
        info['devices'] = []
        self.refused(rig, 'no devices', info=info)

    def test_the_kmd_index_is_the_fallback_when_sysfs_gave_no_pci_address(self):
        rig = self.rig()
        info = rig.staging_info()
        for entry in info['devices']:
            entry['pci_bus_id'] = None
        links = [rig.make_link(d) for d in range(4)]
        for d, link in enumerate(links):
            link.open(info)
            self.assertEqual(link.shard, d)

    def test_a_staging_record_that_contradicts_itself_about_the_kmd_index_is_refused(self):
        rig = self.rig()
        info = rig.staging_info()
        info['devices'][0]['pcie_device'] = 99
        self.refused(rig, '/dev/tenstorrent/99', info=info)

    def test_a_chardev_that_does_not_end_in_an_index_is_refused(self):
        rig = self.rig()
        link = cl.CardLink('/dev/tenstorrent/by-name', fc.BDFS[0], 'x', 202, sysfs_class=fc.CLASS_ROOT, sysfs_pci=fc.PCI_ROOT, ops=rig.kmd)
        with self.assertRaises(cl.CardLinkUnavailable):
            link.open(rig.staging_info())

    def test_the_default_wait_after_the_power_state_is_the_one_the_qwen_link_uses(self):
        self.assertEqual(inspect.signature(cl.CardLink.__init__).parameters['settle_s'].default, 1.0)
        rig = self.rig()
        board = rig.boards[0]
        link = cl.CardLink(board.chardev, board.bdf, board.allocate_path, rig.window_id, shard_object_bytes=rig.shard_object_bytes, slot_objects=rig.slot_objects,
                           sysfs_class=fc.CLASS_ROOT, sysfs_pci=fc.PCI_ROOT, ops=rig.kmd)
        rig.links.append(link)
        with mock.patch.object(cl.time, 'sleep') as sleep:
            link.open(rig.staging_info())
        sleep.assert_called_once_with(1.0)
        self.assertEqual([name for name, shard in rig.kmd.log if shard == 0][:2], ['power', 'allocate'], 'the wait follows SET_POWER_STATE and precedes the window')

    def test_constructor_arguments_are_checked(self):
        with self.assertRaises(ValueError):
            cl.CardLink('/dev/tenstorrent/1', fc.BDFS[0], 'x', 202, shard_object_bytes=1000)
        with self.assertRaises(ValueError):
            cl.CardLink('/dev/tenstorrent/1', fc.BDFS[0], 'x', 202, ordering='fast')
        with self.assertRaises(ValueError):
            cl.CardLink('/dev/tenstorrent/1', '../etc', 'x', 202)
        self.assertEqual(cl.normalise_bdf('04:00.0'), '0000:04:00.0')
        self.assertEqual(cl.normalise_bdf('0000:0F:00.1'), '0000:0f:00.1')

    def test_the_chardev_can_be_found_from_the_pci_address(self):
        rig = self.rig()
        self.assertEqual(cl.chardev_of_bdf(fc.BDFS[2], sysfs_class=fc.CLASS_ROOT, ops=rig.kmd), '/dev/tenstorrent/%d' % rig.boards[2].index)
        with self.assertRaises(cl.CardLinkUnavailable):
            cl.chardev_of_bdf('0000:0f:00.0', sysfs_class=fc.CLASS_ROOT, ops=rig.kmd)


class AimTests(RigCase):
    def setUp(self):
        self.r = self.rig()
        self.links = self.r.open_links()

    def test_an_aim_configures_then_proves_with_a_canary_read(self):
        link, board = self.links[1], self.r.boards[1]
        before = len(board.configures)
        epoch = link.aim_epoch
        link.aim(3)
        self.assertEqual(link.aimed_bank, 3)
        self.assertEqual(board.aimed, 3)
        self.assertEqual(link.aim_epoch, epoch + 1)
        self.assertEqual(len(board.configures), before + 1)
        link.aim(3)
        self.assertEqual(link.aim_epoch, epoch + 2, 'aiming at the same bank again is a new epoch, and is verified again')

    def test_an_unknown_bank_is_a_bounds_violation(self):
        for bank in (8, -1, 99, None, '1'):
            with self.subTest(bank=bank):
                with self.assertRaises(cl.BoundsViolation):
                    self.links[0].aim(bank)
        self.assertIsNone(self.links[0].poisoned, 'a bad argument is not a window fault')

    def test_a_window_that_ignores_the_aim_is_found_by_the_canary_and_the_link_is_unusable(self):
        link, board = self.links[2], self.r.boards[2]
        link.aim(1)
        board.ignore_configure = True
        with self.assertRaises(cl.CanaryMismatch):
            link.aim(5)
        self.assertIsNone(link.aimed_bank)
        self.assertIn('canary', link.poisoned)
        self.assertEqual(link.stats['canary_mismatches'], 1)
        for call in (lambda: link.aim(1), lambda: link.slot_view(1, 0, 1), lambda: link.begin_io(1, link.aim_epoch)):
            with self.assertRaises(cl.LinkClosed):
                call()

    def test_a_bank_that_lands_on_another_bank_is_found_by_the_canary(self):
        link, board = self.links[0], self.r.boards[0]
        board.alias = {6: 4}
        with self.assertRaises(cl.CanaryMismatch):
            link.aim(6)

    def test_a_re_aim_with_a_bio_in_flight_is_refused_and_does_not_move_the_window(self):
        link, board = self.links[0], self.r.boards[0]
        link.aim(2)
        link.begin_io(2, link.aim_epoch)
        configures = len(board.configures)
        with self.assertRaises(cl.AimWhileBusy):
            link.aim(3)
        self.assertEqual(len(board.configures), configures, 'refused before CONFIGURE_TLB')
        self.assertEqual(link.aimed_bank, 2)
        link.end_io()
        link.aim(3)
        self.assertEqual(link.aimed_bank, 3)

    def test_an_io_planned_for_another_bank_or_an_older_epoch_is_stale(self):
        link = self.links[0]
        link.aim(2)
        epoch = link.aim_epoch
        with self.assertRaises(cl.StaleAim):
            link.begin_io(3, epoch)
        with self.assertRaises(cl.StaleAim):
            link.begin_io(2, epoch - 1)
        link.aim(2)
        with self.assertRaises(cl.StaleAim):
            link.begin_io(2, epoch)             # same bank, but aimed again since: a new epoch
        self.assertEqual(link.inflight, 0, 'a refused begin_io counts nothing')

    def test_end_io_without_begin_io_is_an_error(self):
        with self.assertRaises(cl.CardLinkError):
            self.links[0].end_io()

    def test_scratch_access_needs_the_aimed_bank_and_no_bio_in_flight(self):
        link = self.links[0]
        link.aim(0)
        link.scratch_write(0, b'abcdefgh')
        self.assertEqual(link.scratch_read(0, 8), b'abcdefgh')
        with self.assertRaises(cl.StaleAim):
            link.scratch_read(1, 8)
        for size in (0, 4, 12, PAGE + 8):
            with self.assertRaises(cl.BoundsViolation):
                link.scratch_read(0, size)
        with self.assertRaises(cl.BoundsViolation):
            link.scratch_write(0, b'abcd')
        link.begin_io(0, link.aim_epoch)
        with self.assertRaises(cl.AimWhileBusy):
            link.scratch_write(0, b'x' * 8)
        link.end_io()


class WordAccessTests(RigCase):
    """Every CPU access to the uncached window is one aligned 8-byte load or store (never a memcpy of the range)."""

    def test_open_touches_the_window_only_in_whole_words(self):
        rig = self.rig()
        rig.open_links()
        words_per_page = PAGE // 8
        for board in rig.boards:
            self.assertEqual(board.word_stores, 8 * words_per_page, 'a canary page per bank, written word by word')
            self.assertEqual(board.word_loads, 8 * (1 + words_per_page), 'per bank one 8-byte read before the write and a whole-page read to prove it')

    def test_aim_and_flush_each_read_one_word_and_a_scratch_page_is_written_word_by_word(self):
        rig = self.rig()
        link, board = rig.open_links()[0], rig.boards[0]
        loads, stores = board.word_loads, board.word_stores
        link.aim(3)
        self.assertEqual((board.word_loads - loads, board.word_stores - stores), (1, 0))
        link.flush()
        self.assertEqual(board.word_loads - loads, 2)
        link.scratch_write(3, bytes(range(256)) * (PAGE // 256))
        self.assertEqual(board.word_stores - stores, PAGE // 8)
        self.assertEqual(link.scratch_read(3, PAGE), bytes(range(256)) * (PAGE // 256))

    def test_what_is_not_whole_aligned_words_is_refused_before_the_card_is_touched(self):
        rig = self.rig()
        link, board = rig.open_links()[0], rig.boards[0]
        loads, stores = board.word_loads, board.word_stores
        base = rig.bank_window(0) + rig.lead
        for offset, size in ((base + 1, 8), (base + 4, 8), (base, 7), (base, 12), (base, 0), (-8, 8)):
            with self.subTest(offset=offset, size=size):
                with self.assertRaises(cl.BoundsViolation):
                    link._cpu_read(offset, size)
                with self.assertRaises(cl.BoundsViolation):
                    link._cpu_write(offset, bytes(size))
        self.assertEqual((board.word_loads, board.word_stores), (loads, stores))

    def test_the_system_ops_view_is_a_64_bit_word_view_of_the_mapping(self):
        mapping = mmap.mmap(-1, 4 * PAGE)
        words = cl.SystemOps().words(mapping)
        words[3] = 0x0102030405060708
        self.assertEqual(bytes(mapping[24:32]), (0x0102030405060708).to_bytes(8, sys.byteorder))
        self.assertEqual(words[3], 0x0102030405060708)
        with self.assertRaises(BufferError):
            mapping.close()                 # the view exports the mapping: it is released first
        words.release()
        mapping.close()

    def test_the_link_reads_and_writes_a_real_mapping_through_the_system_ops_view(self):
        """The same accessors the card path uses, on a real buffer."""
        link = cl.CardLink('/dev/tenstorrent/1', fc.BDFS[0], 'x')
        link._cpu = mmap.mmap(-1, 4 * PAGE)
        link._words = link.ops.words(link._cpu)
        try:
            data = bytes(range(256)) * (PAGE // 256)
            link._cpu_write(PAGE, data)
            self.assertEqual(bytes(link._cpu[PAGE:2 * PAGE]), data)
            self.assertEqual(link._cpu_read(PAGE, PAGE), data)
            self.assertEqual(link._cpu_read(PAGE + 8, 16), data[8:24])
            self.assertEqual(bytes(link._cpu[:PAGE]), bytes(PAGE), 'nothing else was written')
        finally:
            link._words.release()
            link._cpu.close()


class FlushTests(RigCase):
    def test_flush_reads_the_canary_and_never_raises(self):
        rig = self.rig()
        links = rig.open_links()
        link = links[0]
        link.flush()
        self.assertEqual((link.stats['flushes'], link.stats['flush_mismatches']), (1, 0))
        rig.boards[0].poke(7, rig.canary_offset(7), b'\xff' * 8)            # the canary page was overwritten: advisory, but counted
        link.flush()
        self.assertEqual(link.stats['flush_mismatches'], 1)
        link.close()
        link.flush()                                                         # closed: a no-op


class SlotViewTests(RigCase):
    def setUp(self):
        self.r = self.rig()
        self.links = self.r.open_links()

    def test_one_view_per_object_each_a_shard_long_and_in_order(self):
        link = self.links[1]
        views = link.slot_view(3, 0, fc.SMALL_SLOT)
        self.assertEqual(len(views), fc.SMALL_SLOT)
        for j, view in enumerate(views):
            self.assertEqual(view.nbytes, fc.SMALL_SHARD)
            self.assertEqual(link.locate(view), self.r.data_offset(3, j))
        self.assertEqual(link.locate(views[1]) - link.locate(views[0]), fc.SMALL_SHARD, 'objects follow each other')
        for view in views:
            view.release()

    def test_a_view_is_the_banks_bytes_when_the_window_is_aimed_there(self):
        link, board = self.links[2], self.r.boards[2]
        link.aim(4)
        board.poke(4, self.r.data_offset(4, 2), b'hello' * 10)
        view = link.slot_view(4, 2, 1)[0]
        self.assertEqual(bytes(view[:50]), b'hello' * 10)
        view.release()

    def test_in_lockstep_two_banks_have_the_same_addresses_so_the_aim_is_what_tells_them_apart(self):
        link = self.links[0]
        a, b = link.slot_view(0, 0, 1)[0], link.slot_view(1, 0, 1)[0]
        self.assertEqual(link.locate(a), self.r.data_offset(0, 0))
        self.assertEqual(link.locate(b), self.r.data_offset(1, 0))
        self.assertEqual(link.locate(a), link.locate(b), 'one mapping: the bank is whichever the window is aimed at (hence the epoch check)')
        a.release()
        b.release()

    def test_banks_with_different_addresses_have_different_views(self):
        rig = self.rig(per_bank_offset=0x10000)
        link = rig.open_links()[0]
        a, b = link.slot_view(0, 0, 1)[0], link.slot_view(1, 0, 1)[0]
        self.assertEqual(link.locate(b) - link.locate(a), 0x10000)
        a.release()
        b.release()

    def test_bounds(self):
        link = self.links[0]
        last = fc.SMALL_SLOT
        cases = [(0, 0, 0), (0, -1, 1), (0, 0, last + 1), (0, last, 1), (0, last - 1, 2), (0, 5, 0), (0, 5, -1), (8, 0, 1), (-1, 0, 1), (99, 0, 1),
                 (0, 0.0, 1), (0, 0, 1.0), (0, True, 1), (0, None, 1), ('0', 0, 1)]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(cl.BoundsViolation):
                    link.slot_view(*args)
        for good in ((0, 0, 1), (0, last - 1, 1), (0, 0, last), (7, 1, last - 1)):
            with self.subTest(args=good):
                views = link.slot_view(*good)
                self.assertEqual(len(views), good[2])
                for view in views:
                    view.release()

    def test_the_canary_page_is_not_reachable_through_a_view(self):
        link = self.links[0]
        views = link.slot_view(0, 0, fc.SMALL_SLOT)
        low = min(link.locate(v) for v in views)
        self.assertEqual(low, self.r.canary_offset(0) + PAGE)
        for view in views:
            view.release()

    def test_expected_slice_offset_is_bounds_checked(self):
        link = self.links[0]
        self.assertEqual(link.expected_slice_offset(2, 3), self.r.data_offset(2, 3))
        for bank, obj in ((2, -1), (2, fc.SMALL_SLOT), (2, 1.5), (9, 0)):
            with self.assertRaises(cl.BoundsViolation):
                link.expected_slice_offset(bank, obj)

    def test_locate_says_where_a_buffer_really_is(self):
        link, other = self.links[0], self.links[1]
        view = link.slot_view(2, 1, 1)[0]
        foreign = other.slot_view(2, 1, 1)[0]
        self.assertEqual(link.locate(view), self.r.data_offset(2, 1))
        self.assertIsNone(link.locate(foreign), 'another card\'s mapping')
        self.assertEqual(link.locate(view[1:]), self.r.data_offset(2, 1) + 1)
        self.assertIsNone(link.locate(bytearray(100)), 'host memory')
        self.assertIsNone(link.locate(memoryview(b'abc')), 'read-only memory')
        self.assertIsNone(link.locate(object()))
        view.release()
        foreign.release()


class CloseTests(RigCase):
    def test_close_releases_everything_and_is_idempotent(self):
        rig = self.rig()
        links = rig.open_links()
        for link in links:
            link.close()
            link.close()
            self.assertTrue(link.closed)
            self.assertEqual(link.stats['unmap_failed'], 0)
            self.assertTrue(link._slice_map is None and link._cpu is None and link._words is None)
        self.assertEqual(rig.kmd.open_fds, 0)
        for board in rig.boards:
            self.assertEqual(board.taken, set(), 'FREE_TLB')
            self.assertFalse(board.slice_mapped, 'the mapping was unmapped: the lease is over')
            self.assertEqual(board.available, rig.region_size)
        with self.assertRaises(cl.LinkClosed):
            links[0].aim(0)

    def test_a_view_that_is_still_alive_stops_the_unmap_and_close_says_so(self):
        """The kernel ends the p2pmem lease at munmap, not when the descriptor closes. A view of the mapping that outlives close keeps it mapped: that is logged and
        counted, and the lease ends when the view goes."""
        rig = self.rig()
        links = rig.open_links()
        views = [link.slot_view(0, 0, 2) for link in links]
        mapping = links[0]._slice_map
        for link in links:
            with self.assertLogs('ttkvp2p.cardlink', level='WARNING') as logged:
                link.close()
            self.assertTrue(any('could not be unmapped' in line for line in logged.output), logged.output)
            self.assertEqual(link.stats['unmap_failed'], 1)
            self.assertTrue(link.closed)
        self.assertFalse(mapping.closed, 'the mapping is still mapped')
        self.assertEqual(rig.kmd.open_fds, 0, 'the descriptors are closed all the same')
        for board in rig.boards:
            self.assertTrue(board.slice_mapped, 'the lease is held while a view of the mapping exists')
            self.assertEqual(board.available, 0)
        del views, mapping
        gc.collect()
        for board in rig.boards:
            self.assertFalse(board.slice_mapped, 'the last view went: the mapping was unmapped and the lease is over')
            self.assertEqual(board.available, rig.region_size)

    def test_a_released_view_does_not_stop_the_unmap(self):
        rig = self.rig()
        link = rig.open_links()[0]
        for view in link.slot_view(0, 0, 2):
            view.release()
        link.close()
        self.assertEqual(link.stats['unmap_failed'], 0)
        self.assertFalse(rig.boards[0].slice_mapped)

    def test_an_abandoned_link_releases_nothing(self):
        rig = self.rig()
        link = rig.open_links()[0]
        fds = rig.kmd.open_fds
        link.abandon()
        with self.assertLogs('ttkvp2p.cardlink', level='WARNING'):
            link.close()
        self.assertTrue(link.closed)
        self.assertEqual(rig.kmd.open_fds, fds, 'a bio may still be in flight: nothing is unmapped or freed')
        self.assertEqual(rig.boards[0].pinned, {202})

    def test_open_twice_is_refused(self):
        rig = self.rig()
        link = rig.open_links()[0]
        with self.assertRaises(cl.CardLinkError):
            link.open(rig.staging_info())


class DifferentBankAddressesTests(RigCase):
    def test_banks_whose_ranges_start_at_different_addresses(self):
        rig = self.rig(per_bank_offset=0x10000)
        links = rig.open_links()
        for bank in range(8):
            self.assertEqual(links[0].expected_slice_offset(bank, 0), rig.data_offset(bank, 0))
        links[0].aim(5)


class GemmaGeometryTests(RigCase):
    def test_the_real_shard_size(self):
        self.assertEqual(cl.GEMMA_SHARD_OBJECT_BYTES, 696320)
        self.assertEqual(cl.GEMMA_SHARD_OBJECT_BYTES, 10 * 69632)
        self.assertEqual(cl.GEMMA_SHARD_OBJECT_BYTES % PAGE, 0)
        self.assertEqual(4 * cl.GEMMA_SHARD_OBJECT_BYTES, 2785280)
        # S = 64 MiB per bank leaves room for 96 objects after the canary page (design 2.2)
        self.assertEqual((64 * 2 ** 20 - PAGE - PAGE) // cl.GEMMA_SHARD_OBJECT_BYTES, 96)

    def test_open_with_the_real_geometry(self):
        rig = self.rig(shard_object_bytes=fc.GEMMA_SHARD, slot_objects=3)
        links = rig.open_links()
        self.assertEqual(links[0].shard_object_bytes, 696320)
        self.assertEqual(links[0].expected_slice_offset(2, 2) - links[0].expected_slice_offset(2, 0), 2 * 696320)

    def test_a_default_link_has_the_gemma_shard_and_at_most_128_objects(self):
        link = cl.CardLink('/dev/tenstorrent/1', fc.BDFS[0], 'x')
        self.assertEqual(link.shard_object_bytes, 696320)
        self.assertEqual(link.window_id, 202)


class InterfaceTests(unittest.TestCase):
    """INTERFACE.md 3.2, as written."""

    def test_signatures(self):
        sig = inspect.signature
        self.assertEqual(list(sig(cl.CardLink.__init__).parameters)[:5], ['self', 'chardev', 'bdf', 'allocate_path', 'window_id'])
        self.assertEqual(sig(cl.CardLink.__init__).parameters['window_id'].default, 202)
        self.assertEqual(list(sig(cl.CardLink.open).parameters), ['self', 'staging_info'])
        self.assertEqual(list(sig(cl.CardLink.aim).parameters), ['self', 'bank'])
        self.assertEqual(list(sig(cl.CardLink.slot_view).parameters), ['self', 'bank', 'first_obj', 'n_obj'])
        self.assertEqual(list(sig(cl.CardLink.flush).parameters), ['self'])
        self.assertEqual(list(sig(cl.CardLink.close).parameters), ['self'])
        # everything beyond the first four constructor arguments is keyword-only: a call as the interface writes it keeps working
        for name, parameter in list(sig(cl.CardLink.__init__).parameters.items())[5:]:
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY, name)

    def test_the_abi_equals_the_qwen_module_that_ran_on_a_card(self):
        path = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'scripts', 'ci', 'qwen_p2p_restore.py')
        if not os.path.exists(path):
            self.skipTest('scripts/ci/qwen_p2p_restore.py is not in this tree')
        saved = sys.dont_write_bytecode
        sys.dont_write_bytecode = True                  # (no __pycache__ in scripts/ci from a test of another directory)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location('qwen_p2p_restore_for_abi', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except ImportError as error:
            self.skipTest('the Qwen module needs %s' % error)
        finally:
            sys.dont_write_bytecode = saved
        for name in ('TT_MAGIC', 'IOCTL_ALLOCATE_TLB', 'IOCTL_FREE_TLB', 'IOCTL_CONFIGURE_TLB', 'IOCTL_SET_POWER_STATE', 'IOCTL_EXPORT_TLB_DMABUF', 'BUSY_FLAGS',
                     'BUSY_VALIDITY', 'WINDOW_4G', 'FIRST_4G_TLB'):
            self.assertEqual(getattr(cl, name), getattr(module, name), name)
        for name in ('ALLOCATE_IN', 'ALLOCATE_OUT', 'CONFIGURE_IN', 'POWER_STATE', 'EXPORT_ARG'):
            self.assertEqual(getattr(cl, name).format, getattr(module, name).format, name)
            self.assertEqual(getattr(cl, name).size, getattr(module, name).size, name)


class HygieneTests(unittest.TestCase):
    """The repo is public: nothing from the rig goes into these files."""

    FILES = ('README.md', 'ttkvp2p/cardlink.py', 'ttkvp2p/p2pio.py', 'tests/fake_card.py', 'tests/test_cardlink.py', 'tests/test_p2pio.py', 'tests/test_p2pio_mutants.py',
             'tests/test_wp4_wiring.py')

    def test_no_rig_identifiers(self):
        # (each is split so that this file does not match its own patterns)
        patterns = [r'/dev/disk/' + 'by-id', r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', r'zo' + r't\.', r'sha256:[0-9a-f]{16}', r'\.loc' + r'al\b', r'/ho' + 'me/',
                    r'(?i)sams' + r'ung', r'(?i)\bnv' + r'me\d', r'ss' + r'h ', r'(?i)pass' + r'word', r'(?i)tok' + r'en\s*=']
        bdf = re.compile(r'(?<![0-9a-fA-F:])(?:[0-9a-f]{4}:)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])(?![0-9a-fA-F])')
        for name in self.FILES:
            path = os.path.join(PACKAGE_ROOT, name)
            self.assertTrue(os.path.exists(path), '%s is a file of this work package and must exist' % name)
            with open(path) as handle:
                text = handle.read()
            for pattern in patterns:
                self.assertIsNone(re.search(pattern, text), '%s matches %s' % (name, pattern))
            for found in bdf.finditer(text):
                # only the synthetic cards of fake_card.BDFS (buses 01-04) and the short example in a docstring test
                self.assertIn(found.group(1), ('00', '01', '02', '03', '04', '0f'), '%s has a PCI address that is not synthetic: %s' % (name, found.group(0)))


if __name__ == '__main__':
    unittest.main()
