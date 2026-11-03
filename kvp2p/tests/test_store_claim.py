"""The drive store (superblocks, geometry, the exclusive open, discard) and the claim tool.

The tier NEVER formats: it refuses an absent or unknown superblock. The claim tool is the only writer, and it refuses to touch a drive that
carries anything it does not recognise as its own, unless the operator names the format seen.
"""

import io
import os
import struct
import subprocess
import sys
import unittest
from contextlib import redirect_stderr

import tier_kit
from tier_kit import BY_ID, OBJ, TierCase

from ttkvp2p import claim, health, store
from ttkvp2p.spec import LAYOUTS


def sb(**kw):
    values = dict(generation=1, uuid=bytes(range(16)), layout_id='gemma-chipmajor-v1', object_bytes=2785280, num_slots=1000,
                  data_off=1 << 20, device_bytes=(1 << 20) + 1000 * 2785280, claim=bytes(32), mode='process', created=1700000000)
    values.update(kw)
    return store.Superblock(**values)


class SuperblockFormat(unittest.TestCase):
    def test_round_trip(self):
        block = store.encode_superblock(sb())
        self.assertEqual(len(block), 4096)
        self.assertEqual(block[:8], b'TTKVGEM\x00')
        self.assertEqual(store.decode_superblock(block), sb())

    def test_the_magic_is_not_qwens_and_the_version_is_2(self):
        block = store.encode_superblock(sb())
        self.assertNotEqual(block[:8], store.QWEN_MAGIC)
        self.assertEqual(struct.unpack_from('<I', block, 8)[0], 2)

    def test_a_flipped_bit_anywhere_is_refused(self):
        block = bytearray(store.encode_superblock(sb()))
        for offset in (0, 7, 8, 16, 40, 100, 150):
            broken = bytearray(block)
            broken[offset] ^= 0x01
            self.assertIsNone(store.decode_superblock(bytes(broken)), 'a flip at byte %d decoded' % offset)

    def test_another_version_is_refused(self):
        block = bytearray(store.encode_superblock(sb()))
        struct.pack_into('<I', block, 8, 1)               # Qwen's v1 number: even with a fixed-up checksum the version gates it
        self.assertIsNone(store.decode_superblock(bytes(block)))

    def test_garbage_and_zeros_are_not_superblocks(self):
        self.assertIsNone(store.decode_superblock(bytes(4096)))
        self.assertIsNone(store.decode_superblock(os.urandom(4096)))
        self.assertIsNone(store.decode_superblock(b'short'))

    def test_the_claim_covers_the_name_and_the_size(self):
        base = store.claim_digest('drive-a', 1000204886016)
        self.assertEqual(len(base), 32)
        self.assertNotEqual(base, store.claim_digest('drive-b', 1000204886016))
        self.assertNotEqual(base, store.claim_digest('drive-a', 1000204886017))
        self.assertEqual(base, store.claim_digest('drive-a', 1000204886016))


class Geometry(unittest.TestCase):
    OBJECT = 2785280

    def test_a_real_drive_gets_a_mib_aligned_data_region_and_whole_slots(self):
        size = 1_000_204_886_016                       # a nominal 1 TB namespace
        geometry = store.plan_geometry(size, self.OBJECT)
        self.assertEqual(geometry.data_off, 1 << 20)
        self.assertEqual(geometry.num_slots, (size - (1 << 20)) // self.OBJECT)
        self.assertLessEqual(geometry.data_off + geometry.num_slots * self.OBJECT, size)
        self.assertGreater(geometry.data_off + (geometry.num_slots + 1) * self.OBJECT, size, 'only a partial slot is left over')
        self.assertEqual(geometry.canary_off, 8192)
        self.assertEqual(geometry.canary_bytes, 8192)

    def test_every_slot_is_4k_aligned_so_it_can_be_read_with_o_direct(self):
        geometry = store.plan_geometry(50 << 30, self.OBJECT)
        for slot in (0, 1, 777, geometry.num_slots - 1):
            self.assertEqual((geometry.data_off + slot * geometry.object_bytes) % 4096, 0)

    def test_the_header_holds_two_superblocks_and_the_dma_canary(self):
        self.assertEqual(store.SB_SLOT_A, 0)
        self.assertEqual(store.SB_SLOT_B, 4096)
        self.assertEqual(store.CANARY_OFF, 8192)
        self.assertEqual(store.HEADER_BYTES, 16384)
        for size in (store.BIG_DEVICE - 4096, store.BIG_DEVICE, 10 << 30):
            self.assertGreaterEqual(store.plan_geometry(size, 4096).data_off, store.HEADER_BYTES)

    def test_a_small_file_keeps_most_of_itself_for_data(self):
        geometry = store.plan_geometry(1 << 20, 8192)
        self.assertEqual(geometry.data_off, 16384)

    def test_an_unaligned_object_size_is_refused_for_a_drive(self):
        with self.assertRaises(ValueError):
            store.plan_geometry(10 << 30, 5120)
        self.assertEqual(store.plan_geometry(1 << 20, 5120, object_align=store.SECTOR).object_bytes, 5120)

    def test_a_device_too_small_for_one_object_is_capacity_short(self):
        with self.assertRaises(store.CapacityShort):
            store.plan_geometry(20000, 8192)

    def test_the_claim_tools_layout_table_agrees_with_the_tiers(self):
        for name, object_bytes in store.KNOWN_LAYOUTS.items():
            if name in LAYOUTS:
                self.assertEqual(LAYOUTS[name].object_bytes, object_bytes, name)
        self.assertEqual(store.KNOWN_LAYOUTS['gemma-chipmajor-v1'], LAYOUTS['gemma-chipmajor-v1'].object_bytes)
        self.assertEqual(LAYOUTS['gemma-chipmajor-v1'].object_bytes, 2785280)


class OpenAndVerify(TierCase):
    def open_store(self, **kw):
        args = dict(layout_id=self.tree.layout_id, object_bytes=OBJ, persistence='process', allow_regular=True)
        args.update(kw)
        return store.open_store(self.tree.drive, **args)

    def test_a_claimed_drive_opens_and_names_its_slots(self):
        self.tier(num_slots=32)
        opened = self.open_store(expect_claim=store.claim_digest(BY_ID, self.tree.device_bytes))
        self.addCleanup(opened.close)
        self.assertEqual(opened.num_slots, 32)
        self.assertEqual(opened.object_bytes, OBJ)
        self.assertEqual(opened.slot_offset(0), self.tree.data_off)
        self.assertEqual(opened.slot_offset(31), self.tree.data_off + 31 * OBJ)
        with self.assertRaises(IndexError):
            opened.slot_offset(32)

    def test_a_blank_drive_has_no_superblock_and_the_tier_does_not_format_it(self):
        self.tier(claimed=False)
        before = open(self.tree.drive, 'rb').read(1 << 16)
        with self.assertRaises(store.UnknownSuperblock) as raised:
            self.open_store()
        self.assertEqual(raised.exception.code, 'superblock_unknown')
        self.assertEqual(open(self.tree.drive, 'rb').read(1 << 16), before, 'the tier wrote to a drive it does not own')

    def test_another_formats_superblock_is_unknown(self):
        self.tier(claimed=False)
        with open(self.tree.drive, 'r+b') as handle:
            handle.write(store.QWEN_MAGIC + bytes(100))
        with self.assertRaises(store.UnknownSuperblock):
            self.open_store()

    def test_random_data_is_unknown_the_p1_pattern_included(self):
        self.tier(claimed=False)
        with open(self.tree.drive, 'r+b') as handle:
            handle.write(os.urandom(self.tree.device_bytes))
        with self.assertRaises(store.UnknownSuperblock):
            self.open_store()

    def test_a_layout_a_size_or_a_mode_that_is_not_this_engines_is_a_layout_mismatch(self):
        self.tier()
        for kw, what in (({'layout_id': 'some-other-layout-v9'}, 'layout'), ({'object_bytes': OBJ * 2}, 'size'), ({'persistence': 'durable'}, 'mode')):
            with self.assertRaises(store.LayoutMismatch, msg=what) as raised:
                self.open_store(**kw)
            self.assertEqual(raised.exception.code, 'layout_mismatch')

    def test_a_drive_that_shrank_is_capacity_short(self):
        self.tier()
        with open(self.tree.drive, 'r+b') as handle:
            handle.truncate(self.tree.data_off + 10 * OBJ)
        with self.assertRaises(store.CapacityShort):
            self.open_store()

    def test_a_drive_that_reads_zero_bytes_is_gone_not_blank(self):
        self.tier()
        with open(self.tree.drive, 'r+b') as handle:
            handle.truncate(0)
        with self.assertRaises(store.DeviceGone) as raised:
            self.open_store()
        self.assertEqual(raised.exception.code, 'device_dead')

    def test_a_resized_namespace_is_not_the_claimed_drive(self):
        self.tier()
        with open(self.tree.drive, 'r+b') as handle:
            handle.truncate(self.tree.device_bytes + 4096 * 10)
        with self.assertRaises(store.UnknownSuperblock):
            self.open_store()

    def test_a_cloned_drive_fails_the_claim(self):
        self.tier()
        with self.assertRaises(store.UnknownSuperblock):
            self.open_store(expect_claim=store.claim_digest('some-other-drive', self.tree.device_bytes))

    def test_a_regular_file_is_refused_unless_the_test_flag_allows_it(self):
        self.tier()
        with self.assertRaises(store.DeviceAbsent) as raised:
            self.open_store(allow_regular=False)
        self.assertEqual(raised.exception.code, 'device_absent')

    def test_a_missing_path_is_device_absent(self):
        self.tier()
        with self.assertRaises(store.DeviceAbsent):
            store.open_store(os.path.join(self.root, 'no-such-drive'), layout_id='x', object_bytes=OBJ, allow_regular=True)

    def test_a_second_open_is_claim_busy(self):
        self.tier()
        first = self.open_store()
        self.addCleanup(first.close)
        with self.assertRaises(store.DeviceBusy) as raised:
            self.open_store()
        self.assertEqual(raised.exception.code, 'claim_busy')

    def test_one_torn_superblock_slot_is_survived_by_the_other(self):
        self.tier()
        with open(self.tree.drive, 'r+b') as handle:
            handle.seek(5)
            handle.write(b'\xff\xff\xff')                     # slot A torn
        opened = self.open_store()
        self.addCleanup(opened.close)
        self.assertEqual(opened.superblock.generation, 2)     # slot B, written second, carries the higher generation

    def test_the_higher_generation_wins(self):
        self.tier()
        drive = store.DriveFile(self.tree.drive, allow_regular=True)
        self.addCleanup(drive.close)
        found = store.read_superblocks(drive)
        self.assertEqual([s.generation for s in found], [2, 1])

    def test_opening_never_writes_a_superblock(self):
        self.tier()
        before = open(self.tree.drive, 'rb').read(store.HEADER_BYTES)
        opened = self.open_store()
        opened.close()
        self.assertEqual(open(self.tree.drive, 'rb').read(store.HEADER_BYTES), before)


class Discard(TierCase):
    def test_discard_deallocates_a_range_of_a_file_drive(self):
        self.tier()
        drive = store.DriveFile(self.tree.drive, allow_regular=True)
        self.addCleanup(drive.close)
        with open(self.tree.drive, 'r+b') as handle:
            handle.seek(self.tree.data_off)
            handle.write(b'\x5a' * (OBJ * 4))
        drive.discard(self.tree.data_off, OBJ * 2)
        with open(self.tree.drive, 'rb') as handle:
            handle.seek(self.tree.data_off)
            data = handle.read(OBJ * 4)
        self.assertEqual(data[:OBJ * 2], bytes(OBJ * 2))
        self.assertEqual(data[OBJ * 2:], b'\x5a' * (OBJ * 2))

    def test_a_discard_outside_the_drive_is_refused(self):
        self.tier()
        drive = store.DriveFile(self.tree.drive, allow_regular=True)
        self.addCleanup(drive.close)
        with self.assertRaises(ValueError):
            drive.discard(self.tree.device_bytes - OBJ, OBJ * 3)

    def test_runs_split_a_region_into_bounded_pieces(self):
        self.assertEqual(store.region_runs(0, 10, 4), [(0, 4), (4, 4), (8, 2)])
        self.assertEqual(store.region_runs(5, 0, 4), [])


class ClaimTool(TierCase):
    def setUp(self):
        super().setUp()
        self.tier(claimed=False)

    def claim(self, **kw):
        return self.tree.claim(self.clock, **kw)

    def test_a_blank_drive_is_claimed_and_then_verifies(self):
        report = self.claim()
        self.assertEqual(report['format_seen'], 'blank')
        self.assertTrue(report['written'])
        self.assertEqual(report['num_slots'], self.tree.num_slots)
        opened = store.open_store(self.tree.drive, layout_id=self.tree.layout_id, object_bytes=OBJ, allow_regular=True,
                                  expect_claim=store.claim_digest(BY_ID, self.tree.device_bytes))
        opened.close()

    def test_a_second_claim_of_the_same_drive_changes_nothing(self):
        self.claim()
        before = open(self.tree.drive, 'rb').read(store.HEADER_BYTES)
        report = self.claim()
        self.assertTrue(report.get('already_claimed'))
        self.assertFalse(report['written'])
        self.assertEqual(open(self.tree.drive, 'rb').read(store.HEADER_BYTES), before)

    def test_a_dry_run_writes_nothing(self):
        before = open(self.tree.drive, 'rb').read()
        report = self.claim(dry_run=True)
        self.assertEqual(report['format_seen'], 'blank')
        self.assertFalse(report['written'])
        self.assertEqual(open(self.tree.drive, 'rb').read(), before)

    def test_the_name_must_be_typed_back(self):
        before = open(self.tree.drive, 'rb').read()
        with self.assertRaises(claim.Refused) as raised:
            self.claim(confirm='not-the-name')
        self.assertIn('typed back', str(raised.exception))
        self.assertEqual(open(self.tree.drive, 'rb').read(), before)

    def test_a_path_that_is_not_a_by_id_symlink_is_refused(self):
        by_id_dir = os.path.join(self.root, 'by-id')
        os.makedirs(by_id_dir)
        link = os.path.join(by_id_dir, BY_ID)
        os.symlink(self.tree.drive, link)
        self.assertEqual(claim.check_by_id.__name__, 'check_by_id')
        saved = claim.BY_ID_DIR
        claim.BY_ID_DIR = by_id_dir
        self.addCleanup(setattr, claim, 'BY_ID_DIR', saved)
        name, path = claim.check_by_id(link, BY_ID, False)
        self.assertEqual((name, path), (BY_ID, os.path.realpath(self.tree.drive)))
        with self.assertRaises(claim.Refused):
            claim.check_by_id(self.tree.drive, os.path.basename(self.tree.drive), False)        # the realpath, not a by-id entry
        with self.assertRaises(claim.Refused):
            claim.check_by_id(link, 'wrong', False)

    def test_the_status_file_gates_the_claim(self):
        before = open(self.tree.drive, 'rb').read()
        cases = [
            ('a stale status file', lambda: self.tree.write_status(self.clock, timestamp=health.iso_utc(self.clock.now - 3600))),
            ('a status file for another drive', lambda: self.tree.write_status(self.clock, by_id='some-other-drive')),
            ('a drive that is not ok', lambda: self.tree.write_status(self.clock, state='not_live')),
            ('a recorded size that differs', lambda: self.tree.write_status(self.clock, size_bytes=self.tree.device_bytes + 4096)),
        ]
        for what, change in cases:
            change()
            with self.assertRaises(claim.Refused, msg=what):
                self.claim()
        self.assertEqual(open(self.tree.drive, 'rb').read(), before)

    def test_without_a_recorded_size_the_operator_types_it(self):
        self.tree.write_status(self.clock, size_bytes=None)
        with self.assertRaises(claim.Refused) as raised:
            self.claim()
        self.assertIn('--size', str(raised.exception))
        with self.assertRaises(claim.Refused):
            self.claim(typed_size=self.tree.device_bytes + 1)
        self.assertTrue(self.claim(typed_size=self.tree.device_bytes)['written'])

    def test_a_drive_another_process_holds_is_refused(self):
        held = store.DriveFile(self.tree.drive, allow_regular=True)
        self.addCleanup(held.close)
        with self.assertRaises(claim.Refused) as raised:
            self.claim()
        self.assertIn('in use', str(raised.exception))

    def test_every_other_format_is_refused_unless_the_operator_names_it(self):
        def put(offset, data):
            with open(self.tree.drive, 'r+b') as handle:
                handle.truncate(0)
                handle.truncate(self.tree.device_bytes)
                handle.seek(offset)
                handle.write(data)
        cases = {
            'qwen-nvme-v1': (0, store.QWEN_MAGIC + bytes(8)),
            'gemma-kv-v2': (4096, store.SB_MAGIC + bytes(8)),
            'gpt': (512, b'EFI PART'),
            'mbr': (510, b'\x55\xaa'),
            'ext': (1080, b'\x53\xef'),
            'xfs': (0, b'XFSB'),
            'ntfs': (3, b'NTFS    '),
            'luks': (0, b'LUKS\xba\xbe'),
            'swap': (4086, b'SWAPSPACE2'),
            'lvm': (512, b'LABELONE'),
            'unknown-data': (100, b'\x01\x02\x03 any bytes'),
        }
        for seen, (offset, data) in cases.items():
            put(offset, data)
            with self.assertRaises(claim.Refused, msg=seen) as raised:
                self.claim()
            self.assertIn('--overwrite=%s' % seen, str(raised.exception))
            with self.assertRaises(claim.Refused, msg='wrong name for ' + seen):
                self.claim(overwrite='blank' if seen != 'unknown-data' else 'gpt')
            self.assertTrue(self.claim(overwrite=seen)['written'], seen)

    def test_the_p1_random_pattern_is_unknown_data_and_needs_its_name(self):
        with open(self.tree.drive, 'r+b') as handle:
            handle.write(os.urandom(1 << 16))
        with self.assertRaises(claim.Refused) as raised:
            self.claim()
        self.assertIn('unknown-data', str(raised.exception))
        self.assertIn('P1', str(raised.exception))
        report = self.claim(overwrite='unknown-data')
        self.assertTrue(report['written'])
        self.assertEqual(report['format_seen'], 'unknown-data')

    def test_an_overwrite_removes_the_old_signatures(self):
        with open(self.tree.drive, 'r+b') as handle:
            handle.seek(512)
            handle.write(b'EFI PART' + bytes(100))
            handle.seek(65536)
            handle.write(b'stale data after the header')
        self.claim(overwrite='gpt')
        head = open(self.tree.drive, 'rb').read(1 << 17)
        self.assertNotIn(b'EFI PART', head)
        self.assertEqual(head[8192:16384], bytes(8192), 'the DMA canary block is zero')

    def test_an_unknown_overwrite_token_is_refused(self):
        with self.assertRaises(claim.Refused):
            self.claim(overwrite='whatever')

    def test_an_explicit_object_size_is_for_regular_files_only(self):
        with self.assertRaises(claim.Refused):
            claim.claim(self.tree.by_id_path, BY_ID, status_path=os.path.join(self.tree.status_dir, 'status.json'), object_bytes=OBJ,
                        allow_regular=False)

    def test_the_command_line_exits_2_on_a_refusal_and_0_on_a_claim(self):
        argv = ['--device', self.tree.by_id_path, '--confirm', 'wrong', '--status', os.path.join(self.tree.status_dir, 'status.json'),
                '--allow-regular', '--layout', self.tree.layout_id]
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(claim.main(argv), 2)
        self.assertIn('REFUSED', err.getvalue())

    def test_the_tool_is_importable_without_vllm(self):
        code = ('import sys; sys.modules["vllm"] = None; sys.modules["torch"] = None; import ttkvp2p.claim, ttkvp2p.store, ttkvp2p.health; print("ok")')
        env = dict(os.environ)
        env['PYTHONPATH'] = tier_kit.fakes.PACKAGE_ROOT
        done = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-1500:])
        self.assertIn('ok', done.stdout)


if __name__ == '__main__':
    unittest.main()
