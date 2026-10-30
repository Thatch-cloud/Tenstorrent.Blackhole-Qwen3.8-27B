"""p2p_staging_card.run() on the CPU: the card check of the P2P ops against an in-memory fake of the extension (numpy), at four chips, so that the check
itself is known to pass a correct implementation and to FAIL each plausible wrong one before a card is spent on it.

Run from this directory:  python3 -B -m unittest test_p2p_staging_card      (numpy and torch; python 3.11 or later in CI)
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import p2p_staging_card as card  # noqa: E402
import p2p_staging_ref as ref  # noqa: E402

DEVICES = 4
SLICE = card.PACKED_HEAD_BLOCK_BYTES
STAGING_MIB = 16


class FakeCache(object):
    def __init__(self, blocks, slice_bytes, devices):
        self.blocks = blocks
        self.data = np.zeros((devices, blocks, slice_bytes), dtype=np.uint8)
        self.shape = (blocks, 1, 64, 256)


class FakeStaging(object):
    def __init__(self, size, bank, page, devices, banks=8):
        self.size_bytes, self.bank, self.page_bytes, self.num_banks = size, bank, page, banks
        self.devices = devices
        self.banks = {}
        self.released = False

    @property
    def data(self):
        """Banks are made when first touched (eight of them would be a gigabyte of zeros)."""
        staging = self

        class Banks(object):
            def __getitem__(self, bank):
                if bank not in staging.banks:
                    staging.banks[bank] = np.zeros((staging.devices, staging.size_bytes), dtype=np.uint8)
                return staging.banks[bank]
        return Banks()

    def release(self):
        self.released = True


class FakeMesh(object):
    def get_num_devices(self):
        return DEVICES


class FakeTTNN(object):
    """The version 3 extension on numpy. The knobs are the mistakes the card check must catch."""
    bfloat8_b, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bfloat8_b', 'TILE', 'DRAM'

    def __init__(self, devices=DEVICES, **knobs):
        self.devices = devices
        self.knobs = knobs
        self.staging = []
        FakeMesh.get_num_devices = lambda mesh: devices

    ReplicateTensorToMesh = staticmethod(lambda mesh: 'replicate')

    def as_tensor(self, tensor, device=None, dtype=None, layout=None, memory_config=None, mapper=None, mesh_mapper=None):
        return FakeCache(tensor.shape[0], SLICE * tensor.shape[1], self.devices)

    def deallocate(self, tensor):
        pass

    def qwen_block_bytes(self, cache):
        return cache.data.shape[2] + self.knobs.get('block_bytes_off', 0)

    def qwen_read_blocks_raw(self, cache, out, blocks):
        out[:] = cache.data[:, list(blocks)]

    def qwen_write_blocks_raw(self, cache, data, blocks):
        cache.data[:, list(blocks)] = data

    def qwen_p2p_staging_alloc(self, mesh, size_bytes=128 << 20, bank=0, page_bytes=65536, bottom_up=False):
        problem = ref.check_staging(size_bytes, page_bytes, bank, 8, 4 << 30)
        if problem:
            raise ValueError(problem)
        staging = FakeStaging(size_bytes, bank, page_bytes, self.devices)
        self.staging.append(staging)
        return staging

    def qwen_p2p_staging_info(self, staging):
        banks = [dict(dram_bank=b, noc_x=17 + b // 4, noc_y=12 + 3 * (b % 4) + 2, noc_endpoints=[[17 + b // 4, 12 + 3 * (b % 4) + p] for p in range(3)])
                 for b in range(8)]
        entries = []
        for d in range(self.devices):
            address = 1048576 + 192 + (7 if self.knobs.get('misaligned') else 0) + (64 if self.knobs.get('address_differs') and d == 2 else 0)
            lead = (4096 - address % 4096) % 4096
            entry = dict(banks[staging.bank], mesh_index=d, device_id=d, pcie_device=d, pci_bus_id='0000:%02x:00.0' % (0x88 + d),
                         address=address, aligned_4k_offset=lead, aligned_4k_address=address + lead,
                         aligned_4k_size=(staging.size_bytes - lead) // 4096 * 4096, size=staging.size_bytes,
                         bank_offset=0, page_bytes=staging.page_bytes, bytes_reserved_per_bank=staging.size_bytes, bytes_reserved=staging.size_bytes * 8,
                         dram_free_per_bank_before=3 << 30, dram_free_per_bank_after=(3 << 30) - staging.size_bytes, banks=banks)
            entries.append(entry)
        if self.knobs.get('short_info'):
            entries = entries[:-1]
        return entries

    def _region(self, staging, offset, length, bank):
        if offset % staging.page_bytes or length % staging.page_bytes or offset + length > staging.size_bytes:
            raise ValueError('byte_offset and the per-device length must be multiples of the staging page size and inside the range')
        return staging.data[staging.bank if bank is None else bank]

    def qwen_p2p_staging_write(self, staging, offset, data, bank=None):
        bank_array = self._region(staging, offset, data.shape[1], bank)
        bank_array[:, offset:offset + data.shape[1]] = data

    def qwen_p2p_staging_read(self, staging, offset, out, bank=None):
        bank_array = self._region(staging, offset, out.shape[1], bank)
        out[:] = bank_array[:, offset:offset + out.shape[1]]

    def qwen_staging_checksum(self, staging, offset, length, bank=None, cores=64, count=None):
        problem = ref.check_checksum(offset, length, staging.size_bytes, cores, 1 if count is None else count)
        if problem:
            raise ValueError(problem)
        use = staging.bank if bank is None or self.knobs.get('ignore_bank') else bank
        if count is None:
            sums = [ref.checksum_numpy(bytes(staging.data[use][d, offset:offset + length])) for d in range(self.devices)]
            if self.knobs.get('bad_checksum'):
                sums[1] ^= 1
            return sums
        sums = [[ref.checksum_numpy(bytes(staging.data[use][d, offset + j * length:offset + (j + 1) * length])) for j in range(count)] for d in range(self.devices)]
        if self.knobs.get('ranges_swapped'):
            sums = [row[::-1] for row in sums]
        if self.knobs.get('k_continues'):
            # the mistake of one running k: every range after the first is summed as if it continued the first (k not restarted)
            sums = [[row[0]] + [ref.checksum_numpy(bytes(staging.data[use][d, offset:offset + (j + 1) * length]))
                                - ref.checksum_numpy(bytes(staging.data[use][d, offset:offset + j * length])) & ref.MASK64 for j in range(1, count)]
                    for d, row in enumerate(sums)]
        if self.knobs.get('count_short'):
            sums = [row[:-1] for row in sums]
        return sums

    def qwen_write_blocks_from_staging(self, cache, staging, byte_offset, blocks, bank=None, row_stride=None):
        blocks = list(blocks)
        slice_bytes = cache.data.shape[2]
        problem = ref.check_scatter(byte_offset, slice_bytes, ref.DRAM_ALIGNMENT * 17, len(blocks), staging.size_bytes, row_stride)
        if problem:
            raise ValueError(problem)
        if len(set(blocks)) != len(blocks):
            raise ValueError('the block ids must be distinct')
        use = staging.bank if bank is None or self.knobs.get('ignore_bank') else bank
        stride = slice_bytes if row_stride is None or self.knobs.get('stride_ignored') else row_stride - (64 if self.knobs.get('stride_short') else 0)
        data = staging.data[use]
        rows = np.stack([data[:, byte_offset + i * stride:byte_offset + i * stride + slice_bytes] for i in range(len(blocks))], axis=1).copy()
        if self.knobs.get('same_data_everywhere'):
            rows[:] = rows[0]
        if self.knobs.get('forget_block'):
            rows = rows[:, 1:]
            blocks = blocks[1:]
        if self.knobs.get('flip'):
            rows[2, -1, 5] ^= 1
        cache.data[:, blocks] = rows
        if self.knobs.get('spill'):
            cache.data[:, 3] = rows[:, 0]


def run(**knobs):
    ttnn = FakeTTNN(**knobs)
    return ttnn, card.run(ttnn, FakeMesh(), DEVICES, blocks=1024, heads_per_chip=1, staging_mib=STAGING_MIB, bank=3)


class CardCheckOnTheFakeTests(unittest.TestCase):
    def test_a_correct_extension_passes_every_check(self):
        ttnn, report = run()
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual(sorted(report['cases']),
                         ['end_of_ring', 'flipped_byte', 'many_ranges', 'one_block', 'other_bank', 'ranged_flipped_byte', 'run', 'scattered', 'shuffled',
                          'strided_chipmajor', 'strided_end_of_ring', 'strided_gaps', 'strided_shuffled', 'strided_two_launches', 'two_launches', 'two_runs_and_one'])
        self.assertTrue(all(case['checksum_equal'] and case['bytes_equal'] and case['devices_differ'] for name, case in report['cases'].items()
                            if name in ('end_of_ring', 'one_block', 'other_bank', 'run', 'scattered', 'shuffled', 'two_launches', 'two_runs_and_one')))
        self.assertEqual(report['device_ops'], 'strided')
        self.assertTrue(report['cases']['many_ranges']['checksum_equal'])
        self.assertGreater(report['cases']['many_ranges']['ranges'], 512, 'more ranges than one launch carries')
        self.assertTrue(report['cases']['ranged_flipped_byte']['only_that_range'])
        self.assertTrue(report['whole_cache_equal'])
        self.assertTrue(report['cases']['other_bank']['banks_differ'])
        self.assertTrue(report['cases']['flipped_byte']['only_device_1_changed'])
        self.assertTrue(ttnn.staging[0].released, 'the staging buffer is released at the end')
        self.assertEqual(report['staging']['bank'], 3)
        self.assertEqual(report['staging']['bytes_reserved_per_card'], STAGING_MIB * 8 << 20)
        self.assertTrue(report['staging']['formula_first_endpoint_in_info'])

    def test_the_strided_cases_stage_chipmajor_chunks_with_a_row_stride_and_count(self):
        ttnn, report = run()
        cases = report['cases']
        shard = card.SHARD_TENSORS * SLICE
        self.assertEqual(shard % 4096, 0, 'a block\'s shard is whole 4096-byte units, as production\'s 32 x 17,408 = 136 units')
        for name in ('strided_chipmajor', 'strided_shuffled', 'strided_end_of_ring'):
            self.assertEqual((cases[name]['row_stride'], cases[name]['shard_bytes']), (shard, shard), name)
            self.assertTrue(cases[name]['checksum_equal'] and cases[name]['single_equal'] and cases[name]['bytes_equal'] and cases[name]['devices_differ'], name)
        self.assertGreater(cases['strided_gaps']['row_stride'], cases['strided_gaps']['shard_bytes'], 'a stride with gaps between the shards')
        self.assertNotIn('checksum_equal', cases['strided_gaps'], 'the gaps are not part of any checksummed range')
        self.assertTrue(cases['strided_gaps']['bytes_equal'])
        end = cases['strided_end_of_ring']
        self.assertEqual(end['offset'] + end['object_bytes'], STAGING_MIB << 20, 'the last shard ends on the last byte of the ring')
        self.assertEqual(report['cases']['strided_two_launches']['blocks'], card.LAUNCH_BLOCKS + 70)

    def test_the_end_of_ring_object_ends_on_the_last_byte(self):
        _, report = run()
        case = report['cases']['end_of_ring']
        self.assertEqual(case['offset'] + case['object_bytes'], STAGING_MIB << 20)
        self.assertEqual(case['offset'] % 4096, 0)
        self.assertNotEqual(case['offset'] % 1088, 0, 'the offset is not a multiple of the page: the kernel must not assume it')

    def test_the_object_offsets_include_one_that_is_not_a_page_multiple(self):
        _, report = run()
        self.assertTrue(any(case['offset'] % 4096 == 0 and case['offset'] % 1088 != 0 for case in report['cases'].values() if 'offset' in case))

    def test_each_plausible_mistake_fails_the_check(self):
        mistakes = {
            'bad_checksum': 'checksum',
            'flip': 'differs from the staged bytes',
            'forget_block': 'differs from the staged bytes',
            'spill': 'differ from the expected state',
            'ignore_bank': 'bank=',
            'same_data_everywhere': 'differs from the staged bytes',
            'short_info': 'staging_info has 3 entries',
            'address_differs': 'address differs between devices',
            'misaligned': 'not 64-byte aligned',
            'block_bytes_off': 'qwen_block_bytes',
            # the two new forms
            'stride_ignored': 'row_stride',
            'stride_short': 'row_stride',
            'ranges_swapped': 'count=',
            'k_continues': 'count=',
            'count_short': 'count=',
        }
        for knob, text in mistakes.items():
            with self.subTest(mistake=knob):
                _, report = run(**{knob: 1})
                self.assertFalse(report['ok'], knob)
                self.assertTrue(any(text in problem for problem in report['problems']), (knob, report['problems']))

    def test_a_build_without_the_new_keywords_is_the_compat_build_and_still_passes(self):
        """An image carrying the old .so: no row_stride, no count. The card check runs the cases it always ran, says `compat`, and passes (the host works in compat mode)."""
        ttnn = OldBuildTTNN()
        report = card.run(ttnn, FakeMesh(), DEVICES, blocks=1024, heads_per_chip=1, staging_mib=STAGING_MIB, bank=3)
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual((report['device_ops'], report['row_stride'], report['count']), ('compat', False, False))
        self.assertFalse([name for name in report['cases'] if name.startswith('strided') or name.startswith('many') or name.startswith('ranged')])

    def test_requiring_the_new_forms_fails_the_old_build_and_passes_the_new_one(self):
        report = card.run(OldBuildTTNN(), FakeMesh(), DEVICES, blocks=1024, heads_per_chip=1, staging_mib=STAGING_MIB, bank=3, strided=True)
        self.assertFalse(report['ok'])
        self.assertTrue(any('lacks row_stride on qwen_write_blocks_from_staging and count on qwen_staging_checksum' in problem for problem in report['problems']), report['problems'])
        report = card.run(FakeTTNN(), FakeMesh(), DEVICES, blocks=1024, heads_per_chip=1, staging_mib=STAGING_MIB, bank=3, strided=True)
        self.assertTrue(report['ok'], report['problems'])

    def test_skipping_the_new_forms_runs_the_old_cases_on_the_new_build(self):
        report = card.run(FakeTTNN(), FakeMesh(), DEVICES, blocks=1024, heads_per_chip=1, staging_mib=STAGING_MIB, bank=3, strided=False)
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual(report['device_ops'], 'strided', 'what the build has is reported even when the cases are skipped')
        self.assertFalse([name for name in report['cases'] if name.startswith('strided')])

    def test_the_keywords_are_found_from_a_signature_or_a_docstring(self):
        def with_signature(staging, offset, length, bank=None, cores=64, count=None):
            return None

        class DocOnly(object):
            """qwen_staging_checksum(staging: QwenP2PStaging, byte_offset: int, length: int, bank: int | None = None, cores: int = 64, count: int | None = None) -> object"""
            @property
            def __signature__(self):
                raise ValueError('no signature found for builtin')

            def __call__(self, *args, **kwargs):
                return None

        class OldDoc(DocOnly):
            """qwen_staging_checksum(staging: QwenP2PStaging, byte_offset: int, length: int, bank: int | None = None, cores: int = 64) -> list[int]
            Per device, S = sum over the account of the words. Blocking: returns a list with one int per device."""

        self.assertTrue(card.takes(with_signature, 'count'))
        self.assertFalse(card.takes(with_signature, 'row_stride'))
        self.assertTrue(card.takes(DocOnly(), 'count'))
        self.assertFalse(card.takes(OldDoc(), 'count'), 'a word that merely contains `count` (account) is not the keyword')
        self.assertTrue(card.takes(OldDoc(), 'cores'))

    def test_an_extension_without_the_p2p_ops_is_reported_absent(self):
        ttnn = FakeTTNN()
        del FakeTTNN.qwen_staging_checksum
        try:
            report = card.run(ttnn, FakeMesh(), DEVICES, staging_mib=STAGING_MIB)
        finally:
            FakeTTNN.qwen_staging_checksum = _CHECKSUM
        self.assertFalse(report['ok'])
        self.assertEqual(report['status'], 'absent')
        self.assertEqual(report['missing'], ['qwen_staging_checksum'])

    def test_the_staging_is_released_even_when_a_check_raises(self):
        ttnn = FakeTTNN()
        original = ttnn.qwen_staging_checksum

        def boom(*args, **kwargs):
            raise RuntimeError('a device error')
        ttnn.qwen_staging_checksum = boom
        with self.assertRaises(RuntimeError):
            card.run(ttnn, FakeMesh(), DEVICES, staging_mib=STAGING_MIB)
        self.assertTrue(ttnn.staging[0].released)
        ttnn.qwen_staging_checksum = original

    def test_the_formula_mirror_is_the_probe_formula(self):
        self.assertEqual([card.formula_endpoint(b) for b in range(8)],
                         [(17, 12), (17, 15), (17, 18), (17, 21), (18, 12), (18, 15), (18, 18), (18, 21)])

    def test_the_default_run_fits_the_sizes_the_cases_need(self):
        """The long case needs more blocks than one launch carries, and its object must fit the ring."""
        self.assertGreaterEqual(1024, card.LAUNCH_BLOCKS + 80)
        self.assertLess((card.LAUNCH_BLOCKS + 70) * SLICE + card.STAGING_PAGE, STAGING_MIB << 20)
        self.assertLess((card.LAUNCH_BLOCKS + 70) * SLICE + card.STAGING_PAGE, 32 << 20, 'and the default 32 MiB ring of the command line')


class OldBuildTTNN(FakeTTNN):
    """The extension as it was built before row_stride and count: contiguous rows, one checksum range, the same ops otherwise."""

    def qwen_staging_checksum(self, staging, offset, length, bank=None, cores=64):
        return FakeTTNN.qwen_staging_checksum(self, staging, offset, length, bank, cores)

    def qwen_write_blocks_from_staging(self, cache, staging, byte_offset, blocks, bank=None):
        return FakeTTNN.qwen_write_blocks_from_staging(self, cache, staging, byte_offset, blocks, bank)


_CHECKSUM = FakeTTNN.qwen_staging_checksum

if __name__ == '__main__':
    unittest.main()
