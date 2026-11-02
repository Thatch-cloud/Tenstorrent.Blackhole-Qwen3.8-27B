"""p2p_staging_card.run() on the CPU: the card check of the P2P ops (version 4) against an in-memory fake of the extension (numpy), at four chips, so that the check itself is
known to pass a correct implementation and to FAIL each plausible wrong one before a card is spent on it.

Run from this directory:  python3 -B -m unittest test_p2p_staging_card      (numpy; python 3.10 or later)
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
ROW = card.BLOCK_BYTES
SHARD = card.SHARD
STAGING_MIB = 8
BLOCKS = 40


class Refused(ValueError):
    """QwenP2PRefused."""


class FakeCache(object):
    def __init__(self, shape, devices, knobs):
        tiles = shape[1] * (shape[2] // 32) * (shape[3] // 32)
        self.row = tiles * card.PAGE_BYTES + knobs.get('block_bytes_off', 0)
        self.data = np.zeros((devices, shape[0], tiles * card.PAGE_BYTES), dtype=np.uint8)
        self.pages = tiles


class FakeStaging(object):
    def __init__(self, size, page, devices):
        self.size_bytes, self.page_bytes, self.num_banks = size, page, 8
        self.devices = devices
        self.banks = {}
        self.released = False
        self.frozen = False
        self.warmed = False
        self.keys = set()

    def bank(self, index):
        if index not in self.banks:
            self.banks[index] = np.zeros((self.devices, self.size_bytes), dtype=np.uint8)
        return self.banks[index]

    def release(self):
        self.released = True


class FakeDigest(object):
    def __init__(self, values, ready_after):
        self.values, self.polls, self.ready_after = values, 0, ready_after

    def ready(self):
        self.polls += 1
        return self.polls > self.ready_after

    def result(self):
        assert self.polls > self.ready_after, 'result() before ready()'
        return [[int(v) for v in row] for row in self.values]


class FakeEvent(object):
    def __init__(self, never):
        self.never, self.polls = never, 0

    def query(self):
        self.polls += 1
        return (not self.never) and self.polls > 1


class FakeMesh(object):
    pass


class FakeTTNN(object):
    """Version 4 of the extension on numpy. The knobs are the mistakes the card check must catch."""
    bfloat8_b, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bfloat8_b', 'TILE', 'DRAM'
    QwenP2PRefused = Refused

    def __init__(self, devices=DEVICES, **knobs):
        self.devices = devices
        self.knobs = knobs
        self.staging = []

    def empty(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        return FakeCache(shape, self.devices, self.knobs)

    def deallocate(self, tensor):
        pass

    def qwen_block_bytes(self, cache):
        return cache.row

    def qwen_read_blocks_raw(self, cache, out, blocks):
        out[:] = cache.data[:, list(blocks)]

    def qwen_write_blocks_raw(self, cache, data, blocks):
        cache.data[:, list(blocks)] = data

    def qwen_p2p_staging_alloc(self, mesh, bytes_per_bank, bottom_up=False, page_bytes=65536):
        problem = ref.check_staging(bytes_per_bank, page_bytes, 8, 4 << 30)
        if problem:
            raise Refused(problem)
        staging = FakeStaging(bytes_per_bank, page_bytes, self.devices)
        self.staging.append(staging)
        return staging

    def qwen_p2p_staging_info(self, staging):
        address = 1048576 + 192 + (7 if self.knobs.get('misaligned') else 0)
        lead = (4096 - address % 4096) % 4096
        banks = [dict(bank=b, noc_x=17 + b // 4, noc_y=12 + 3 * (b % 4), noc_endpoints=[[17 + b // 4, 12 + 3 * (b % 4) + p] for p in range(3)], bank_offset=0,
                      address=address) for b in range(8 - (1 if self.knobs.get('short_banks') else 0))]
        devices = [dict(mesh_coord=[0, d], mesh_index=d, device_id=d, pcie_device=d, pci_bus_id='0000:%02x:00.0' % (0x10 + d)) for d in range(self.devices)]
        if self.knobs.get('short_info'):
            devices = devices[:-1]
        return dict(address=address, size=staging.size_bytes, page_bytes=staging.page_bytes, aligned_4k_offset=lead, aligned_4k_address=address + lead,
                    aligned_4k_size=(staging.size_bytes - lead) // 4096 * 4096, banks=banks, devices=devices, num_banks=8, bytes_reserved_per_bank=staging.size_bytes,
                    bytes_reserved=staging.size_bytes * 8, dram_free_per_bank_before=3 << 30, dram_free_per_bank_after=(3 << 30) - staging.size_bytes)

    def _region(self, staging, offset, length):
        if offset % staging.page_bytes or length % staging.page_bytes or offset + length > staging.size_bytes:
            raise Refused('byte_offset and the per-device length must be multiples of the staging page size and inside the range')

    def qwen_p2p_staging_write(self, staging, offset, data, bank=None):
        self._region(staging, offset, data.shape[1])
        staging.bank(0 if bank is None else bank)[:, offset:offset + data.shape[1]] = data

    def qwen_p2p_staging_read(self, staging, offset, out, bank=None):
        self._region(staging, offset, out.shape[1])
        out[:] = staging.bank(0 if bank is None else bank)[:, offset:offset + out.shape[1]]

    def qwen_staging_checksum(self, staging, offset, length, bank=None, cores=64, count=None):
        problem = ref.check_checksum(offset, length, staging.size_bytes, cores, 1 if count is None else count)
        if problem:
            raise Refused(problem)
        data = staging.bank(0 if bank is None else bank)
        if count is None:
            sums = [ref.checksum_numpy(bytes(data[d, offset:offset + length])) for d in range(self.devices)]
            if self.knobs.get('bad_checksum'):
                sums[1] ^= 1
            return sums
        return [[ref.checksum_numpy(bytes(data[d, offset + j * length:offset + (j + 1) * length])) for j in range(count)] for d in range(self.devices)]

    def _check_key(self, staging, cache, op):
        key = cache.pages
        if staging.frozen and key not in staging.keys and not self.knobs.get('freeze_not_enforced'):
            raise Refused('%s: a program key that was not warmed' % op)
        staging.keys.add(key)

    def _rows_call(self, op, cache, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id, distinct):
        blocks = list(blocks)
        if staging.released:
            raise Refused('released')
        if cq_id != 0:
            raise Refused('cq_id')
        if bank >= 8 and not self.knobs.get('refusal_missing'):
            raise Refused('bank')
        problem = ref.check_gather(byte_offset, cache.data.shape[2], card.PAGE_BYTES, pages_per_block, len(blocks), staging.size_bytes, row_stride)
        if problem and not (self.knobs.get('refusal_missing') and 'pages_per_block' in problem):
            raise (RuntimeError(problem) if self.knobs.get('refusal_wrong_type') else Refused(problem))
        reason, _ = ref.check_block_ids(blocks, cache.data.shape[1], distinct)
        if reason:
            raise Refused(reason)
        self._check_key(staging, cache, op)
        return blocks

    def qwen_blocks_to_staging(self, cache, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id=0):
        blocks = self._rows_call('gather', cache, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id, False)
        use = (bank + 1) % 8 if self.knobs.get('gather_wrong_bank') else bank
        stride = cache.data.shape[2] if self.knobs.get('gather_stride_ignored') else row_stride
        data = staging.bank(use)
        for i, block in enumerate(blocks):
            data[:, byte_offset + i * stride:byte_offset + i * stride + cache.data.shape[2]] = cache.data[:, block]
        if self.knobs.get('gather_touches_gaps') and stride == row_stride and row_stride > cache.data.shape[2]:
            for i in range(len(blocks)):
                data[:, byte_offset + i * stride + cache.data.shape[2]:byte_offset + i * stride + cache.data.shape[2] + 64] = 0

    def qwen_write_blocks_from_staging(self, cache, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id=0):
        blocks = self._rows_call('scatter', cache, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id, True)
        data = staging.bank(bank)
        rows = np.stack([data[:, byte_offset + i * row_stride:byte_offset + i * row_stride + cache.data.shape[2]] for i in range(len(blocks))], axis=1).copy()
        if self.knobs.get('scatter_forgets_block'):
            rows, blocks = rows[:, 1:], blocks[1:]
        if self.knobs.get('scatter_shifted'):
            blocks = [(b + 1) % cache.data.shape[1] for b in blocks]
        if self.knobs.get('scatter_flip'):
            rows[2, -1, 5] ^= 1
        cache.data[:, blocks] = rows
        if self.knobs.get('scatter_touches_others'):
            cache.data[:, 3] = rows[:, 0]

    def qwen_blocks_digest_async(self, caches, blocks, pages_per_block, cq_id=0):
        blocks = list(blocks)
        if not caches or len(caches) > 16 or not blocks or len(blocks) > 128:
            raise Refused('count')
        if cq_id != 0:
            raise Refused('cq_id')
        staging = self.staging[-1]
        for cache in caches:
            self._check_key(staging, cache, 'digest')
        order = list(caches)
        if self.knobs.get('digest_tensor_order'):
            order = order[::-1]
        if self.knobs.get('digest_ignores_last_tensor'):
            order = order[:-1]
        values = []
        for d in range(self.devices):
            row = []
            for b in blocks:
                if self.knobs.get('digest_k_restarts'):
                    row.append(sum(ref.checksum_numpy(bytes(c.data[d, b])) for c in order) & ref.MASK64)
                else:
                    row.append(ref.checksum_numpy(b''.join(bytes(c.data[d, b]) for c in order)))
            values.append(row)
        if self.knobs.get('digest_one_device'):
            values = [values[0]] * self.devices
        return FakeDigest(values, 3 if self.knobs.get('slow_digest', True) else 0)

    def qwen_p2p_record_event(self, mesh, cq_id=0):
        return FakeEvent(self.knobs.get('event_never'))

    def qwen_p2p_warm(self, staging, tensors, null_block=0, bank=0, byte_offset=None, cq_id=0):
        for cache in tensors:
            staging.keys.add(cache.pages)
        staging.warmed = True
        if self.knobs.get('warm_changes_null_block'):
            tensors[3].data[:, 0] = 0

    def qwen_p2p_freeze(self, staging):
        if not staging.warmed:
            raise Refused('warm first')
        staging.frozen = not self.knobs.get('freeze_noop')


def run(**knobs):
    ttnn = FakeTTNN(**knobs)
    return ttnn, card.run(ttnn, FakeMesh(), DEVICES, blocks=BLOCKS, staging_mib=STAGING_MIB)


class CardCheckOnTheFakeTests(unittest.TestCase):
    def test_a_correct_extension_passes_every_check(self):
        ttnn, report = run()
        self.assertTrue(report['ok'], report['problems'])
        cases = report['cases']
        self.assertEqual(sorted(cases), sorted(['staging_diagnostics', 'oracle_checksum', 'warm_freeze', 'digest_96', 'digest_repeats', 'digest_flipped_byte', 'scatter',
                                                'round_trip', 'refusals', 'new_key_after_freeze'] + ['gather_bank_%d' % b for b in range(8)]))
        for bank in range(8):
            self.assertTrue(cases['gather_bank_%d' % bank]['bytes_equal'] and cases['gather_bank_%d' % bank]['canary_untouched'], bank)
        for bank in (0, 3, 7):
            self.assertTrue(cases['gather_bank_%d' % bank]['digest_equals_staged_checksum'], bank)
        self.assertTrue(cases['digest_96']['equal'] and cases['digest_repeats']['same'] and cases['digest_flipped_byte']['only_that_block_and_device'])
        self.assertTrue(cases['scatter']['bytes_equal'] and cases['scatter']['destination_digest_equals_staged_checksum'] and cases['scatter']['destination_digest_equals_numpy'])
        self.assertTrue(cases['round_trip']['digests_agree'] and cases['round_trip']['equals_numpy'])
        self.assertEqual(set(cases['refusals'].values()), {'refused'})
        self.assertEqual(cases['new_key_after_freeze'], 'refused')
        self.assertTrue(cases['warm_freeze']['frozen'] and cases['warm_freeze']['warmed'])
        self.assertTrue(report['whole_cache_equal'])
        self.assertTrue(ttnn.staging[0].released, 'the staging buffer is released at the end')
        self.assertEqual(report['staging']['num_banks'], 8)
        self.assertTrue(all(report['staging']['formula_endpoint_in_info']))
        self.assertEqual(report['staging']['bytes_reserved_per_card'], STAGING_MIB * 8 << 20)

    def test_the_digest_is_polled_never_waited_for(self):
        ttnn, report = run()
        self.assertGreater(report['cases']['digest_96']['polls'], 1, 'the fake answers ready() false three times first: the check polls')
        self.assertGreater(report['cases']['gather_bank_0']['event_polls'], 1)

    def test_each_plausible_mistake_fails_the_check(self):
        mistakes = {
            'gather_wrong_bank': 'staged image differs',
            'gather_stride_ignored': 'staged image differs',
            'gather_touches_gaps': 'staged image differs',
            'digest_tensor_order': 'equal is false',
            'digest_k_restarts': 'equal is false',
            'digest_ignores_last_tensor': 'equal is false',
            'digest_one_device': 'equal is false',
            'scatter_shifted': 'read back differs',
            'scatter_forgets_block': 'read back differs',
            'scatter_flip': 'read back differs',
            'scatter_touches_others': 'differ from the expected state',
            'warm_changes_null_block': 'warm changed block 0',
            'freeze_not_enforced': 'was not refused after freeze',
            'refusal_missing': 'was not refused',
            'refusal_wrong_type': 'not QwenP2PRefused',
            'bad_checksum': 'oracle_checksum: equal is false',
            'short_info': 'lists 3 devices',
            'short_banks': 'lists 7 banks',
            'misaligned': 'not 64-byte aligned',
            'block_bytes_off': 'qwen_block_bytes',
        }
        for knob, text in mistakes.items():
            with self.subTest(mistake=knob):
                _, report = run(**{knob: 1})
                self.assertFalse(report['ok'], knob)
                self.assertTrue(any(text in problem for problem in report['problems']), (knob, report['problems']))

    def test_an_event_that_never_completes_is_a_timeout_not_a_hang(self):
        original = card.POLL_LIMIT_S
        card.POLL_LIMIT_S = 0.05
        try:
            with self.assertRaises(TimeoutError):
                run(event_never=1)
        finally:
            card.POLL_LIMIT_S = original

    def test_the_staging_is_released_even_when_a_check_raises(self):
        ttnn = FakeTTNN()
        original = ttnn.qwen_blocks_to_staging

        def boom(*args, **kwargs):
            raise RuntimeError('a device error')
        ttnn.qwen_blocks_to_staging = boom
        with self.assertRaises(RuntimeError):
            card.run(ttnn, FakeMesh(), DEVICES, blocks=BLOCKS, staging_mib=STAGING_MIB)
        self.assertTrue(ttnn.staging[0].released)
        ttnn.qwen_blocks_to_staging = original

    def test_an_extension_without_the_ops_is_reported_absent(self):
        ttnn = FakeTTNN()
        saved = FakeTTNN.qwen_blocks_digest_async
        del FakeTTNN.qwen_blocks_digest_async
        try:
            report = card.run(ttnn, FakeMesh(), DEVICES, staging_mib=STAGING_MIB)
        finally:
            FakeTTNN.qwen_blocks_digest_async = saved
        self.assertFalse(report['ok'])
        self.assertEqual(report['status'], 'absent')
        self.assertEqual(report['missing'], ['qwen_blocks_digest_async'])

    def test_a_ring_too_small_for_the_cases_is_said_so(self):
        report = card.run(FakeTTNN(), FakeMesh(), DEVICES, blocks=BLOCKS, staging_mib=4)
        self.assertFalse(report['ok'])
        self.assertEqual(report['status'], 'small')

    def test_the_version_3_probe_keywords_are_accepted_and_a_version_3_extension_is_reported_absent(self):
        report = card.run(FakeTTNN(), FakeMesh(), DEVICES, blocks=BLOCKS, heads_per_chip=1, staging_mib=STAGING_MIB, bank=0, long_run=False)
        self.assertTrue(report['ok'], report['problems'])
        old = FakeTTNN()
        del old.__class__.qwen_blocks_to_staging
        try:
            report = card.run(old, FakeMesh(), DEVICES, blocks=BLOCKS, heads_per_chip=1, staging_mib=STAGING_MIB, bank=0, long_run=False)
        finally:
            old.__class__.qwen_blocks_to_staging = _GATHER
        self.assertEqual((report['ok'], report['status'], report['missing']), (False, 'absent', ['qwen_blocks_to_staging']))

    def test_the_formula_mirror_is_the_probe_formula(self):
        self.assertEqual([card.formula_endpoint(b) for b in range(8)], [(17, 12), (17, 15), (17, 18), (17, 21), (18, 12), (18, 15), (18, 18), (18, 21)])

    def test_the_geometry_is_gemmas(self):
        self.assertEqual((card.BLOCK_BYTES, card.PAGE_BYTES, card.PAGES_PER_BLOCK, card.TENSORS), (69632, 1088, 64, 10))
        self.assertEqual(card.PAGE_BYTES * card.PAGES_PER_BLOCK, card.BLOCK_BYTES)
        self.assertEqual(card.SHARD, 696320)
        self.assertEqual(card.SHARD % 4096, 0)
        self.assertLessEqual(card.CANARY + card.SLOT_OBJECTS * card.SHARD, 64 << 20, 'a slot of 96 objects fits one 64 MiB bank')

    def test_the_card_script_needs_no_torch(self):
        with open(os.path.join(HERE, 'p2p_staging_card.py'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertNotIn('import torch', text)


_GATHER = FakeTTNN.qwen_blocks_to_staging

if __name__ == '__main__':
    unittest.main()
