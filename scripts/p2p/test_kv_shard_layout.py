import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv_shard_layout as layout


def sliced_payload(geometry):
    """A payload in the host tier's order whose every slice is filled with a value that names its (cache, chip)."""
    g = geometry
    return b''.join(bytes([(cache * 7 + chip * 3 + 1) % 251]) * g.slice_bytes for cache in range(g.caches) for chip in range(g.chips))


class GeometryTest(unittest.TestCase):
    def test_the_serving_geometry_is_the_sized_block_of_the_design_note(self):
        g = layout.KvGeometry()
        self.assertEqual(g.slice_bytes, 16 * 1088)
        self.assertEqual(g.slice_bytes, 17408)
        self.assertEqual(g.shard_bytes, 557056)
        self.assertEqual(g.block_bytes, 2228224)
        # about 35 KB a token over all four chips, a block of 64 tokens about 2.23 MB
        self.assertAlmostEqual(g.bytes_per_token, 34816.0)
        self.assertTrue(2.2e6 < g.block_bytes < 2.25e6)

    def test_a_long_session_in_blocks_and_bytes(self):
        g = layout.KvGeometry()
        blocks = g.blocks_for_tokens(254000)
        self.assertEqual(blocks, 3969)
        self.assertEqual(blocks * g.block_bytes, 3969 * 2228224)
        self.assertTrue(8.8e9 < blocks * g.block_bytes < 8.9e9)
        self.assertEqual(g.blocks_for_tokens(64), 1)
        self.assertEqual(g.blocks_for_tokens(65), 2)

    def test_every_size_that_one_read_touches_is_a_whole_number_of_sectors(self):
        g = layout.KvGeometry()
        self.assertTrue(layout.sector_aligned(g))
        self.assertEqual(g.shard_bytes % 4096, 0)
        for blocks_before in (0, 1, 31, 99):
            offset, length = layout.object_extent(g, blocks_before, 7)
            self.assertEqual(offset % 4096, 0)
            self.assertEqual(length % 4096, 0)

    def test_a_geometry_whose_shard_is_not_sector_sized_says_so(self):
        self.assertFalse(layout.sector_aligned(layout.KvGeometry(caches=3)))


class ShardsTest(unittest.TestCase):
    def test_payload_to_shards_gives_each_chip_only_its_own_slices_cache_by_cache(self):
        g = layout.KvGeometry()
        shards = layout.payload_to_shards(sliced_payload(g), g)
        self.assertEqual(len(shards), g.chips)
        for chip, shard in enumerate(shards):
            self.assertEqual(len(shard), g.shard_bytes)
            for cache in range(g.caches):
                piece = shard[cache * g.slice_bytes:(cache + 1) * g.slice_bytes]
                self.assertEqual(set(piece), {(cache * 7 + chip * 3 + 1) % 251})

    def test_the_round_trip_is_the_identity_on_random_bytes(self):
        g = layout.KvGeometry(caches=6, chips=4, pages_per_slice=16, page_bytes=1088)
        payload = random.Random(5).randbytes(g.block_bytes)
        self.assertEqual(layout.shards_to_payload(layout.payload_to_shards(payload, g), g), payload)

    def test_wrong_sizes_are_refused(self):
        g = layout.KvGeometry()
        with self.assertRaises(ValueError):
            layout.payload_to_shards(b'x' * (g.block_bytes - 1), g)
        with self.assertRaises(ValueError):
            layout.shards_to_payload([b'x'] * g.chips, g)


class ObjectTest(unittest.TestCase):
    def setUp(self):
        self.g = layout.KvGeometry(caches=4, chips=4, pages_per_slice=16, page_bytes=1088)
        self.layout_id = bytes(range(32))
        rng = random.Random(9)
        self.hashes = [rng.randbytes(32) for _ in range(5)]
        self.shards = [rng.randbytes(self.g.shard_bytes) for _ in range(5)]

    def test_header_page_then_the_shards_in_chain_order(self):
        blob = layout.encode_shard_object(self.g, 2, self.layout_id, self.hashes, self.shards)
        self.assertEqual(len(blob), layout.HEADER_BYTES + 5 * self.g.shard_bytes)
        self.assertEqual(len(blob) % 4096, 0)
        self.assertEqual(blob[layout.HEADER_BYTES:], b''.join(self.shards))
        offset, length = layout.object_extent(self.g, 2, 3)
        self.assertEqual(blob[offset:offset + length], b''.join(self.shards[2:5]))

    def test_the_header_reads_back(self):
        blob = layout.encode_shard_object(self.g, 2, self.layout_id, self.hashes, self.shards)
        head = layout.decode_shard_header(blob[:layout.HEADER_BYTES])
        self.assertEqual((head['chip'], head['chips'], head['shard_bytes']), (2, 4, self.g.shard_bytes))
        self.assertEqual(head['layout_id'], self.layout_id)
        self.assertEqual(head['hashes'], self.hashes)
        self.assertEqual(head['digests'], [layout.block_digest(shard) for shard in self.shards])
        import hashlib
        self.assertEqual(head['payload_sha256'], hashlib.sha256(blob[layout.HEADER_BYTES:]).digest())

    def test_a_header_with_the_most_blocks_still_fits_its_page(self):
        self.assertEqual(layout._HEADER.size, 56)
        self.assertEqual(layout.MAX_BLOCKS_PER_OBJECT, 100)
        self.assertLessEqual(56 + layout.MAX_BLOCKS_PER_OBJECT * 40 + 32, layout.HEADER_BYTES)
        self.assertGreater(56 + (layout.MAX_BLOCKS_PER_OBJECT + 1) * 40 + 32, layout.HEADER_BYTES)
        g = layout.KvGeometry(caches=1, chips=1, pages_per_slice=4, page_bytes=1024)
        n = layout.MAX_BLOCKS_PER_OBJECT
        blob = layout.encode_shard_object(g, 0, self.layout_id, [bytes([i]) * 32 for i in range(n)], [bytes(g.shard_bytes)] * n)
        self.assertEqual(len(layout.decode_shard_header(blob)['hashes']), n)
        with self.assertRaises(ValueError):
            layout.encode_shard_object(g, 0, self.layout_id, [b'h' * 32] * (n + 1), [bytes(g.shard_bytes)] * (n + 1))

    def test_a_geometry_whose_shard_is_not_sector_sized_cannot_be_written(self):
        g = layout.KvGeometry(caches=3, chips=4)
        with self.assertRaisesRegex(ValueError, 'sectors'):
            layout.encode_shard_object(g, 0, self.layout_id, [b'h' * 32], [bytes(g.shard_bytes)])

    def test_the_chip_must_be_one_of_the_chips(self):
        blob = bytearray(layout.encode_shard_object(self.g, 2, self.layout_id, self.hashes, self.shards)[:layout.HEADER_BYTES])
        struct_chip = layout._HEADER
        magic, version, kind, layout_id, chip, chips, count, shard_bytes, reserved = struct_chip.unpack_from(blob, 0)
        struct_chip.pack_into(blob, 0, magic, version, kind, layout_id, 9, chips, count, shard_bytes, reserved)
        with self.assertRaisesRegex(ValueError, 'chip 9 of 4'):
            layout.decode_shard_header(bytes(blob))

    def test_what_the_format_does_not_allow_is_refused(self):
        blob = layout.encode_shard_object(self.g, 0, self.layout_id, self.hashes, self.shards)
        with self.assertRaises(ValueError):
            layout.decode_shard_header(b'XXXX' + blob[4:layout.HEADER_BYTES])
        with self.assertRaises(ValueError):
            layout.decode_shard_header(blob[:100])
        with self.assertRaises(ValueError):
            layout.encode_shard_object(self.g, 0, self.layout_id, self.hashes[:2], self.shards)
        with self.assertRaises(ValueError):
            layout.encode_shard_object(self.g, 0, self.layout_id, self.hashes, self.shards[:4] + [b'short'])
        with self.assertRaises(ValueError):
            layout.encode_shard_object(self.g, 0, b'short', self.hashes, self.shards)


class DigestTest(unittest.TestCase):
    def test_a_known_value(self):
        data = b''.join(w.to_bytes(4, 'little') for w in (1, 2, 3))
        self.assertEqual(layout.block_digest(data), (1 * 1 + 2 * 2 + 3 * 3) << 32 | 6)

    def test_it_catches_what_it_is_for(self):
        data = bytearray(random.Random(1).randbytes(4096))
        base = layout.block_digest(bytes(data))
        flipped = bytearray(data)
        flipped[1000] ^= 0x10
        self.assertNotEqual(layout.block_digest(bytes(flipped)), base)
        swapped = bytearray(data)
        swapped[0:4], swapped[8:12] = data[8:12], data[0:4]
        if bytes(swapped) != bytes(data):
            self.assertNotEqual(layout.block_digest(bytes(swapped)), base)
        dropped = bytes(data[:100]) + bytes(data[104:]) + b'\0\0\0\0'
        self.assertNotEqual(layout.block_digest(dropped), base)
        doubled = bytes(data[:100]) + bytes(data[100:104]) + bytes(data[100:104]) + bytes(data[104:-4])
        self.assertNotEqual(layout.block_digest(doubled), base)

    def test_whole_words_only(self):
        with self.assertRaises(ValueError):
            layout.block_digest(b'abc')


class DirectLayoutTest(unittest.TestCase):
    def setUp(self):
        self.g = layout.KvGeometry()

    def test_a_blocks_pages_alternate_over_the_banks_two_to_a_bank(self):
        banks = [layout.page_location(self.g, 0, t)[0] for t in range(16)]
        self.assertEqual(banks, list(range(8)) * 2)
        self.assertEqual([layout.page_location(self.g, 0, t)[1] for t in (0, 7, 8, 15)], [0, 0, 1088, 1088])
        # block 1 starts 2 pages further into each bank
        self.assertEqual(layout.page_location(self.g, 1, 0), (0, 2 * 1088))
        self.assertEqual(layout.page_location(self.g, 5, 3), (3, 10 * 1088))

    def test_a_run_of_consecutive_blocks_is_one_contiguous_range_per_tensor_and_bank(self):
        pieces = layout.direct_pieces(self.g, range(64, 96))
        self.assertEqual(len(pieces), 32 * 8)
        cache, bank, offset, length = pieces[0]
        self.assertEqual((cache, bank, offset, length), (0, 0, 64 * 2 * 1088, 32 * 2 * 1088))
        # and the range is what the per-page rule says: its first page is block 64's first page on that bank
        self.assertEqual(layout.page_location(self.g, 64, 0), (0, offset))
        self.assertEqual(len({piece[3] for piece in pieces}), 1)

    def test_fragmented_block_ids_multiply_the_pieces(self):
        ids = [0, 1, 2, 10, 11, 40]
        self.assertEqual(layout.runs_of(ids), [(0, 3), (10, 2), (40, 1)])
        self.assertEqual(len(layout.direct_pieces(self.g, ids)), 3 * 32 * 8)

    def test_a_run_must_be_a_whole_number_of_sectors_per_bank_to_be_read_directly(self):
        # 2 pages of 1088 bytes per block per bank: 2176 = 17 * 128, so 32 blocks (2048 tokens) fill 17 sectors of 4096, 4 blocks fill 17 of 512
        self.assertEqual(layout.aligned_run_unit(self.g), 32)
        self.assertEqual(layout.aligned_run_unit(self.g, 512), 4)
        unit = layout.aligned_run_unit(self.g)
        _, _, _, length = layout.direct_pieces(self.g, range(unit))[0]
        self.assertEqual(length % 4096, 0)
        _, _, _, short = layout.direct_pieces(self.g, range(unit - 1))[0]
        self.assertNotEqual(short % 4096, 0)

    def test_a_run_must_start_on_the_unit_as_well_as_be_a_multiple_of_it(self):
        self.assertTrue(layout.run_is_direct_readable(self.g, 0, 32))
        self.assertTrue(layout.run_is_direct_readable(self.g, 64, 64))
        self.assertFalse(layout.run_is_direct_readable(self.g, 16, 32))       # 32 long but starting mid-unit
        self.assertFalse(layout.run_is_direct_readable(self.g, 0, 16))
        self.assertFalse(layout.run_is_direct_readable(self.g, 0, 0))
        self.assertTrue(layout.run_is_direct_readable(self.g, 4, 4, sector=512))
        # and the arithmetic behind it: the byte offset of a run in its bank is a whole number of sectors only from such a start
        per_block = 2 * 1088
        self.assertNotEqual((16 * per_block) % 4096, 0)

    def test_pages_must_divide_over_the_banks(self):
        with self.assertRaises(ValueError):
            layout.direct_pieces(layout.KvGeometry(pages_per_slice=12), [0])


class BandwidthTest(unittest.TestCase):
    def test_nominal_link_rates(self):
        self.assertTrue(14.0 < layout.pcie_gbps(5, 4) < 15.75)
        self.assertTrue(7.0 < layout.pcie_gbps(4, 4) < 7.88)
        self.assertAlmostEqual(layout.pcie_gbps(5, 16), 4 * layout.pcie_gbps(5, 4))

    def test_one_drive_feeding_four_cards_is_the_drives_rate(self):
        fabric = layout.Fabric({'d0': {'read': 14.0}}, {'c%d' % i: {'ingress': 20.0} for i in range(4)})
        seconds = layout.restore_seconds(fabric, 2.14e9, {'c%d' % i: 'd0' for i in range(4)})
        self.assertAlmostEqual(seconds, 4 * 2.14e9 / 14e9, places=6)

    def test_a_drive_per_card_is_the_cards_ingress_when_that_is_lower(self):
        drives = {'d%d' % i: {'read': 14.0} for i in range(4)}
        cards = {'c%d' % i: {'ingress': 13.0} for i in range(4)}
        seconds = layout.restore_seconds(layout.Fabric(drives, cards), 2.0e9, {'c%d' % i: 'd%d' % i for i in range(4)})
        self.assertAlmostEqual(seconds, 2.0e9 / 13e9, places=6)

    def test_acs_redirect_sends_switch_local_traffic_through_the_uplink(self):
        drives = {'d0': {'read': 14.0, 'switch': 's'}, 'd1': {'read': 14.0, 'switch': 's'}}
        cards = {'c0': {'ingress': 14.0, 'switch': 's'}, 'c1': {'ingress': 14.0, 'switch': 's'}}
        switches = {'s': {'up': 14.0, 'down': 14.0}}
        assign = {'c0': 'd0', 'c1': 'd1'}
        with_redirect = layout.restore_seconds(layout.Fabric(drives, cards, switches, acs_redirect=True), 1e9, assign)
        local = layout.restore_seconds(layout.Fabric(drives, cards, switches, acs_redirect=False), 1e9, assign)
        self.assertAlmostEqual(with_redirect, 2 * local, places=6)

    def test_a_flow_that_finishes_gives_its_share_to_the_others(self):
        fabric = layout.Fabric({'d': {'read': 10.0}}, {'a': {'ingress': 99.0}, 'b': {'ingress': 99.0}})
        seconds, finish = layout.simulate(fabric, [('d', 'a', 10e9), ('d', 'b', 5e9)])
        # both at 5 GB/s until b's 5 GB are done at 1 s, then a alone at 10 GB/s for its remaining 5 GB
        self.assertAlmostEqual(finish[1], 1.0, places=6)
        self.assertAlmostEqual(seconds, 1.5, places=6)

    def test_a_card_behind_a_switch_reached_from_a_root_port_drive_uses_the_downlink_only(self):
        fabric = layout.Fabric({'d': {'read': 14.0}}, {'c': {'ingress': 99.0, 'switch': 's'}}, {'s': {'up': 1.0, 'down': 7.0}})
        self.assertAlmostEqual(layout.restore_seconds(fabric, 7e9, {'c': 'd'}), 1.0, places=6)

    def test_the_fair_share_takes_each_bottleneck_in_turn(self):
        # d1 (10) feeds a and b; d2 (99) feeds c; c and b share card x (6). Max-min: b = c = 3 on x, then a gets d1's remaining 7.
        fabric = layout.Fabric({'d1': {'read': 10.0}, 'd2': {'read': 99.0}},
                               {'a': {'ingress': 99.0}, 'b': {'ingress': 6.0}, 'c': {'ingress': 6.0}})
        uses = [fabric.resources_of('d1', 'a'), fabric.resources_of('d1', 'b'), fabric.resources_of('d2', 'b')]
        rates = layout._fair_rates({0, 1, 2}, uses)
        self.assertEqual({k: round(v, 6) for k, v in rates.items()}, {0: 7.0, 1: 3.0, 2: 3.0})

    def test_a_rate_that_is_not_positive_is_refused(self):
        with self.assertRaises(ValueError):
            layout.Fabric({'d': {'read': 0.0}}, {'c': {'ingress': 1.0}})
        with self.assertRaises(ValueError):
            layout.Fabric({'d': {'read': 1.0}}, {'c': {'ingress': 1.0, 'switch': 's'}}, {'s': {'up': 1.0, 'down': 0.0}})

    def test_no_data_takes_no_time(self):
        fabric = layout.Fabric({'d': {'read': 10.0}}, {'a': {'ingress': 10.0}})
        self.assertEqual(layout.simulate(fabric, [('d', 'a', 0)])[0], 0.0)


class EnduranceTest(unittest.TestCase):
    def test_daily_writes_and_drive_writes_per_day(self):
        self.assertAlmostEqual(layout.daily_write_gb(70, 7, 2e9), 20.0)
        self.assertAlmostEqual(layout.drive_writes_per_day(20.0, 4.0), 0.005)


if __name__ == '__main__':
    unittest.main()
