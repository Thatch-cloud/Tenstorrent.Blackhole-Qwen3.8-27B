"""The CPU model of the K/V writers (kv_page_model_tp4): the theorem the card settles, stated on a model of the packer.

    page_write == served_write  exactly when  pack(unpack(x)) is the identity on every block pack can emit.

That is plan unknown U1 and only a card can say whether the hardware packer satisfies it. What the CPU holds: the model's faithful packers do (both rounding
modes, both overflow policies, on random and edge blocks), the page writer's transliterated reader / compute / writer then matches the served chain bit for
bit on the shapes the stack writes, and a packer that does NOT satisfy it (a lossy drift, or a sign-of-zero quirk) makes the two differ - so the card's
comparison is not vacuous. Also the model's own layout facts: a bfloat8_b tile is 1,088 bytes, a bfloat16 tile is face-ordered.
"""

import random
import unittest

import kv_page_model_tp4 as model
import kv_page_writer_tp4 as kvpw

FAITHFUL = [model.Bfp8Model('rne', 'saturate'), model.Bfp8Model('trunc', 'bump'), model.Bfp8Model('rne', 'bump'), model.Bfp8Model('trunc', 'saturate')]


def tables_for(groups=4, entries=6):
    """Group g's table: entry e -> block 100 g + e (a block no other group has)."""
    tables = []
    for group in range(groups):
        tables += [[100 * group + entry for entry in range(entries)]] * 16
    return tables


def filled(packer, tables, positions, seed, entries=6):
    """A device-packed cache: every tile of every block the positions' entries and the next one name, from random bfloat16 content."""
    rng = random.Random(seed)
    cache = model.Cache()
    for group in range(4):
        touched = {(position >> 6) for position in positions[group * 16:(group + 1) * 16]}
        for entry in sorted(touched | {max(touched) + 1}):
            if entry >= entries:
                continue
            for tile_row in range(2):
                for column in range(8):
                    rows = [[model.canonical(model.float_to_bf16(rng.gauss(0, 1) * 2.0 ** rng.randint(-6, 6))) for unused in range(32)]
                            for unused in range(32)]
                    cache.put(tables[group * 16][entry], tile_row, column, packer.pack_tile(rows))
    return cache


def positions_of(starts):
    return [start + row for start in starts for row in range(16)]


def scenario(packer, seed, starts, kinds=None, wt=2, ordered=False):
    rng = random.Random(seed)
    tables = tables_for()
    positions = positions_of(starts)
    served = filled(packer, tables, positions, seed)
    page = served.clone()
    rows = []
    for row in range(64):
        if kinds and rng.random() < 0.6:
            rows.append(model.edge_row(rng, rng.choice(kinds)))
        else:
            rows.append(model.random_row(rng))
    model.served_write(packer, served, positions, tables, rows)
    model.page_write(packer, page, positions, tables, rows, wt=wt, ordered=ordered)
    return served, page


class LayoutTests(unittest.TestCase):
    def test_tile_formats(self):
        packer = model.Bfp8Model()
        rows = [[model.float_to_bf16((row * 32 + column + 1) / 64.0) for column in range(32)] for row in range(32)]
        tile = packer.pack_tile(rows)
        self.assertEqual(len(tile), model.BF8_TILE_BYTES)
        self.assertEqual(len(model.tile_bytes_from_rows(rows)), model.BF16_TILE_BYTES)
        self.assertEqual(model.rows_from_tile_bytes(model.tile_bytes_from_rows(rows)), rows)
        again = packer.unpack_tile(tile)
        self.assertEqual(packer.pack_tile(again), tile)

    def test_the_prepared_tile_holds_the_head_row_at_row_zero_of_faces_0_and_1(self):
        row = list(range(1, 33))
        data = model.update_tile(row, garbage=3)
        words = [data[2 * index] | (data[2 * index + 1] << 8) for index in range(1024)]
        self.assertEqual(words[:16], row[:16])
        self.assertEqual(words[256:272], row[16:], 'face 1 starts at byte 512')
        self.assertNotEqual(words[16:32], row[:16], 'row 1 of the face is padding')

    def test_bfloat16_conversions_and_the_canonical_rule(self):
        for value in (0.0, 1.0, -2.5, 3.140625, 2.0 ** -120, 1e30):
            self.assertAlmostEqual(model.bf16_to_float(model.float_to_bf16(value)), value, delta=abs(value) / 100 + 1e-45)
        self.assertEqual([model.canonical(bits) for bits in (0x8000, 0x0001, 0x807F, 0x0080, 0x3F80)], [0, 0, 0, 0x0080, 0x3F80])

    def test_a_zero_tile_unpacks_to_zeros(self):
        self.assertEqual(model.Bfp8Model().unpack_tile(bytes(1088)), [[0] * 32 for unused in range(32)])


class IdempotenceTests(unittest.TestCase):
    def blocks(self, seed, count):
        rng = random.Random(seed)
        found = []
        for index in range(count):
            row = model.random_row(rng) if index % 3 else model.edge_row(rng, model.EDGE_KINDS[index % len(model.EDGE_KINDS)])
            found.append([model.canonical(value) for value in row[:16]])
        return found

    def test_a_faithful_packer_is_the_identity_on_what_it_writes(self):
        for packer in FAITHFUL:
            for block in self.blocks(1, 400):
                shared, packed = packer.pack_block(block)
                unpacked = [model.canonical(value) for value in packer.unpack_block(shared, packed)]
                self.assertEqual(packer.pack_block(unpacked), (shared, packed), (packer.rounding, packer.overflow, block))

    def test_a_sign_of_zero_quirk_breaks_it_on_exactly_the_underflowing_negatives(self):
        packer = model.Bfp8Model(sign_of_zero='keep')
        block = [model.float_to_bf16(1000.0)] + [model.float_to_bf16(-1e-9)] + [model.float_to_bf16(2.0)] * 14
        shared, packed = packer.pack_block(block)
        self.assertEqual(packed[1], 0x80, 'the underflowed negative keeps its sign bit')
        unpacked = [model.canonical(value) for value in packer.unpack_block(shared, packed)]
        self.assertNotEqual(packer.pack_block(unpacked), (shared, packed), 'and the tilize-side flush turns -0 into +0')

    def test_a_lossy_packer_is_not_the_identity(self):
        packer = model.Bfp8Model(drift=True)
        block = model.random_row(random.Random(5))[:16]
        shared, packed = packer.pack_block(block)
        again = packer.pack_block(packer.unpack_block(shared, packed))
        self.assertNotEqual(again, (shared, packed))


class PageEqualsServedTests(unittest.TestCase):
    STARTS = [
        [0, 64, 128, 192],                  # aligned, one tile row each, four pages
        [17, 81, 145, 209],                 # across the 32-row boundary
        [49, 113, 177, 241],                # across the 64-token page boundary
        [31, 96, 160, 224],                 # 31 (two tile rows), then aligned and unaligned middles
        [5, 70, 135, 200],
    ]

    def test_the_page_writer_leaves_the_served_bytes_on_the_stacks_shapes(self):
        for index, starts in enumerate(self.STARTS):
            for packer in FAITHFUL[:2]:
                served, page = scenario(packer, index, starts, kinds=model.EDGE_KINDS)
                self.assertEqual(served.differing(page), [], (starts, packer.rounding, packer.overflow))

    def test_every_unit_width_is_the_same_cache(self):
        starts = [17, 81, 145, 209]
        packer = model.Bfp8Model()
        results = [scenario(packer, 3, starts, wt=wt)[1] for wt in (1, 2, 4, 8)]
        for other in results[1:]:
            self.assertEqual(results[0].differing(other), [])

    def test_the_warm_forwards_ordered_mode_over_a_shared_tile_row_matches_a_single_chain(self):
        packer = model.Bfp8Model()
        rng = random.Random(9)
        tables = [[7, 8, 9, 10, 11, 12]] * 64            # every group on one table: the placeholders' page
        positions = positions_of([2, 2, 2, 2])           # and one tile row
        served = model.Cache()
        for column in range(8):
            served.put(7, 0, column, packer.pack_tile([[model.float_to_bf16(rng.gauss(0, 1)) for unused in range(32)] for unused in range(32)]))
        page = served.clone()
        rows = [model.random_row(rng) for unused in range(64)]
        model.served_write(packer, served, positions, tables, rows)
        model.page_write(packer, page, positions, tables, rows, wt=2, ordered=True)
        self.assertEqual(served.differing(page), [], 'groups in order, last group last: the single chain\'s final rows')
        with self.assertRaisesRegex(AssertionError, 'kv_conflict'):
            model.page_write(packer, served.clone(), positions, tables, rows, wt=2, ordered=False)

    def test_an_invalid_second_slot_writes_nothing(self):
        packer = model.Bfp8Model()
        tables = tables_for()
        positions = positions_of([0, 64, 128, 192])
        cache = filled(packer, tables, positions, 1)
        before = cache.clone()
        rows = [model.random_row(random.Random(index)) for index in range(64)]
        model.page_write(packer, cache, positions, tables, rows)
        touched = {(block, 0) for block in (0, 101, 202, 303)}
        self.assertEqual({(key[0], key[1]) for key in cache.differing(before)}, touched, 'only tile row 0 of each group\'s block')

    def test_the_padding_rows_of_the_prepared_tile_never_reach_the_cache(self):
        packer = model.Bfp8Model()
        tables = tables_for()
        positions = positions_of([0, 64, 128, 192])
        cache = model.Cache()
        rows = [[model.float_to_bf16(1.0)] * 256 for unused in range(64)]
        model.page_write(packer, cache, positions, tables, rows)
        tile = packer.unpack_tile(cache.get(0, 0, 0))
        for row in range(16):
            self.assertEqual(tile[row], [model.float_to_bf16(1.0)] * 32)
        self.assertEqual(tile[16:], [[0] * 32 for unused in range(16)], 'nothing but the 16 written rows')


class PackerThatBreaksIdempotenceTests(unittest.TestCase):
    def test_a_lossy_packer_makes_the_page_writer_differ_from_the_served_chain(self):
        served, page = scenario(model.Bfp8Model(drift=True), 2, [17, 81, 145, 209])
        self.assertTrue(served.differing(page))

    def test_the_sign_of_zero_quirk_makes_them_differ_on_edge_rows(self):
        served, page = scenario(model.Bfp8Model(sign_of_zero='keep'), 4, [17, 81, 145, 209], kinds=('outlier', 'tiny', 'denormal'))
        self.assertTrue(served.differing(page))


if __name__ == '__main__':
    unittest.main()
