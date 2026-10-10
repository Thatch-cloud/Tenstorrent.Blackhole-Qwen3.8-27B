"""A CPU model of the K/V cache writers, for tests and for the card test's fake device. NOT evidence about silicon.

It models, byte for byte where the repo knows them and by an explicit parameter where only a card can say:

  * bfloat8_b (Bfp8Model): a 32x32 tile is 1,088 bytes - 64 shared-exponent bytes (one per 16-column face row, face-major) then 1,024
    sign-and-7-bit-magnitude bytes in face order. Pack: shared exponent = the block's largest exponent; each magnitude is the element's 8-bit
    significand shifted right by (shared - own + 1) with `rounding` ('rne' | 'trunc') and `overflow` ('saturate' | 'bump'); an element with a
    zero exponent field (zero, denormal) packs as zero (`zero_exponent='flush'`, the unpacker's rule). Unpack is the exact inverse of what pack can
    emit. These are parameters because the packer's rounding is plan unknown U1, not because the model disagrees with the hardware.
  * the served read-modify-write (served_row_write): unpack the row's 8 cache tiles, untilize, replace one row (the update row went through the
    same unpack, so a zero-exponent element is already +0), tilize, pack.
  * the served chained writer (served_write): that, row by row, in row order.
  * the page writer (page_write): the reader / compute / writer of kv_page_writer_tp4.cpp, transliterated - resolve each unit's tile row,
    assemble the synthetic update tiles from 64-byte source spans, untilize both, copy the masked rows, tilize, pack, once per unit.

The theorem the tests hold: when pack(unpack(x)) is the identity on every block pack can emit (idempotence), page_write equals served_write; and
a packer that is not idempotent (a `drift` model) makes them differ, so the card's comparison can fail.
"""

import struct

import kv_page_writer_tp4 as kvpw

TILE_ELEMENTS = 32 * 32
BF8_TILE_BYTES = 1088
BF16_TILE_BYTES = 2048
HALF = 16


# ---- bfloat16 ----

def float_to_bf16(value):
    """The bfloat16 bit pattern nearest to a float (ties to even); NaN and infinity pass."""
    bits = struct.unpack('>I', struct.pack('>f', value))[0]
    if (bits & 0x7F800000) == 0x7F800000:
        return bits >> 16
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16) & 0xFFFF


def bf16_to_float(bits):
    return struct.unpack('>f', struct.pack('>I', (bits & 0xFFFF) << 16))[0]


def canonical(bits):
    """The unpacker's rule (V2's audit; CANON_DENORM in gdn_rows_dma8_tp.cpp): a bfloat16 with a zero exponent field (-0, denormals) comes out +0."""
    return 0 if (bits & 0x7F80) == 0 else bits


# ---- bfloat8_b ----

class Bfp8Model(object):
    """pack / unpack of 16-element blocks and 32x32 tiles.

    rounding       'rne' | 'trunc': how the shifted-out bits of a magnitude are treated.
    overflow       'saturate' | 'bump': a rounded magnitude of 128 becomes 127, or raises the shared exponent by one and repacks.
    sign_of_zero   'drop' | 'keep': whether a magnitude-0 element (a zero, or a value that underflowed against a larger block mate) keeps its sign bit
                   in the tile byte, and unpacks as -0 when it does. 'keep' with the tilize-side flush (canonical()) is the case where
                   pack(unpack(x)) differs from x by the sign of a zero: the packer-quirk the card's edge regimes exist to catch.
    drift          test only: pack loses one more magnitude bit than a faithful packer, so pack(unpack(x)) is NOT the identity."""

    def __init__(self, rounding='rne', overflow='saturate', sign_of_zero='drop', drift=False):
        if rounding not in ('rne', 'trunc') or overflow not in ('saturate', 'bump') or sign_of_zero not in ('drop', 'keep'):
            raise ValueError('unknown packer model')
        self.rounding, self.overflow, self.sign_of_zero, self.drift = rounding, overflow, sign_of_zero, drift

    def _shift(self, significand, shift):
        if shift <= 0:
            return significand
        if shift > 9:
            return 0
        if self.rounding == 'trunc':
            return significand >> shift
        half = 1 << (shift - 1)
        quotient = significand >> shift
        remainder = significand & ((1 << shift) - 1)
        if remainder > half or (remainder == half and (quotient & 1)):
            quotient += 1
        return quotient

    def pack_block(self, elements):
        """(shared exponent, [sign << 7 | magnitude]) of 16 bfloat16 bit patterns. An element with a zero exponent field packs as zero."""
        parts = []
        for bits in elements:
            exponent = (bits >> 7) & 0xFF
            parts.append((bits >> 15, exponent, (128 | (bits & 0x7F)) if exponent else 0))
        shared = max([exponent for sign, exponent, significand in parts if significand] or [0])
        while True:
            packed, bumped = [], False
            for sign, exponent, significand in parts:
                if not significand:
                    packed.append(0)
                    continue
                magnitude = self._shift(significand, shared - exponent + 1)
                if self.drift:
                    magnitude >>= 1
                if magnitude > 127:
                    if self.overflow == 'saturate':
                        magnitude = 127
                    else:
                        bumped = True
                packed.append(((sign << 7) if (magnitude or self.sign_of_zero == 'keep') else 0) | magnitude)
            if not bumped:
                return shared, packed
            shared += 1

    def unpack_block(self, shared, mantissas):
        elements = []
        for byte in mantissas:
            magnitude = byte & 0x7F
            sign = byte >> 7
            if not magnitude:
                elements.append(0x8000 if (sign and self.sign_of_zero == 'keep') else 0)
                continue
            top = magnitude.bit_length() - 1
            exponent = shared - (6 - top)
            if exponent <= 0:
                elements.append(0)
                continue
            fraction = (magnitude << (7 - top)) & 0x7F
            elements.append((sign << 15) | (exponent << 7) | fraction)
        return elements

    def pack_tile(self, rows):
        """1,088 bytes of a 32x32 tile given as 32 rows of 32 bit patterns."""
        exponents = bytearray(64)
        mantissas = bytearray(1024)
        for face in range(4):
            row0, column0 = (face // 2) * 16, (face % 2) * 16
            for row in range(16):
                shared, packed = self.pack_block(rows[row0 + row][column0:column0 + 16])
                exponents[face * 16 + row] = shared
                mantissas[face * 256 + row * 16:face * 256 + row * 16 + 16] = bytes(packed)
        return bytes(exponents) + bytes(mantissas)

    def unpack_tile(self, data):
        if len(data) != BF8_TILE_BYTES:
            raise ValueError('a bfloat8_b tile is %d bytes' % BF8_TILE_BYTES)
        rows = [[0] * 32 for unused in range(32)]
        for face in range(4):
            row0, column0 = (face // 2) * 16, (face % 2) * 16
            for row in range(16):
                shared = data[face * 16 + row]
                block = self.unpack_block(shared, list(data[64 + face * 256 + row * 16:64 + face * 256 + row * 16 + 16]))
                rows[row0 + row][column0:column0 + 16] = block
        return rows


# ---- bfloat16 tiles ----

def tile_bytes_from_rows(rows):
    """The face-ordered 2,048 bytes of 32 rows of 32 bfloat16 bit patterns."""
    out = bytearray(BF16_TILE_BYTES)
    for row in range(32):
        for half in range(2):
            offset = kvpw.face_row_offset(row, half)
            for column in range(16):
                value = rows[row][half * 16 + column]
                out[offset + 2 * column] = value & 0xFF
                out[offset + 2 * column + 1] = value >> 8
    return bytes(out)


def rows_from_tile_bytes(data):
    rows = [[0] * 32 for unused in range(32)]
    for row in range(32):
        for half in range(2):
            offset = kvpw.face_row_offset(row, half)
            for column in range(16):
                rows[row][half * 16 + column] = data[offset + 2 * column] | (data[offset + 2 * column + 1] << 8)
    return rows


def update_tile(row_bits, garbage=0):
    """The prepared K/V tile for one row (column slice of 32 values in tile row 0, as the AttnPrep output holds a one-head chip's row): every
    other tile row carries `garbage`-seeded padding that a correct reader never touches."""
    rows = [[(garbage * 2654435761 + row * 40503 + column * 7919) & 0xFFFF for column in range(32)] for row in range(32)]
    rows[0] = list(row_bits)
    return tile_bytes_from_rows(rows)


# ---- the cache ----

class Cache(object):
    """A one-head paged bfloat8_b cache: tiles keyed (block, tile row, column), unwritten tiles all-zero bytes."""

    def __init__(self, tiles=None):
        self.tiles = dict(tiles or {})

    def get(self, block, tile_row, column):
        return self.tiles.get((block, tile_row, column), bytes(BF8_TILE_BYTES))

    def put(self, block, tile_row, column, data):
        if len(data) != BF8_TILE_BYTES:
            raise ValueError('a bfloat8_b tile is %d bytes' % BF8_TILE_BYTES)
        self.tiles[(block, tile_row, column)] = bytes(data)

    def clone(self):
        return Cache(self.tiles)

    def __eq__(self, other):
        keys = set(self.tiles) | set(other.tiles)
        return all(self.get(*key) == other.get(*key) for key in keys)

    def differing(self, other):
        keys = sorted(set(self.tiles) | set(other.tiles))
        return [key for key in keys if self.get(*key) != other.get(*key)]


def row_slice(row_bits, column):
    """The 32 values of column tile `column` of a 256-wide row."""
    return list(row_bits[column * 32:(column + 1) * 32])


# ---- the served writer ----

def served_row_write(model, cache, block, tile_row, offset, row_bits):
    """ordered_cache's read-modify-write of one row: all 8 cache tiles of the tile row are unpacked, the row replaced, the tiles packed."""
    for column in range(kvpw.ROW_TILES):
        rows = model.unpack_tile(cache.get(block, tile_row, column))
        update = rows_from_tile_bytes(update_tile(row_slice(row_bits, column)))[0]
        rows[offset] = [canonical(value) for value in update]
        cache.put(block, tile_row, column, model.pack_tile([[canonical(value) for value in row] for row in rows]))


def served_write(model, cache, positions, tables, rows_bits, spans=None):
    """The chained writer: row r of the block to cache[tables[r][pos // 64], pos % 64], in row order (spans only name which rows are one chain;
    chains of different users write disjoint tile rows, so row order is the order)."""
    for row, position in enumerate(positions):
        block = int(tables[row][position >> 6])
        served_row_write(model, cache, block, (position >> 5) & 1, position & 31, rows_bits[row])


# ---- the page writer ----

def page_unit(model, cache, positions, table, group, slot, first_column, wt, rows_bits):
    """One unit of kv_page_writer_tp4.cpp: the reader's key and mask, 64-byte source spans, synthetic update tiles; the compute kernel's untilize of
    both and tilize; the writer's masked row copy and write. `positions` is the whole block's, `table` the group's page-table row."""
    first = group * kvpw.GROUP_ROWS
    group_positions = [int(value) for value in positions[first:first + kvpw.GROUP_ROWS]]
    unit = kvpw.resolve_unit(group_positions, table, slot)
    if not unit['valid']:
        return None
    synthetic = [bytearray(BF16_TILE_BYTES) for unused in range(wt)]
    for row in unit['rows']:
        offset = group_positions[row] & 31
        for u in range(wt):
            source = update_tile(row_slice(rows_bits[first + row], first_column + u), garbage=row)
            span_a, span_b = source[0:64], source[512:576]
            synthetic[u][kvpw.face_row_offset(offset, 0):kvpw.face_row_offset(offset, 0) + 32] = span_a[0:32]
            synthetic[u][kvpw.face_row_offset(offset, 1):kvpw.face_row_offset(offset, 1) + 32] = span_b[0:32]
    untilized_update = [[[canonical(value) for value in row] for row in rows_from_tile_bytes(bytes(tile))] for tile in synthetic]
    rows_cache = [model.unpack_tile(cache.get(unit['block'], unit['tile_row'], first_column + u)) for u in range(wt)]
    for offset in range(32):
        if (unit['mask'] >> offset) & 1:
            for u in range(wt):
                rows_cache[u][offset] = untilized_update[u][offset]
    for u in range(wt):
        cache.put(unit['block'], unit['tile_row'], first_column + u,
                  model.pack_tile([[canonical(value) for value in row] for row in rows_cache[u]]))
    return unit


def page_write(model, cache, positions, tables, rows_bits, wt=kvpw.DEFAULT_WT, ordered=False):
    """One cache's share of the page launch, unit by unit. Units never share a tile (kv_conflict); `ordered` runs group g's units after group g-1's,
    as the warm forward's semaphore chain does; otherwise units run in core order (a conflict would then be a race, which the model resolves by
    unit order and the card by luck - tests assert there is none)."""
    order = []
    for group in range(kvpw.GROUPS):
        for slot in range(kvpw.SLOTS):
            for pair in range(kvpw.pairs(wt)):
                order.append((group, slot, pair * wt))
    written = {}
    for group, slot, first_column in order:
        table = tables[group * kvpw.GROUP_ROWS]
        unit = page_unit(model, cache, positions, table, group, slot, first_column, wt, rows_bits)
        if unit is not None:
            for u in range(wt):
                key = (unit['block'], unit['tile_row'], first_column + u)
                if key in written and not ordered:
                    raise AssertionError('two units wrote tile %r: a kv_conflict the host guard must have refused' % (key,))
                written[key] = (group, slot)
    return written


def random_row(rng, scale=1.0):
    """256 bfloat16 bit patterns: gaussian values of random block scale."""
    row = []
    for block in range(16):
        exponent = rng.randint(-8, 8)
        for unused in range(16):
            row.append(float_to_bf16(rng.gauss(0.0, 1.0) * scale * (2.0 ** exponent)))
    return row


def edge_row(rng, kind):
    """256 bit patterns of one edge regime: 'zeros', 'negzero', 'denormal', 'outlier', 'carry', 'huge', 'tiny'."""
    row = []
    for block in range(16):
        for column in range(16):
            if kind == 'zeros':
                row.append(0)
            elif kind == 'negzero':
                row.append(0x8000 if column % 2 else float_to_bf16(rng.gauss(0.0, 1.0)))
            elif kind == 'denormal':
                row.append(rng.choice((0x0001, 0x8001, 0x007F, 0x807F)) if column % 3 == 0 else float_to_bf16(rng.gauss(0.0, 1e-30)))
            elif kind == 'outlier':
                row.append(float_to_bf16(2.0 ** 20) if column == block % 16 else float_to_bf16(rng.gauss(0.0, 1.0)))
            elif kind == 'carry':
                row.append(0x3FFF if column % 2 else 0x3F7F)
            elif kind == 'huge':
                row.append(float_to_bf16(rng.choice((-1, 1)) * 1.9921875 * 2.0 ** 120))
            elif kind == 'tiny':
                row.append(float_to_bf16(rng.choice((-1, 1)) * 1.5 * 2.0 ** -120))
            else:
                raise ValueError('unknown edge kind %r' % (kind,))
    return row


EDGE_KINDS = ('zeros', 'negzero', 'denormal', 'outlier', 'carry', 'huge', 'tiny')
