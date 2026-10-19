"""How an evicted KV block would sit on an NVMe drive so that one O_DIRECT read per card lands it in that card's DRAM, and what such a restore costs.

Three things, all pure arithmetic with no hardware (CPU tests only):

  1. The layout. The host KV tier keeps a block as ONE payload, chip-interleaved: for each KV cache tensor (layer by layer, K then V), for each chip,
     that chip's slice of the block (qwen_prefix_model_patch._QwenKvTierIO). A card cannot be restored from that with one read: its bytes are
     scattered through the payload. `payload_to_shards` turns a payload into one stream per chip (cache tensor by cache tensor), and a shard object
     (`encode_shard_object`) is a header page plus the shards of up to 100 consecutive blocks of one chip, every size a multiple of 4096 so that
     a single O_DIRECT read of the payload part needs no bounce and no CPU touch.
  2. The direct-to-final-layout arithmetic. Each cache tensor is interleaved page by page over the card's DRAM banks, so a run of consecutive
     block ids is one contiguous range per (tensor, bank). `direct_pieces` counts the pieces and `aligned_run_unit` says how long a run must be for
     a piece to be a whole number of sectors. That is the reason the recommended path stages in one bank and lets a device op scatter.
  3. The bandwidth model. `simulate` is a max-min-fair flow model of drives, cards, a PCIe switch and its uplink, with ACS redirect on or off;
     `restore_seconds` applies it to a session. Link rates are nominal PCIe figures; measured rates are passed in by the caller.

The integrity digest (`block_digest`) is the specification of the card-side check: a checksum cheap enough for a Tensix kernel and strong enough for
media errors and mis-aimed DMA, not for an adversary (the drive's encryption and the cell's authentication cover that).

Stdlib only.
"""

import hashlib
import struct
from math import gcd

SECTOR = 4096
HEADER_BYTES = 4096
MAGIC = b'KVP2'
VERSION = 1
KIND_KV_BLOCK = 1
HASH_BYTES = 32
# What fits in the header page after its fixed part and the payload's sha256: a 32-byte hash and an 8-byte digest per block.
MAX_BLOCKS_PER_OBJECT = (HEADER_BYTES - 56 - 32) // 40


class KvGeometry(object):
    """The shape of the paged attention KV on a mesh, per block.

    The defaults are the TP4 serving geometry: 16 full-attention layers, K and V each a cache tensor (32 tensors per card); one KV head and a
    256-wide head per chip; a block of 64 tokens is 64 x 256 elements = 16 bfloat8_b tiles of 1088 bytes = 17,408 bytes per tensor per chip;
    each tensor is interleaved page by page (one tile a page) over the 8 DRAM banks."""

    def __init__(self, caches=32, chips=4, block_tokens=64, pages_per_slice=16, page_bytes=1088, banks=8):
        self.caches = caches
        self.chips = chips
        self.block_tokens = block_tokens
        self.pages_per_slice = pages_per_slice
        self.page_bytes = page_bytes
        self.banks = banks

    @property
    def slice_bytes(self):
        """One cache tensor, one chip, one block."""
        return self.pages_per_slice * self.page_bytes

    @property
    def shard_bytes(self):
        """One chip's share of one block: every tensor's slice."""
        return self.caches * self.slice_bytes

    @property
    def block_bytes(self):
        """The host tier's payload for one block, all chips."""
        return self.chips * self.shard_bytes

    @property
    def bytes_per_token(self):
        return self.block_bytes / float(self.block_tokens)

    def blocks_for_tokens(self, tokens):
        return -(-tokens // self.block_tokens)


# -- 1. payload <-> shards -------------------------------------------------------------------------------------------------

def payload_to_shards(payload, geometry):
    """The host tier's chip-interleaved payload [cache][chip][slice] -> one bytes per chip, [cache][slice] (a card's whole share of the block)."""
    g = geometry
    if len(payload) != g.block_bytes:
        raise ValueError('a payload of %d bytes, the block is %d' % (len(payload), g.block_bytes))
    view = memoryview(payload)
    shards = []
    for chip in range(g.chips):
        pieces = [view[(cache * g.chips + chip) * g.slice_bytes:(cache * g.chips + chip + 1) * g.slice_bytes] for cache in range(g.caches)]
        shards.append(b''.join(pieces))
    return shards


def shards_to_payload(shards, geometry):
    """The inverse: one bytes per chip -> the chip-interleaved payload the host tier (and its write path) holds."""
    g = geometry
    if len(shards) != g.chips or any(len(shard) != g.shard_bytes for shard in shards):
        raise ValueError('need %d shards of %d bytes' % (g.chips, g.shard_bytes))
    views = [memoryview(shard) for shard in shards]
    pieces = []
    for cache in range(g.caches):
        for chip in range(g.chips):
            pieces.append(views[chip][cache * g.slice_bytes:(cache + 1) * g.slice_bytes])
    return b''.join(pieces)


def block_digest(data):
    """The card-side check of one block shard: over the little-endian 32-bit words w_i, s1 = sum(w_i) and s2 = sum((i+1) * w_i), both mod 2^32,
    returned as s2 << 32 | s1. It catches a flipped bit, a dropped or doubled word and two swapped words; it is meant to be computed by a device
    kernel as the restored pages are written, and compared with the value stored beside the block."""
    if len(data) % 4:
        raise ValueError('a digest is taken over whole 32-bit words')
    s1 = 0
    s2 = 0
    for index, (word,) in enumerate(struct.iter_unpack('<I', data), 1):
        s1 = (s1 + word) & 0xFFFFFFFF
        s2 = (s2 + index * word) & 0xFFFFFFFF
    return s2 << 32 | s1


_HEADER = struct.Struct('<4sHH32sHHIII')       # magic, version, kind, layout id, chip, chips, nblocks, shard_bytes, reserved
_HEADER_FIXED = _HEADER.size


def encode_shard_object(geometry, chip, layout_id, block_hashes, shards_of_blocks):
    """A shard object: HEADER_BYTES of header, then the chip's shard of each block, in chain order. `shards_of_blocks` is, per block, the
    chip's shard bytes. The header holds the block hashes (the chain), the per-block digests, and a sha256 of the payload part."""
    g = geometry
    if not sector_aligned(g):
        raise ValueError('a shard of %d bytes is not a whole number of %d-byte sectors: one O_DIRECT read could not take it' % (g.shard_bytes, SECTOR))
    count = len(block_hashes)
    if not 1 <= count <= MAX_BLOCKS_PER_OBJECT or count != len(shards_of_blocks):
        raise ValueError('1 to %d blocks, with one shard each' % MAX_BLOCKS_PER_OBJECT)
    if len(layout_id) != 32 or any(len(h) != HASH_BYTES for h in block_hashes):
        raise ValueError('a layout id is 32 bytes and a block hash is %d' % HASH_BYTES)
    if any(len(shard) != g.shard_bytes for shard in shards_of_blocks):
        raise ValueError('a shard is %d bytes' % g.shard_bytes)
    payload = b''.join(shards_of_blocks)
    header = bytearray(HEADER_BYTES)
    _HEADER.pack_into(header, 0, MAGIC, VERSION, KIND_KV_BLOCK, layout_id, chip, g.chips, count, g.shard_bytes, 0)
    cursor = _HEADER_FIXED
    for block_hash in block_hashes:
        header[cursor:cursor + HASH_BYTES] = block_hash
        cursor += HASH_BYTES
    for shard in shards_of_blocks:
        struct.pack_into('<Q', header, cursor, block_digest(shard))
        cursor += 8
    header[cursor:cursor + 32] = hashlib.sha256(payload).digest()
    return bytes(header) + payload


def decode_shard_header(header):
    """-> dict(chip, chips, layout_id, hashes, digests, payload_sha256, shard_bytes). Raises ValueError on anything the format does not allow."""
    if len(header) < HEADER_BYTES:
        raise ValueError('a header is %d bytes' % HEADER_BYTES)
    magic, version, kind, layout_id, chip, chips, count, shard_bytes, _ = _HEADER.unpack_from(header, 0)
    if magic != MAGIC or version != VERSION or kind != KIND_KV_BLOCK:
        raise ValueError('not a v%d KV shard object' % VERSION)
    if not 0 <= chip < chips:
        raise ValueError('chip %d of %d' % (chip, chips))
    if not 1 <= count <= MAX_BLOCKS_PER_OBJECT:
        raise ValueError('block count %d' % count)
    cursor = _HEADER_FIXED
    hashes = [bytes(header[cursor + i * HASH_BYTES:cursor + (i + 1) * HASH_BYTES]) for i in range(count)]
    cursor += count * HASH_BYTES
    digests = list(struct.unpack_from('<%dQ' % count, header, cursor))
    cursor += 8 * count
    return {'chip': chip, 'chips': chips, 'layout_id': bytes(layout_id), 'hashes': hashes, 'digests': digests,
            'payload_sha256': bytes(header[cursor:cursor + 32]), 'shard_bytes': shard_bytes}


def object_extent(geometry, blocks_before, blocks):
    """(offset, length) of the payload of `blocks` blocks that follow `blocks_before` in an object: what one O_DIRECT read asks for."""
    g = geometry
    return HEADER_BYTES + blocks_before * g.shard_bytes, blocks * g.shard_bytes


def sector_aligned(geometry, sector=SECTOR):
    """True when every shard, and so every run of shards and the header-plus-payload split, is a whole number of sectors."""
    return geometry.shard_bytes % sector == 0 and HEADER_BYTES % sector == 0


# -- 2. direct to the final layout ----------------------------------------------------------------------------------------

def page_location(geometry, block_id, page_in_slice):
    """(bank, offset in that bank, relative to the tensor's base) of one page of one cache tensor. Interleaved: page p of the tensor is on bank
    p % banks, at (p // banks) pages into that bank's share."""
    g = geometry
    page = block_id * g.pages_per_slice + page_in_slice
    return page % g.banks, (page // g.banks) * g.page_bytes


def runs_of(block_ids):
    """Consecutive block ids as (first, count) runs, in id order."""
    runs = []
    for block_id in sorted(set(block_ids)):
        if runs and runs[-1][0] + runs[-1][1] == block_id:
            runs[-1][1] += 1
        else:
            runs.append([block_id, 1])
    return [tuple(run) for run in runs]


def direct_pieces(geometry, block_ids):
    """The ranges a restore straight into the final layout must write: for each cache tensor, each bank, each run of consecutive block ids,
    one contiguous range (tensor, bank, offset in bank, length). Needs pages_per_slice to divide evenly over the banks."""
    g = geometry
    if g.pages_per_slice % g.banks:
        raise ValueError('a block would not put the same number of pages on every bank')
    per_bank = g.pages_per_slice // g.banks
    pieces = []
    for first, count in runs_of(block_ids):
        offset = first * per_bank * g.page_bytes
        length = count * per_bank * g.page_bytes
        for cache in range(g.caches):
            for bank in range(g.banks):
                pieces.append((cache, bank, offset, length))
    return pieces


def aligned_run_unit(geometry, sector=SECTOR):
    """The fewest consecutive block ids whose per-bank range is a whole number of sectors: below this, a direct read would have to write part
    of a sector it does not own. A run must also START at a multiple of it, because a range's offset in its bank is first_id * per-block bytes
    (see run_is_direct_readable)."""
    g = geometry
    per_block = (g.pages_per_slice // g.banks) * g.page_bytes
    return sector // gcd(sector, per_block)


def run_is_direct_readable(geometry, first, count, sector=SECTOR):
    """True when blocks first .. first+count-1 can be read from a drive straight into the final layout: both ends of every per-bank range fall on a
    sector boundary, i.e. first and count are multiples of aligned_run_unit."""
    unit = aligned_run_unit(geometry, sector)
    return count > 0 and first % unit == 0 and count % unit == 0


# -- 3. bandwidth ---------------------------------------------------------------------------------------------------------

def pcie_gbps(generation, lanes, efficiency=0.93):
    """Nominal payload GB/s per direction of a PCIe link: transfer rate x lanes x encoding / 8 x TLP efficiency."""
    rate = {3: 8.0, 4: 16.0, 5: 32.0}[generation]
    return rate * lanes * (128.0 / 130.0) / 8.0 * efficiency


class Fabric(object):
    """Drives, cards, and what they share.

    drives: name -> dict(read=GB/s, switch=name or None); cards: name -> dict(ingress=GB/s, switch=name or None);
    switches: name -> dict(up=GB/s, down=GB/s) (the uplink, each direction); acs_redirect: P2P between two devices behind one switch goes up
    to the root complex and back (True) or stays in the switch (False); root=GB/s caps what the root complex forwards between ports, or None."""

    def __init__(self, drives, cards, switches=None, acs_redirect=True, root=None):
        capacities = [d['read'] for d in drives.values()] + [c['ingress'] for c in cards.values()]
        for sw in (switches or {}).values():
            capacities += [sw['up'], sw['down']]
        capacities += [] if root is None else [root]
        if any(value <= 0 for value in capacities):
            raise ValueError('every rate must be positive')
        self.drives = drives
        self.cards = cards
        self.switches = switches or {}
        self.acs_redirect = acs_redirect
        self.root = root

    def resources_of(self, drive, card):
        """The shared resources a drive -> card flow uses, and the capacity of each."""
        d, c = self.drives[drive], self.cards[card]
        used = {('drive', drive): d['read'], ('card', card): c['ingress']}
        ds, cs = d.get('switch'), c.get('switch')
        local = ds is not None and ds == cs and not self.acs_redirect
        if not local:
            if ds is not None:
                used[('up', ds)] = self.switches[ds]['up']
            if cs is not None:
                used[('down', cs)] = self.switches[cs]['down']
            if self.root is not None:
                used[('root',)] = self.root
        return used


def simulate(fabric, flows):
    """Seconds until every flow has finished, with max-min fair sharing recomputed each time one finishes. `flows` is a list of
    (drive, card, bytes). Returns (seconds, per-flow finish times)."""
    left = [float(size) for _, _, size in flows]
    uses = [fabric.resources_of(drive, card) for drive, card, _ in flows]
    finish = [0.0] * len(flows)
    now = 0.0
    live = set(index for index, size in enumerate(left) if size > 0)
    while live:
        rate = _fair_rates(live, uses)
        step = min(left[i] / rate[i] / 1e9 for i in live)
        now += step
        for i in list(live):
            left[i] -= rate[i] * 1e9 * step
            if left[i] <= 1e-6 * flows[i][2]:
                finish[i] = now
                live.discard(i)
    return now, finish


def _fair_rates(live, uses):
    capacity = {}
    for i in live:
        capacity.update(uses[i])
    remaining = dict(capacity)
    active = set(live)
    rate = {}
    while active:
        load = {}
        for i in active:
            for key in uses[i]:
                load[key] = load.get(key, 0) + 1
        key = min(load, key=lambda k: remaining[k] / load[k])
        share = remaining[key] / load[key]
        for i in [i for i in active if key in uses[i]]:
            rate[i] = share
            active.discard(i)
            for other in uses[i]:
                remaining[other] -= share
    return rate


def restore_seconds(fabric, bytes_per_card, assignment):
    """Time to restore `bytes_per_card` into every card when card c reads its shard from drive assignment[c] (all four at once)."""
    flows = [(assignment[card], card, bytes_per_card) for card in sorted(fabric.cards)]
    return simulate(fabric, flows)[0]


# -- endurance ------------------------------------------------------------------------------------------------------------

def daily_write_gb(blocks_per_period, period_days, block_bytes):
    return blocks_per_period * block_bytes / 1e9 / period_days


def drive_writes_per_day(daily_gb, capacity_tb):
    return daily_gb / (capacity_tb * 1e3)
