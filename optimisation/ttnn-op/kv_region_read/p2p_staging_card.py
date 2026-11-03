"""Card check of the P2P KV offload's device ops (version 4 of the qwen_kv_read extension: qwen_p2p_staging_alloc / _info, qwen_blocks_to_staging (the gather),
qwen_write_blocks_from_staging (the scatter), qwen_blocks_digest_async, qwen_p2p_record_event, qwen_p2p_warm / _freeze, and the oracles qwen_staging_checksum,
qwen_p2p_staging_write / _read), with the Gemma geometry: ten KV tensors per chip whose block is 64 pages of 1,088 bytes (69,632 bytes), an object of 696,320 bytes per
chip (ten rows), a ring of 8 banks. No drive and no P2P are needed: the host puts bytes into staging through the command queue (what the drive's DMA will do).

NOT RUN ON CARDS YET. Run it in a card window (never beside production), in a throwaway container of the serving image with no vLLM, as the W2a step 4 does, or by hand:

    python3 p2p_staging_card.py --devices 4 --staging-mib 8

The mesh is opened as production opens it, (devices, 1): the coordinate order the host's worker reads from the info is the one tested. Memory: the numpy shadow of every
cache is 10 x devices x blocks x 69,632 bytes (0.7 GB at the default of 256 blocks and 4 devices, 2.7 GB at 1,024) and each numpy image of the 96 objects of a slot adds
0.27 GB; the whole check peaked at about 1.4 GB of host memory at 256 blocks on the CPU fake (slot case not included: estimate 2 GB with it), where 1,024 blocks would need
over 4 GB, more than a 4 GB container or a rig that also serves has. The device holds 10 caches of the same blocks (0.18 GB per card at 256 blocks).

What it settles, per device and with different bytes on every device and every tensor:
  * the staging buffer: alloc, info (8 banks, one entry per device in mesh order and with that mesh's coordinates, the same range on every device, the 4 KiB lead, the NOC
    coordinates of every bank compared with the UMD/probe formula, the tt-kmd device number and the PCI address of each device from the cluster descriptor, cross-checked
    against sysfs, none of them missing), the DRAM it cost;
  * the GATHER, object-major, exactly as the spill lays an object out (row_stride 696,320, byte_offset = slot base + tensor x 69,632): the staged bytes read back through
    the diagnostics equal the numpy image of ten rows per object; the bytes between the rows are untouched; in each of the 8 banks; blocks the call does not name are not
    read into anything (the caches are unchanged);
  * the DIGEST, polled through ready() without blocking: it equals the numpy reference, it equals the oracle checksum of the object the gather staged (so a gather bug and
    a digest bug cannot hide each other), one flipped byte in one tensor of one block changes that block's digest only, a block named twice digests twice;
  * the SCATTER, from random objects written into staging: the caches' rows equal the staged rows, the blocks not named are untouched, and the digest over the
    DESTINATION rows equals the checksum of the staged object (the restore's check);
  * the round trip gather -> scatter into other blocks: the bytes arrive, and the two digests agree;
  * the event: query() is False or True while work is queued and True once it is through; the host never blocks on it;
  * warm -> freeze: after freeze the three programs keep working and a new program key is refused; before the first capture, as the engine start does;
  * a REAL trace capture (begin_trace_capture on CQ0, one small op so that the trace is not empty): every op that enqueues, and the allocation, raises QwenP2PRefused naming
    the trace; the capture ends and is released; the next valid digest and gather are byte-exact;
  * the refusals (bad bank, offset, stride, ids, counts) raise QwenP2PRefused before anything is enqueued, and the next valid call is unaffected;
  * the cost: enqueue time and completion time of a slot of 96 objects, gather, scatter and digest, and the effective GB/s per card.

Prints one JSON object and exits 1 on any mismatch.
"""

import argparse
import json
import sys
import time

BLOCK_BYTES = 69632                      # one block's row of one tensor on one chip
PAGE_BYTES = 1088                        # one bfp8 tile
PAGES_PER_BLOCK = 64
TENSORS = 10                             # K0, V0 .. K4, V4
SHARD = TENSORS * BLOCK_BYTES            # 696,320: an object on one chip, 170 x 4096
CANARY = 4096                            # the first 4 KiB of a bank's slot is its canary page
STAGING_PAGE = 4096                      # the staging page of this check: byte offsets and lengths of the diagnostics are multiples of it
SLOT_OBJECTS = 96
P2P_NAMES = ('qwen_p2p_staging_alloc', 'qwen_p2p_staging_info', 'qwen_blocks_to_staging', 'qwen_write_blocks_from_staging', 'qwen_blocks_digest_async',
             'qwen_p2p_record_event', 'qwen_p2p_warm', 'qwen_p2p_freeze', 'qwen_staging_checksum', 'qwen_p2p_staging_write', 'qwen_p2p_staging_read')
RAW_NAMES = ('qwen_block_bytes', 'qwen_read_blocks_raw', 'qwen_write_blocks_raw')
POLL_LIMIT_S = 30.0
DEFAULT_BLOCKS = 256                     # blocks of each scratch cache (see the memory note above)


def _ref():
    import p2p_staging_ref
    return p2p_staging_ref


def formula_endpoint(bank):
    """The UMD/probe formula for the first endpoint of a bank (banks 0-3 in column 17, 4-7 in column 18, three endpoints per bank going down from row 12)."""
    return 17 + bank // 4, 12 + 3 * (bank % 4)


def mesh_dims(mesh, devices):
    """The shape of the open mesh as a list; (devices, 1), the production shape, when the mesh does not say."""
    shape = getattr(mesh, 'shape', None)
    try:
        dims = [int(shape[i]) for i in range(len(shape))]
    except Exception:  # noqa: BLE001
        dims = []
    product = 1
    for dim in dims:
        product *= dim
    return dims if dims and all(dim > 0 for dim in dims) and product == devices else [devices, 1]


def row_major_coordinates(dims):
    """The coordinates of a mesh of shape `dims` in row-major order, as the info lists its devices."""
    coordinates = [[]]
    for dim in dims:
        coordinates = [prefix + [index] for prefix in coordinates for index in range(dim)]
    return coordinates


def missing_names(ttnn):
    return [name for name in P2P_NAMES + RAW_NAMES if not callable(getattr(ttnn, name, None))]


def make_cache(ttnn, mesh, pool_blocks, shape_index):
    """A scratch paged cache of the per-chip shape, bfloat8_b tiles in DRAM, replicated over the mesh as the pool is. Two shapes with the same block (64 tiles), like the Gemma
    pool's sliding and full layers. Its bytes are whatever the allocator left: the check writes every block it will rely on."""
    heads, width = ((1, 512), (2, 256))[shape_index % 2]
    return ttnn.empty([pool_blocks, heads, 128, width], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def synchronize(ttnn, mesh):
    sync = getattr(ttnn, 'synchronize_device', None)
    if callable(sync):
        sync(mesh)


def poll(function, what, limit=POLL_LIMIT_S):
    """Polls function() until it is true, never blocking inside it; -> (seconds, polls). Raises when it stays false for `limit` seconds."""
    start = time.perf_counter()
    polls = 0
    while True:
        polls += 1
        if function():
            return time.perf_counter() - start, polls
        if time.perf_counter() - start > limit:
            raise TimeoutError('%s was not ready after %.0f s' % (what, limit))


def check_info(info, devices, staging_bytes, report, problems, dims=None):
    """The description the host aims its window and maps its cards with."""
    banks, listed = info.get('banks', []), info.get('devices', [])
    report['info_banks'], report['info_devices'] = len(banks), len(listed)
    if len(banks) != 8:
        problems.append('staging_info lists %d banks, not 8' % len(banks))
    if len(listed) != devices:
        problems.append('staging_info lists %d devices for %d' % (len(listed), devices))
    if info.get('size') != staging_bytes:
        problems.append('info size %s, asked for %d' % (info.get('size'), staging_bytes))
    if info.get('address', 1) % 64:
        problems.append('the staging address %s is not 64-byte aligned' % info.get('address'))
    lead = info.get('aligned_4k_offset', -1)
    if lead < 0 or lead >= 4096 or (info.get('address', 0) + lead) % 4096 or info.get('aligned_4k_address') != info.get('address', 0) + lead:
        problems.append('aligned_4k_offset %s does not lead from address %s to a 4096-aligned address' % (lead, info.get('address')))
    elif info.get('aligned_4k_size') != (staging_bytes - lead) // 4096 * 4096:
        problems.append('aligned_4k_size %s is not the whole 4096-byte units after the lead' % info.get('aligned_4k_size'))
    if [entry.get('bank') for entry in banks] != list(range(len(banks))):
        problems.append('the banks are not listed in order')
    if any(entry.get('noc_x') is None or entry.get('noc_y') is None for entry in banks):
        problems.append('a bank has no NOC coordinates')
    buses = [entry.get('pci_bus_id') for entry in listed]
    if all(buses) and len(set(buses)) != len(buses):
        problems.append('two devices report the same PCI address')
    # The shard map of the offload is read from these fields (mesh coordinate d -> PCI address -> the drive's p2pmem window): a missing or inconsistent one is a failure here,
    # where the host would otherwise find it at start-up.
    for index, entry in enumerate(listed):
        if not entry.get('pci_bus_id'):
            problems.append('device %d has no PCI address (pci_bus_id is %r): the host cannot map shard %d to a card' % (index, entry.get('pci_bus_id'), index))
        if entry.get('pcie_device') is None:
            problems.append('device %d has no tt-kmd device number (pcie_device is None)' % index)
        if entry.get('pci_bus_id_agrees') is False:
            problems.append('device %d: the cluster descriptor says %s but sysfs says %s for tt-kmd device %s: the chip -> card map is not consistent' % (
                index, entry.get('pci_bus_id'), entry.get('pci_bus_id_sysfs'), entry.get('pcie_device')))
    kmd = [entry.get('pcie_device') for entry in listed]
    if all(number is not None for number in kmd) and len(set(kmd)) != len(kmd):
        problems.append('two devices report the same tt-kmd device number')
    if [entry.get('mesh_index') for entry in listed] != list(range(len(listed))):
        problems.append('the devices are not listed in mesh order')
    wanted = row_major_coordinates(dims if dims else [devices, 1])
    coordinates = [[int(v) for v in entry.get('mesh_coord', [])] for entry in listed]
    if coordinates != wanted:
        problems.append('the devices\' mesh coordinates are %s, not the row-major order %s of a mesh of shape %s' % (coordinates, wanted, dims if dims else [devices, 1]))
    # Not a failure: the formula is an assumption of the probe, the info is tt-metal's own answer, and the report says whether they agree.
    in_endpoints = [tuple(formula_endpoint(entry['bank'])) in [tuple(p) for p in entry.get('noc_endpoints', [])] for entry in banks]
    report['staging'] = dict(address=info.get('address'), size=info.get('size'), aligned_4k_offset=lead, num_banks=info.get('num_banks'),
                             noc=[[entry.get('noc_x'), entry.get('noc_y')] for entry in banks], formula_endpoint_in_info=in_endpoints,
                             pci_bus_ids=buses, pcie_devices=kmd, mmio_chip_ids=[entry.get('mmio_chip_id') for entry in listed],
                             pci_bus_id_sources=[entry.get('pci_bus_id_source') for entry in listed], pci_bus_id_agrees=[entry.get('pci_bus_id_agrees') for entry in listed],
                             mesh_shape=dims if dims else [devices, 1], device_ids=[entry.get('device_id') for entry in listed], bytes_reserved_per_card=info.get('bytes_reserved'),
                             dram_free_per_bank_before=info.get('dram_free_per_bank_before'), dram_free_per_bank_after=info.get('dram_free_per_bank_after'))
    if not all(in_endpoints):
        report['staging']['note'] = 'the probe formula names an endpoint tt-metal does not list for some bank: aim the window at noc_x/noc_y from info'


def run(ttnn, mesh, devices, blocks=DEFAULT_BLOCKS, staging_mib=8, banks_to_test=None, **version_3_arguments):
    """The checks on an open mesh -> the report dict (ok, problems, one entry per case, the cost figures). staging_mib: bytes per bank (a slot of 96 objects needs 64).
    version_3_arguments (heads_per_chip, bank, long_run, strided): the keywords the version 3 host probe passes; they mean nothing here and are ignored, so that probe
    reaches the check, which then reports a version 3 extension as `absent` (it lacks the version 4 ops) instead of dying on a TypeError."""
    import numpy as np

    ref = _ref()
    problems, report, cases = [], {}, {}
    absent = missing_names(ttnn)
    if absent:
        return dict(ok=False, status='absent', missing=absent, problems=['the extension lacks %s' % ', '.join(absent)])
    refused = getattr(ttnn, 'QwenP2PRefused', ValueError)
    rng = np.random.default_rng(20261103)
    staging_bytes = staging_mib << 20
    objects_max = (staging_bytes - CANARY) // SHARD
    if objects_max < 12:
        return dict(ok=False, status='small', problems=['a ring of %d MiB per bank holds %d objects: the check needs at least 12' % (staging_mib, objects_max)])
    staging = ttnn.qwen_p2p_staging_alloc(mesh, staging_bytes, False, STAGING_PAGE)
    caches = []
    try:
        info = ttnn.qwen_p2p_staging_info(staging)
        check_info(info, devices, staging_bytes, report, problems, mesh_dims(mesh, devices))
        caches = [make_cache(ttnn, mesh, blocks, t) for t in range(TENSORS)]
        slice_bytes = int(ttnn.qwen_block_bytes(caches[0]))
        report['slice_bytes'] = slice_bytes
        if any(int(ttnn.qwen_block_bytes(cache)) != BLOCK_BYTES for cache in caches):
            problems.append('qwen_block_bytes is not %d for every cache: the Gemma geometry is 64 pages of 1,088 bytes' % BLOCK_BYTES)
            report['ok'], report['problems'] = False, problems
            return report
        everything = list(range(blocks))
        # What each cache holds, block by block: the state the ops must leave behind (a scatter names its blocks and nothing else).
        shadow = []
        for cache in caches:
            data = rng.integers(0, 256, (devices, blocks, BLOCK_BYTES), dtype=np.uint8)
            ttnn.qwen_write_blocks_raw(cache, data, everything)
            shadow.append(data)

        def read_staging(bank, offset, length):
            back = np.empty((devices, length), dtype=np.uint8)
            ttnn.qwen_p2p_staging_read(staging, offset, back, bank)
            return back

        def object_image(selected, source=None):
            """The numpy image of the objects of `selected` blocks: ten rows per object, tensor order, per device -> uint8 [devices, n, SHARD]."""
            source = shadow if source is None else source
            return np.stack([np.concatenate([source[t][:, b] for t in range(TENSORS)], axis=1) for b in selected], axis=1)

        def digests_of(selected, tensors=None):
            """The async digest, polled without blocking -> ([device][block] ints, seconds to ready, polls)."""
            handle = ttnn.qwen_blocks_digest_async(caches if tensors is None else tensors, selected, PAGES_PER_BLOCK)
            seconds, polls = poll(handle.ready, 'the digest')
            return handle.result(), seconds, polls

        def digest_reference(selected, source=None):
            image = object_image(selected, source)
            return [[ref.checksum_numpy(bytes(image[d, i])) for i in range(len(selected))] for d in range(devices)]

        # 0. the oracle once, before warm: the staging round trip through the diagnostics, and the checksum kernel against numpy
        probe = rng.integers(0, 256, (devices, 2 * STAGING_PAGE), dtype=np.uint8)
        ttnn.qwen_p2p_staging_write(staging, 5 * STAGING_PAGE, probe, 1)
        cases['staging_diagnostics'] = dict(equal=bool(np.array_equal(read_staging(1, 5 * STAGING_PAGE, 2 * STAGING_PAGE), probe)))
        sums = [int(v) for v in ttnn.qwen_staging_checksum(staging, 5 * STAGING_PAGE, 2 * STAGING_PAGE, 1)]
        cases['oracle_checksum'] = dict(equal=sums == [ref.checksum_numpy(bytes(probe[d])) for d in range(devices)])

        # 1. warm and freeze, as the engine start does (before any trace)
        begin = time.perf_counter()
        ttnn.qwen_p2p_warm(staging, caches)
        ttnn.qwen_p2p_freeze(staging)
        cases['warm_freeze'] = dict(seconds=round(time.perf_counter() - begin, 3), frozen=bool(staging.frozen), warmed=bool(staging.warmed))
        for t in range(TENSORS):                      # warm scattered the null block's own bytes back: the caches are as they were
            back = np.empty((devices, 1, BLOCK_BYTES), dtype=np.uint8)
            ttnn.qwen_read_blocks_raw(caches[t], back, [0])
            if not np.array_equal(back[:, 0], shadow[t][:, 0]):
                problems.append('warm changed block 0 of tensor %d (it scatters the null block\'s own bytes back)' % t)
                break

        # 2. the gather, object-major, in every bank
        selected = [int(b) for b in rng.permutation(np.arange(1, blocks))[:12]]
        for bank in (list(range(8)) if banks_to_test is None else banks_to_test):
            name = 'gather_bank_%d' % bank
            base = CANARY
            noise = rng.integers(0, 256, (devices, base + len(selected) * SHARD), dtype=np.uint8)
            ttnn.qwen_p2p_staging_write(staging, 0, noise, bank)
            for t in range(TENSORS):
                ttnn.qwen_blocks_to_staging(caches[t], staging, bank, base + t * BLOCK_BYTES, selected, SHARD, PAGES_PER_BLOCK)
            event = ttnn.qwen_p2p_record_event(mesh)
            seconds, polls = poll(event.query, 'the event')
            got = read_staging(bank, 0, base + len(selected) * SHARD)
            want = noise.copy()
            image = object_image(selected)
            for i in range(len(selected)):
                want[:, base + i * SHARD:base + (i + 1) * SHARD] = image[:, i]
            cases[name] = dict(bytes_equal=bool(np.array_equal(got, want)), event_polls=polls, canary_untouched=bool(np.array_equal(got[:, :base], noise[:, :base])))
            if not cases[name]['bytes_equal']:
                bad = [d for d in range(devices) if not np.array_equal(got[d], want[d])]
                problems.append('%s: the staged image differs from the numpy image (devices %s)' % (name, bad))
            # the digest of the source rows equals the checksum of the object just staged: the spill's invariant
            if bank in (0, 3, 7) or banks_to_test is not None:
                got_digests, _, _ = digests_of(selected)
                staged = ttnn.qwen_staging_checksum(staging, base, SHARD, bank, 64, len(selected))
                cases[name]['digest_equals_staged_checksum'] = [[int(v) for v in row] for row in staged] == [[int(v) for v in row] for row in got_digests]
                if not cases[name]['digest_equals_staged_checksum']:
                    problems.append('%s: the digest of the source rows differs from the checksum of the staged objects' % name)

        # 3. the digest against numpy, repeats, and the sensitivity of the comparison
        big = [int(b) for b in rng.permutation(blocks)[:min(96, blocks)]]
        got_big, seconds_big, polls_big = digests_of(big)
        cases['digest_96'] = dict(equal=got_big == digest_reference(big), seconds_to_ready=round(seconds_big, 6), polls=polls_big)
        repeated, _, _ = digests_of([7, 7, 9, 7])
        cases['digest_repeats'] = dict(equal=repeated == digest_reference([7, 7, 9, 7]), same=all(row[0] == row[1] == row[3] for row in repeated))
        before, _, _ = digests_of([11, 12, 13])
        flipped = np.empty((devices, 1, BLOCK_BYTES), dtype=np.uint8)
        ttnn.qwen_read_blocks_raw(caches[7], flipped, [12])
        original = flipped.copy()
        flipped[1, 0, 4321] ^= 0x20
        ttnn.qwen_write_blocks_raw(caches[7], flipped, [12])
        after, _, _ = digests_of([11, 12, 13])
        ttnn.qwen_write_blocks_raw(caches[7], original, [12])
        changed = [(d, i) for d in range(devices) for i in range(3) if before[d][i] != after[d][i]]
        cases['digest_flipped_byte'] = dict(changed=changed, only_that_block_and_device=changed == [(1, 1)])
        if changed != [(1, 1)]:
            problems.append('a flipped byte of tensor 7, block 12, device 1 changed the digests at %s' % changed)

        # 4. the scatter from random objects, then the digest over the destination rows
        count = 12
        destination = [int(b) for b in rng.permutation(np.arange(1, blocks))[:count]]
        new_rows = [rng.integers(0, 256, (devices, count, BLOCK_BYTES), dtype=np.uint8) for _ in range(TENSORS)]
        staged = np.concatenate([np.concatenate([new_rows[t][:, i] for t in range(TENSORS)], axis=1) for i in range(count)], axis=1)       # [devices, count x SHARD]
        padded = np.zeros((devices, CANARY + count * SHARD), dtype=np.uint8)
        padded[:, CANARY:] = staged
        bank = 5
        ttnn.qwen_p2p_staging_write(staging, 0, padded, bank)
        begin = time.perf_counter()
        for t in range(TENSORS):
            ttnn.qwen_write_blocks_from_staging(caches[t], staging, bank, CANARY + t * BLOCK_BYTES, destination, SHARD, PAGES_PER_BLOCK)
        enqueue_seconds = time.perf_counter() - begin
        landed, seconds_ready, _ = digests_of(destination)
        for t in range(TENSORS):
            shadow[t][:, destination] = new_rows[t]
        want_staged = ttnn.qwen_staging_checksum(staging, CANARY, SHARD, bank, 64, count)
        cases['scatter'] = dict(enqueue_ms=round(enqueue_seconds * 1000, 3),
                                destination_digest_equals_staged_checksum=[[int(v) for v in row] for row in want_staged] == [[int(v) for v in row] for row in landed],
                                destination_digest_equals_numpy=landed == digest_reference(destination))
        for field in ('destination_digest_equals_staged_checksum', 'destination_digest_equals_numpy'):
            if not cases['scatter'][field]:
                problems.append('scatter: %s is false' % field)
        equal = True
        for t in range(TENSORS):
            got = np.empty((devices, count, BLOCK_BYTES), dtype=np.uint8)
            ttnn.qwen_read_blocks_raw(caches[t], got, destination)
            if not np.array_equal(got, new_rows[t]):
                equal = False
                problems.append('scatter: tensor %d read back differs from the staged rows (devices %s)' % (t, [d for d in range(devices) if not np.array_equal(got[d], new_rows[t][d])]))
        cases['scatter']['bytes_equal'] = equal

        # 5. the round trip: gather blocks A of the caches, scatter them into blocks B
        source_blocks = [int(b) for b in rng.permutation(np.arange(1, blocks))[:count]]
        target_blocks = [int(b) for b in rng.permutation(np.arange(1, blocks))[:count]]
        while set(target_blocks) & set(source_blocks):
            target_blocks = [int(b) for b in rng.permutation(np.arange(1, blocks))[:count]]
        bank = 2
        for t in range(TENSORS):
            ttnn.qwen_blocks_to_staging(caches[t], staging, bank, CANARY + t * BLOCK_BYTES, source_blocks, SHARD, PAGES_PER_BLOCK)
        spilled, _, _ = digests_of(source_blocks)
        for t in range(TENSORS):
            ttnn.qwen_write_blocks_from_staging(caches[t], staging, bank, CANARY + t * BLOCK_BYTES, target_blocks, SHARD, PAGES_PER_BLOCK)
        restored, _, _ = digests_of(target_blocks)
        for t in range(TENSORS):
            shadow[t][:, target_blocks] = shadow[t][:, source_blocks]
        cases['round_trip'] = dict(digests_agree=spilled == restored, equals_numpy=restored == digest_reference(target_blocks))
        for field, value in cases['round_trip'].items():
            if not value:
                problems.append('round trip: %s is false' % field)

        # 6. the cost of a slot of 96 objects (gather, scatter, digest), enqueue and completion, in bank 0 (needs the ring for 96 objects)
        if objects_max >= SLOT_OBJECTS and blocks > SLOT_OBJECTS + 1:
            slot = [int(b) for b in rng.permutation(np.arange(1, blocks))[:SLOT_OBJECTS]]
            begin = time.perf_counter()
            for t in range(TENSORS):
                ttnn.qwen_blocks_to_staging(caches[t], staging, 0, CANARY + t * BLOCK_BYTES, slot, SHARD, PAGES_PER_BLOCK)
            enqueue_gather = time.perf_counter() - begin
            digest = ttnn.qwen_blocks_digest_async(caches, slot, PAGES_PER_BLOCK)
            poll(digest.ready, 'the slot digest')
            gather_total = time.perf_counter() - begin
            slot_bytes = SLOT_OBJECTS * SHARD
            cases['slot_96'] = dict(gather_enqueue_ms=round(enqueue_gather * 1000, 3), gather_and_digest_ms=round(gather_total * 1000, 3),
                                    gather_gbps_per_card=round(slot_bytes / max(gather_total, 1e-9) / 1e9, 3), equal=digest.result() == digest_reference(slot))
            begin = time.perf_counter()
            for t in range(TENSORS):
                ttnn.qwen_write_blocks_from_staging(caches[t], staging, 0, CANARY + t * BLOCK_BYTES, slot, SHARD, PAGES_PER_BLOCK)
            enqueue_scatter = time.perf_counter() - begin
            digest = ttnn.qwen_blocks_digest_async(caches, slot, PAGES_PER_BLOCK)
            poll(digest.ready, 'the slot digest after the scatter')
            cases['slot_96'].update(scatter_enqueue_ms=round(enqueue_scatter * 1000, 3), scatter_and_digest_ms=round((time.perf_counter() - begin) * 1000, 3),
                                    scatter_gbps_per_card=round(slot_bytes / max(time.perf_counter() - begin, 1e-9) / 1e9, 3), restored_equal=digest.result() == digest_reference(slot))
            for field in ('equal', 'restored_equal'):
                if not cases['slot_96'][field]:
                    problems.append('slot of 96: %s is false' % field)

        # 7. refusals: before anything is enqueued, and the next valid call is unaffected
        bad_calls = [('bank 8', lambda: ttnn.qwen_blocks_to_staging(caches[0], staging, 8, 0, [1], SHARD, 64)),
                     ('offset', lambda: ttnn.qwen_blocks_to_staging(caches[0], staging, 0, 32, [1], SHARD, 64)),
                     ('513 blocks', lambda: ttnn.qwen_blocks_to_staging(caches[0], staging, 0, 0, list(range(513)), BLOCK_BYTES, 64)),
                     ('pages_per_block', lambda: ttnn.qwen_blocks_to_staging(caches[0], staging, 0, 0, [1], SHARD, 63)),
                     ('block id', lambda: ttnn.qwen_blocks_to_staging(caches[0], staging, 0, 0, [blocks], SHARD, 64)),
                     ('stride', lambda: ttnn.qwen_blocks_to_staging(caches[0], staging, 0, 0, [1], BLOCK_BYTES - 64, 64)),
                     ('duplicate ids in a scatter', lambda: ttnn.qwen_write_blocks_from_staging(caches[0], staging, 0, 0, [1, 1], SHARD, 64)),
                     ('129 blocks digest', lambda: ttnn.qwen_blocks_digest_async(caches, list(range(129)), 64)),
                     ('17 tensors digest', lambda: ttnn.qwen_blocks_digest_async(caches + caches[:7], [1], 64))]
        outcome = {}
        for name, call in bad_calls:
            try:
                call()
                outcome[name] = 'not refused'
                problems.append('refusal: %s was not refused' % name)
            except refused:
                outcome[name] = 'refused'
            except Exception as error:  # noqa: BLE001
                outcome[name] = '%s' % type(error).__name__
                problems.append('refusal: %s raised %s, not QwenP2PRefused' % (name, type(error).__name__))
        cases['refusals'] = outcome
        after_refusals, _, _ = digests_of(big)
        if after_refusals != digest_reference(big):
            problems.append('after the refusals the digests of the blocks differ from numpy: a refusal enqueued something')

        # 8. a program key that was not warmed is refused after freeze
        odd = ttnn.empty([blocks, 1, 64, 512], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)     # 32 pages per block
        try:
            ttnn.qwen_blocks_to_staging(odd, staging, 0, 0, [1], 34816, 32)
            cases['new_key_after_freeze'] = 'not refused'
            problems.append('a program key that was not warmed was not refused after freeze')
        except refused:
            cases['new_key_after_freeze'] = 'refused'
        ttnn.deallocate(odd)

        # 8b. a REAL trace capture (not the forced test hook of the mock device): every op that enqueues refuses behind it, the capture ends, the next valid calls are exact
        def trace_capture_case():
            names = ('begin_trace_capture', 'end_trace_capture', 'release_trace')
            begin, end, release = [getattr(ttnn, name, None) for name in names]
            if not (callable(begin) and callable(end) and callable(release)):
                problems.append('trace capture: this ttnn lacks %s: the refusal during a real capture cannot be checked' % ', '.join(names))
                return dict(started=False)
            probe = ttnn.empty([1, 1, 32, 32], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            first_run = ttnn.add(probe, probe)             # compiled before the capture, as every traced op is; the capture then holds one real command
            synchronize(ttnn, mesh)
            ttnn.deallocate(first_run)
            calls = [('gather', lambda: ttnn.qwen_blocks_to_staging(caches[0], staging, 1, CANARY, selected[:2], SHARD, PAGES_PER_BLOCK)),
                     ('scatter', lambda: ttnn.qwen_write_blocks_from_staging(caches[0], staging, 1, CANARY, selected[:2], SHARD, PAGES_PER_BLOCK)),
                     ('digest', lambda: ttnn.qwen_blocks_digest_async(caches, selected[:2], PAGES_PER_BLOCK)),
                     ('event', lambda: ttnn.qwen_p2p_record_event(mesh)),
                     ('warm', lambda: ttnn.qwen_p2p_warm(staging, caches)),
                     ('oracle_checksum', lambda: ttnn.qwen_staging_checksum(staging, 5 * STAGING_PAGE, STAGING_PAGE, 1)),
                     ('staging_write', lambda: ttnn.qwen_p2p_staging_write(staging, 0, np.zeros((devices, STAGING_PAGE), dtype=np.uint8), 1)),
                     ('staging_read', lambda: ttnn.qwen_p2p_staging_read(staging, 0, np.empty((devices, STAGING_PAGE), dtype=np.uint8), 1)),
                     ('alloc', lambda: ttnn.qwen_p2p_staging_alloc(mesh, staging_bytes, False, STAGING_PAGE))]
            outcome, trace_id, traced = {}, None, None
            try:
                trace_id = begin(mesh, cq_id=0)
                traced = ttnn.add(probe, probe)
                for name, call in calls:
                    try:
                        call()
                        outcome[name] = 'not refused'
                        problems.append('trace capture: %s was not refused during the capture' % name)
                    except refused as error:
                        outcome[name] = 'refused' if 'trace' in str(error) else 'refused for another reason'
                        if 'trace' not in str(error):
                            problems.append('trace capture: %s was refused, but not because of the trace: %s' % (name, str(error)[:160]))
                    except Exception as error:  # noqa: BLE001
                        outcome[name] = type(error).__name__
                        problems.append('trace capture: %s raised %s, not QwenP2PRefused' % (name, type(error).__name__))
            finally:
                if trace_id is not None:
                    try:
                        end(mesh, trace_id, cq_id=0)
                    finally:
                        release(mesh, trace_id)
                ttnn.deallocate(probe)
                if traced is not None:
                    ttnn.deallocate(traced)
            # the next valid calls: the refusals enqueued nothing and the capture left no mark
            digests_after, _, _ = digests_of(big)
            ttnn.qwen_blocks_to_staging(caches[0], staging, 1, CANARY, selected[:2], SHARD, PAGES_PER_BLOCK)
            seconds, polls = poll(ttnn.qwen_p2p_record_event(mesh).query, 'the event after the capture')
            staged = read_staging(1, CANARY, 2 * SHARD)
            gather_equal = all(np.array_equal(staged[:, i * SHARD:i * SHARD + BLOCK_BYTES], shadow[0][:, selected[i]]) for i in range(2))
            return dict(started=True, refusals=outcome, equal=digests_after == digest_reference(big), bytes_equal=bool(gather_equal))
        cases['trace_capture'] = trace_capture_case()

        # 9. whole caches: after everything above, every block of every cache is what the shadow says
        wrong = []
        for t, cache in enumerate(caches):
            full = np.empty((devices, blocks, BLOCK_BYTES), dtype=np.uint8)
            ttnn.qwen_read_blocks_raw(cache, full, everything)
            if not np.array_equal(full, shadow[t]):
                differing = np.nonzero((full != shadow[t]).any(axis=(0, 2)))[0]
                wrong.append((t, [int(b) for b in differing[:8]]))
        report['whole_cache_equal'] = not wrong
        if wrong:
            problems.append('after the cases some blocks differ from the expected state (tensor, first blocks): %s' % wrong)
        for name, case in cases.items():
            if isinstance(case, dict):
                for field in ('equal', 'bytes_equal', 'canary_untouched', 'frozen', 'warmed', 'same'):
                    if field in case and not case[field]:
                        problems.append('%s: %s is false' % (name, field))
        report['cases'] = cases
    finally:
        for cache in caches:
            ttnn.deallocate(cache)
        staging.release()
    report['ok'] = not problems
    report['problems'] = problems
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=DEFAULT_BLOCKS, help='blocks of each scratch cache (host memory: about 10 x devices x blocks x 69,632 bytes; the slot case needs more than 97)')
    parser.add_argument('--staging-mib', type=int, default=64, help='bytes per bank (64 holds the slot of 96 objects)')
    parser.add_argument('--banks', type=int, nargs='*', default=None, help='the banks to test (default: all 8)')
    options = parser.parse_args(argv)

    import ttnn

    absent = missing_names(ttnn)
    if absent:
        try:
            import qwen_kv_read  # noqa: F401
        except Exception as error:  # noqa: BLE001
            print(json.dumps({'ok': False, 'problem': 'qwen_kv_read did not import: %s: %s' % (type(error).__name__, str(error)[:200])}))
            return 1
        absent = missing_names(ttnn)
    if absent:
        print(json.dumps({'ok': False, 'problem': 'this ttnn lacks %s: the extension is not version 4' % ', '.join(absent)}))
        return 1
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(options.devices, 1))      # as production opens it: the coordinate order of the info is the one the worker reads
    try:
        report = run(ttnn, mesh, options.devices, options.blocks, options.staging_mib, options.banks)
    finally:
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
