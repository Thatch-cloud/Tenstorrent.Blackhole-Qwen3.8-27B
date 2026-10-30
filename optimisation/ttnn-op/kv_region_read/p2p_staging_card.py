"""Card check of the P2P restore's device ops (version 3 of the qwen_kv_read extension: ttnn.qwen_p2p_staging_alloc / _info / _write / _read,
qwen_write_blocks_from_staging, qwen_staging_checksum; docs/tp4-p2p-restore.md), in the pattern of kv_region_read_card.py: no drive and no P2P are
needed, the host puts the bytes into the staging range through the command queue (what the drive's DMA will do).

NOT RUN ON CARDS YET. Run it in a card window (never beside production), as the host side's probe does (it calls run() below on the served (1, 4)
mesh), or by hand:

    python3 p2p_staging_card.py --devices 4 --staging-mib 32

What it settles, per device and with different bytes on every device:
  * the staging buffer: alloc, info (one entry per device; the same address on every device; 64-byte aligned; the bank, the NOC coordinates and the
    PCI address the host will aim its window at, compared with the UMD/probe formula and reported), the DRAM it cost (the allocator reserves the range
    in all banks);
  * a known object written into the staging range -> qwen_staging_checksum equals the numpy reference (the contract's algorithm), at several object
    offsets (the start, a 4096-aligned offset that is not a multiple of the 1,088-byte page, the END of the ring), and a single flipped byte changes it
    (the comparison has teeth: it is what makes a mis-aimed or truncated DMA a miss);
  * qwen_write_blocks_from_staging into scratch paged caches of the production geometry (bf8 tile pages, one head per chip: 17,408-byte slices) ->
    qwen_read_blocks_raw reads the same bytes back, byte for byte: one block; a run of consecutive blocks (crossing many pages and chunks); scattered
    and shuffled ids; several cache tensors from one object at the offsets INTERFACE section 1 gives; blocks the call does not name are untouched;
    more blocks than one launch carries when the pool allows it;
  * bank= : the same bytes in another bank of the staging buffer read back through bank=, and a different bank holds different bytes;
  * the cost: GB/s of the checksum and the write op (host-side wall time, the op is waited for).

Prints one JSON object and exits 1 on any mismatch.
"""

import argparse
import json
import sys
import time

BLOCK = 64
HEAD_DIM = 256
# One head's block packed as bfloat8_b: 64 x 256 values are 16 tiles of 1,088 bytes.
PACKED_HEAD_BLOCK_BYTES = 17408
PAGE_BYTES = 1088
P2P_NAMES = ('qwen_p2p_staging_alloc', 'qwen_p2p_staging_info', 'qwen_p2p_staging_write', 'qwen_p2p_staging_read', 'qwen_write_blocks_from_staging',
             'qwen_staging_checksum')
RAW_NAMES = ('qwen_block_bytes', 'qwen_read_blocks_raw', 'qwen_write_blocks_raw')
LAUNCH_BLOCKS = 512                      # qwen_p2p_plan.hpp kScatterLaunchBlocks: one launch of qwen_write_blocks_from_staging
STAGING_PAGE = 4096                      # the staging page of this check: byte offsets and lengths are multiples of it (the diagnostics' granularity)


def _ref():
    import p2p_staging_ref
    return p2p_staging_ref


def formula_endpoint(bank):
    """The UMD/probe formula for the first endpoint of a bank (banks 0-3 in column 17, 4-7 in column 18, three endpoints per bank going down from row 12)."""
    return 17 + bank // 4, 12 + 3 * (bank % 4)


def missing_names(ttnn):
    return [name for name in P2P_NAMES + RAW_NAMES if not callable(getattr(ttnn, name, None))]


def make_cache(ttnn, mesh, pool_blocks, heads_per_chip):
    """A scratch paged cache as production allocates one: zeros of the PER-CHIP shape through ReplicateTensorToMesh, bfloat8_b tiles in DRAM."""
    import torch

    return ttnn.as_tensor(torch.zeros(pool_blocks, heads_per_chip, BLOCK, HEAD_DIM, dtype=torch.bfloat16), device=mesh, dtype=ttnn.bfloat8_b,
                          layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def device_slices(rng, devices, tensors, blocks, slice_bytes):
    """Different random bytes for every (device, tensor, block): uint8 [devices][tensors] -> [blocks, slice_bytes]."""
    import numpy as np

    return [[rng.integers(0, 256, (blocks, slice_bytes), dtype=np.uint8) for _ in range(tensors)] for _ in range(devices)]


def objects_from(slices, ref):
    """The P2P object of each device (INTERFACE section 1): its tensors' [blocks, slice_bytes] arrays concatenated in tensor order, zero padded to 4096."""
    return [ref.object_bytes(per_tensor) for per_tensor in slices]


def stack(objects):
    import numpy as np

    return np.stack([np.frombuffer(obj, dtype=np.uint8) for obj in objects])


def synchronize(ttnn, mesh, cache=None):
    """Waits for the enqueued ops: the device sync when ttnn has it, else a blocking one-block read of `cache` (same command queue, so it is ordered after)."""
    sync = getattr(ttnn, 'synchronize_device', None)
    if callable(sync):
        sync(mesh)
        return
    if cache is not None:
        import numpy as np

        scratch = np.empty((int(mesh.get_num_devices()), 1, int(ttnn.qwen_block_bytes(cache))), dtype=np.uint8)
        ttnn.qwen_read_blocks_raw(cache, scratch, [0])


def check_info(info, devices, staging_bytes, bank, report, problems):
    """The per-device description the host aims its window with."""
    report['info_entries'] = len(info)
    if len(info) != devices:
        problems.append('staging_info has %d entries for %d devices' % (len(info), devices))
        return
    addresses = {entry['address'] for entry in info}
    if len(addresses) != 1:
        problems.append('the staging address differs between devices: %s' % sorted(addresses))
    for index, entry in enumerate(info):
        if entry['size'] != staging_bytes:
            problems.append('device %d: info size %s, asked for %d' % (index, entry['size'], staging_bytes))
        if entry['address'] % 64:
            problems.append('device %d: the staging address %s is not 64-byte aligned' % (index, entry['address']))
        if entry['dram_bank'] != bank:
            problems.append('device %d: info says bank %s, asked for %d' % (index, entry['dram_bank'], bank))
        if entry['noc_x'] is None or entry['noc_y'] is None:
            problems.append('device %d: no NOC coordinates' % index)
    buses = [entry.get('pci_bus_id') for entry in info]
    if all(buses) and len(set(buses)) != len(buses):
        problems.append('two devices report the same PCI address: %s' % buses)
    first = info[0]
    expect = formula_endpoint(bank)
    endpoints = [tuple(point) for point in first.get('noc_endpoints', [])]
    report['staging'] = dict(address=first['address'], size=first['size'], bank=first['dram_bank'], noc=[first['noc_x'], first['noc_y']],
                             noc_endpoints=[list(point) for point in endpoints], formula_first_endpoint=list(expect),
                             formula_first_endpoint_in_info=expect in endpoints, pci_bus_ids=[entry.get('pci_bus_id') for entry in info],
                             device_ids=[entry.get('device_id') for entry in info], pcie_devices=[entry.get('pcie_device') for entry in info],
                             bytes_reserved_per_card=first.get('bytes_reserved'), page_bytes=first.get('page_bytes'),
                             dram_free_per_bank_before=first.get('dram_free_per_bank_before'), dram_free_per_bank_after=first.get('dram_free_per_bank_after'))
    for index, entry in enumerate(info):
        lead, start = entry.get('aligned_4k_offset'), entry['address']
        if lead is None or lead < 0 or lead >= 4096 or (start + lead) % 4096 or entry.get('aligned_4k_address') != start + lead:
            problems.append('device %d: aligned_4k_offset %s does not lead from address %s to a 4096-aligned address' % (index, lead, start))
        elif entry.get('aligned_4k_size', -1) != (staging_bytes - lead) // 4096 * 4096:
            problems.append('device %d: aligned_4k_size %s is not the whole 4096-byte units after the lead' % (index, entry.get('aligned_4k_size')))
    # Not a failure: the formula is an assumption of the probe, the info is the allocator's own answer, and the report says whether they agree.
    if not report['staging']['formula_first_endpoint_in_info']:
        report['staging']['note'] = 'the probe formula names an endpoint tt-metal does not list for this bank: aim the window at noc_x/noc_y from info'


def run(ttnn, mesh, devices, blocks=1024, heads_per_chip=1, staging_mib=32, bank=3, long_run=True):
    """The checks on an open mesh -> the report dict (ok, problems, one entry per case, the cost figures)."""
    import numpy as np

    ref = _ref()
    problems, report = [], {}
    absent = missing_names(ttnn)
    if absent:
        return dict(ok=False, status='absent', missing=absent, problems=['the extension lacks %s' % ', '.join(absent)])
    rng = np.random.default_rng(20261030)
    staging_bytes = staging_mib << 20
    staging = ttnn.qwen_p2p_staging_alloc(mesh, staging_bytes, bank=bank, page_bytes=STAGING_PAGE)
    try:
        info = ttnn.qwen_p2p_staging_info(staging)
        check_info(info, devices, staging_bytes, bank, report, problems)
        cache_a = make_cache(ttnn, mesh, blocks, heads_per_chip)
        slice_bytes = int(ttnn.qwen_block_bytes(cache_a))
        report['slice_bytes'] = slice_bytes
        if slice_bytes != PACKED_HEAD_BLOCK_BYTES * heads_per_chip:
            problems.append('qwen_block_bytes is %d, a 64 x 256 bfloat8_b block per head is %d' % (slice_bytes, PACKED_HEAD_BLOCK_BYTES * heads_per_chip))
            report['ok'] = False
            report['problems'] = problems
            return report
        tensors = 3
        caches = [cache_a] + [make_cache(ttnn, mesh, blocks, heads_per_chip) for _ in range(tensors - 1)]
        cases = {}
        everything = list(range(blocks))
        # What each cache holds, block by block: the state the ops must leave behind (a write names its blocks and nothing else).
        shadow = []
        for cache in caches:
            snapshot = np.empty((devices, blocks, slice_bytes), dtype=np.uint8)
            ttnn.qwen_read_blocks_raw(cache, snapshot, everything)
            shadow.append(snapshot)

        def object_case(name, offset, selected, bank_arg=None, tensor_count=tensors):
            """One object (tensor_count tensors of len(selected) blocks) at `offset`: checksum, then write into the caches, read back."""
            n = len(selected)
            slices = device_slices(rng, devices, tensor_count, n, slice_bytes)
            data = stack(objects_from(slices, ref))
            if offset % STAGING_PAGE or data.shape[1] % STAGING_PAGE or offset + data.shape[1] > staging_bytes:
                raise ValueError('%s: the object does not fit the ring at %d' % (name, offset))
            case = dict(blocks=n, offset=offset, object_bytes=int(data.shape[1]))
            kwargs = {} if bank_arg is None else dict(bank=bank_arg)
            ttnn.qwen_p2p_staging_write(staging, offset, data, **kwargs)
            begin = time.perf_counter()
            sums = [int(v) for v in ttnn.qwen_staging_checksum(staging, offset, int(data.shape[1]), **kwargs)]
            case['checksum_ms'] = round((time.perf_counter() - begin) * 1000, 3)
            want = [ref.checksum_numpy(bytes(data[d])) for d in range(devices)]
            case['checksum_equal'] = sums == want
            if sums != want:
                problems.append('%s: checksum %s, numpy reference %s' % (name, sums, want))
            # The staging holds what was written (a read back through the diagnostics, byte for byte).
            back = np.empty_like(data)
            ttnn.qwen_p2p_staging_read(staging, offset, back, **kwargs)
            case['staging_equal'] = bool(np.array_equal(back, data))
            if not case['staging_equal']:
                problems.append('%s: the staging range does not hold the bytes written' % name)
            offsets = ref.tensor_offsets(n, slice_bytes, tensor_count)
            begin = time.perf_counter()
            for tensor in range(tensor_count):
                ttnn.qwen_write_blocks_from_staging(caches[tensor], staging, offset + offsets[tensor], selected, **kwargs)
            synchronize(ttnn, mesh, caches[0])
            case['write_ms'] = round((time.perf_counter() - begin) * 1000, 3)
            case['write_gbps_per_card'] = round(tensor_count * n * slice_bytes / max(case['write_ms'] / 1000.0, 1e-9) / 1e9, 3)
            equal = True
            for tensor in range(tensor_count):
                got = np.empty((devices, n, slice_bytes), dtype=np.uint8)
                ttnn.qwen_read_blocks_raw(caches[tensor], got, selected)
                expected = np.stack([slices[d][tensor] for d in range(devices)])
                shadow[tensor][:, selected] = expected
                same = bool(np.array_equal(got, expected))
                equal = equal and same
                if not same:
                    problems.append('%s: tensor %d read back differs from the staged bytes (devices %s)' % (
                        name, tensor, [d for d in range(devices) if not np.array_equal(got[d], expected[d])]))
            case['bytes_equal'] = equal
            # Different data on every device: the four read-backs are pairwise different (a write that put one device's bytes everywhere fails).
            case['devices_differ'] = len({hash(data[d].tobytes()) for d in range(devices)}) == devices
            cases[name] = case
            return slices

        # 1. one block, at the start of the ring
        object_case('one_block', 0, [blocks // 2], tensor_count=1)
        # 2. a run of consecutive blocks: crosses many pages and the kernel's chunks; the object sits at an offset that is a multiple of 4096 but not of 1,088
        object_case('run', 3 * STAGING_PAGE, list(range(100, 140)))
        # 3. scattered ids in sorted order, then shuffled
        scattered = sorted(int(x) for x in rng.permutation(blocks)[:57])
        object_case('scattered', 7 * STAGING_PAGE, scattered)
        object_case('shuffled', 11 * STAGING_PAGE, [int(x) for x in rng.permutation(blocks)[:33]])
        # 4. two runs and a singleton, and every cache tensor of an object
        object_case('two_runs_and_one', 13 * STAGING_PAGE, list(range(10, 30)) + list(range(200, 217)) + [999 % blocks])
        # 5. the end of the ring: the object's last byte is the range's last byte
        n_end = 6
        padded = len(ref.object_bytes([np.zeros((n_end, slice_bytes), dtype=np.uint8)] * tensors))
        object_case('end_of_ring', staging_bytes - padded, list(range(300, 300 + n_end)))
        # 6. more blocks than one launch carries (a second launch)
        if long_run and blocks >= LAUNCH_BLOCKS + 80:
            n_long = LAUNCH_BLOCKS + 70
            if n_long * slice_bytes + STAGING_PAGE <= staging_bytes:
                object_case('two_launches', 0, [int(x) for x in rng.permutation(blocks)[:n_long]], tensor_count=1)
        # 7. another bank of the same buffer: the same op with bank=, and the banks hold different bytes
        other = (bank + 1) % (len(info[0].get('banks', [])) or 8)
        object_case('other_bank', 17 * STAGING_PAGE, [41, 42, 43, 400], bank_arg=other, tensor_count=1)
        same_range = [int(v) for v in ttnn.qwen_staging_checksum(staging, 17 * STAGING_PAGE, STAGING_PAGE)]
        other_range = [int(v) for v in ttnn.qwen_staging_checksum(staging, 17 * STAGING_PAGE, STAGING_PAGE, bank=other)]
        cases['other_bank']['banks_differ'] = same_range != other_range
        if same_range == other_range:
            problems.append('the same range reads the same checksum in two banks: bank= does not select the bank')
        # 8. a single flipped byte in the staging range changes the checksum (the comparison has teeth)
        n = 5
        slices = device_slices(rng, devices, 1, n, slice_bytes)
        data = stack(objects_from(slices, ref))
        ttnn.qwen_p2p_staging_write(staging, 0, data)
        good = [int(v) for v in ttnn.qwen_staging_checksum(staging, 0, int(data.shape[1]))]
        flipped = data.copy()
        flipped[1, 12345] ^= 0x40
        ttnn.qwen_p2p_staging_write(staging, 0, flipped)
        bad = [int(v) for v in ttnn.qwen_staging_checksum(staging, 0, int(data.shape[1]))]
        cases['flipped_byte'] = dict(good_equal=good == [ref.checksum_numpy(bytes(data[d])) for d in range(devices)],
                                     only_device_1_changed=[g != b for g, b in zip(good, bad)] == [d == 1 for d in range(devices)])
        if not cases['flipped_byte']['only_device_1_changed']:
            problems.append('a flipped byte on device 1 did not change exactly device 1\'s checksum: good %s bad %s' % (good, bad))
        # 9. a write names only its blocks: after everything above, every block of every cache is what the shadow says (the named ones the staged bytes,
        #    all others as they were at the start)
        wrong = []
        for tensor, cache in enumerate(caches):
            full = np.empty((devices, blocks, slice_bytes), dtype=np.uint8)
            ttnn.qwen_read_blocks_raw(cache, full, everything)
            if not np.array_equal(full, shadow[tensor]):
                differing = np.nonzero((full != shadow[tensor]).any(axis=(0, 2)))[0]
                wrong.append((tensor, [int(b) for b in differing[:8]]))
        report['whole_cache_equal'] = not wrong
        if wrong:
            problems.append('after the cases some blocks differ from the expected state (tensor, first blocks): %s' % wrong)
        report['cases'] = cases
        timed = [case['checksum_ms'] for case in cases.values() if 'checksum_ms' in case and case['object_bytes'] >= (1 << 20)]
        if timed:
            biggest = max((case for case in cases.values() if 'checksum_ms' in case), key=lambda case: case['object_bytes'])
            report['checksum_gbps_per_card'] = round(biggest['object_bytes'] / max(biggest['checksum_ms'] / 1000.0, 1e-9) / 1e9, 3)
        for name, case in cases.items():
            for field in ('checksum_equal', 'staging_equal', 'bytes_equal', 'devices_differ'):
                if field in case and not case[field]:
                    problems.append('%s: %s is false' % (name, field))
        for cache in caches:
            ttnn.deallocate(cache)
    finally:
        staging.release()
    report['ok'] = not problems
    report['problems'] = problems
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=1024, help='blocks of each scratch cache')
    parser.add_argument('--heads-per-chip', type=int, default=1)
    parser.add_argument('--staging-mib', type=int, default=32)
    parser.add_argument('--bank', type=int, default=3)
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
        print(json.dumps({'ok': False, 'problem': 'this ttnn lacks %s: the extension is not version 3' % ', '.join(absent)}))
        return 1
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, options.devices))
    try:
        report = run(ttnn, mesh, options.devices, options.blocks, options.heads_per_chip, options.staging_mib, options.bank)
    finally:
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
