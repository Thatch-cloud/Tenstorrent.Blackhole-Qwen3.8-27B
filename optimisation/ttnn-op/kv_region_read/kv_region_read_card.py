"""Card check for ttnn.qwen_read_blocks (the qwen_kv_read extension, optimisation/ttnn-op/kv_region_read; the draft
optimisation/sim/kv-region-read.patch is its source): the region read of a paged KV cache against the whole-cache read, byte for
byte, on a mesh, and what it costs.

NOT RUN ON CARDS YET: the extension is built (build_kv_read.sh) and imports in the production image, but no card has run it.
Run it in a card window (never beside production), as the QUALIFY job does (scripts/ci/tp4_kv_read_probe.py, which opens the
served (1, 4) mesh and calls run() below), or by hand:

    python3 kv_region_read_card.py --devices 4 --blocks 4096 --heads-per-chip 1

Version 2 of the extension adds the host KV tier's raw ops (ttnn.qwen_block_bytes, qwen_read_blocks_raw, qwen_write_blocks_raw); run_raw checks them
when they exist (a version-1 image reports raw=absent and is not failed for it: the tier then refuses to attach). The raw arms (Q1' of the tier
plan): the slice size equals the cache's packed block size; a raw read of the five block sets, written into a second cache at the same ids, reads
back byte-identical through the raw read AND unpacks to the same values as the first cache through the region read (so raw bytes in are the
bytes out, and they are the model's bytes); a deliberately changed byte does not read back as the original (the comparison has teeth); no
program-cache growth; and the GB/s of 32, 256, 1024 and 4096 blocks each way, idle (with a decode trace replaying is the window's
timing job, not this probe).

Prints one JSON object and exits 1 on any mismatch. What it settles (the questions the patch leaves open):
  * the host tensor ttnn.allocate_tensor_on_host(shape=[n, heads_per_chip, 64, 256]) really is the per-device shard
    shape qwen_read_blocks expects (the read refuses a mismatch with a TT_FATAL, so a wrong guess fails loud);
  * several region reads of one shard in a single enqueue_read_shards land where the host offsets say;
  * the unpacked values equal the whole-cache read's, for runs, singletons and a shuffled order;
  * the program cache does not grow (the read compiles nothing) - the property the audit exists to keep;
  * the read volume follows the row: one block costs a small fraction of the whole read (SCALING_FACTOR), and the largest
    set takes less than the whole-cache device read;
  * the cost per block and per read call, which the W-1 minutes are based on (docs/prefix-audit-cost.md);
  * the split of the whole-cache read into the device read (ttnn.from_device: the packed bytes) and the host unpack
    (ttnn.to_torch of the host tensor), the measurement the cost model still lacks (it says whether a packed read
    plus a host-side block select, with no C++, would be enough).

The cache is allocated as production allocates its KV (qwen36_model._allocate_kv_caches_tp: zeros of the PER-CHIP
shape through ReplicateTensorToMesh), then filled with different values on every chip (a copy from a sharded tensor,
which keeps the destination's replicated topology), so the topology the region read and the composer see is the served
one. Every region read must also unpack to the whole-cache read's shape (heads per chip x chips), never a subset of
the chips.
"""

import argparse
import json
import os
import sys
import time

import torch

BLOCK = 64
HEAD_DIM = 256
# One block's read must cost less than this fraction of the whole cache's device read (a read that follows the pool, not
# the row, costs the same for one block as for all of them).
SCALING_FACTOR = 0.1
MIN_TIMED_WHOLE_READ_S = 0.5


def block_sets(blocks, seed=7):
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(blocks, generator=generator)[: max(8, blocks // 16)].tolist()
    return {
        'one': [blocks // 2],
        'run': list(range(5, 5 + min(512, blocks // 4))),
        'scattered': sorted(shuffled),
        'shuffled_order': shuffled,
        'two_runs': list(range(10, 40)) + list(range(100, 140)),
    }


def ensure_extension(ttnn, fallback_dir=None):
    """-> (problem, source). problem is None when ttnn.qwen_read_blocks exists, else the reason it does not. source says where the
    extension came from: 'present' (ttnn already had the name), 'baked' (qwen_kv_read imported from the image) or 'mount'
    (imported from fallback_dir, the rig graft mounted over an image that lacks it); the baked one is always tried first, so a
    mount can never shadow what the image carries."""
    if callable(getattr(ttnn, 'qwen_read_blocks', None)):
        return None, 'present'
    reason = None
    try:
        import qwen_kv_read  # noqa: F401
        source = 'baked'
    except Exception as error:  # noqa: BLE001
        reason, source = 'qwen_kv_read did not import: %s: %s' % (type(error).__name__, str(error)[:200]), None
    if source is None and fallback_dir and os.path.isfile(os.path.join(fallback_dir, 'qwen_kv_read.so')):
        sys.path.append(fallback_dir)
        try:
            import qwen_kv_read  # noqa: F401,F811
            reason, source = None, 'mount'
        except Exception as error:  # noqa: BLE001
            reason = 'qwen_kv_read did not import from %s: %s: %s' % (fallback_dir, type(error).__name__, str(error)[:200])
    if callable(getattr(ttnn, 'qwen_read_blocks', None)):
        return None, source
    return reason or 'this ttnn has no qwen_read_blocks', None


RAW_SIZES = (32, 256, 1024, 4096)
# One head's block packed as bfloat8_b: 64 x 256 values are 16 tiles of 1,088 bytes (a module constant so the CPU fake, which models
# bfloat16, can name its own).
PACKED_HEAD_BLOCK_BYTES = 17408


def run_raw(ttnn, mesh, devices, cache, blocks, heads_per_chip, composer):
    """The version-2 raw block ops against the region read, byte for byte -> (report dict, problems). cache is the filled device cache run() built."""
    import numpy as np

    problems, report = [], {}
    names = ('qwen_block_bytes', 'qwen_read_blocks_raw', 'qwen_write_blocks_raw')
    absent = [name for name in names if not callable(getattr(ttnn, name, None))]
    if absent:
        return dict(status='absent', missing=absent), problems
    slice_bytes = int(ttnn.qwen_block_bytes(cache))
    want_bytes = PACKED_HEAD_BLOCK_BYTES * heads_per_chip
    report['slice_bytes'] = slice_bytes
    if slice_bytes != want_bytes:
        problems.append('qwen_block_bytes is %d, a 64 x 256 bfloat8_b block per head is %d' % (slice_bytes, want_bytes))
        report['status'] = 'FAIL'
        return report, problems
    second = ttnn.as_tensor(torch.zeros(blocks, heads_per_chip, BLOCK, HEAD_DIM, dtype=torch.bfloat16), device=mesh,
                            dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    programs = mesh.num_program_cache_entries()
    sets = block_sets(blocks)
    touched = {index for selected in sets.values() for index in selected}
    outside = [index for index in (3, blocks // 3, blocks - 3) if index not in touched]
    before = np.empty((devices, len(outside), slice_bytes), dtype=np.uint8)
    if outside:
        ttnn.qwen_read_blocks_raw(second, before, outside)
    for name, selected in sets.items():
        count = len(selected)
        out = np.empty((devices, count, slice_bytes), dtype=np.uint8)
        ttnn.qwen_read_blocks_raw(cache, out, selected)
        ttnn.qwen_write_blocks_raw(second, out, selected)
        back = np.empty_like(out)
        ttnn.qwen_read_blocks_raw(second, back, selected)
        same_bytes = bool(np.array_equal(out, back))
        # The second cache, unpacked by ttnn as the model's readers unpack it, must hold the first cache's values at those blocks.
        cache_dims = [int(cache.shape[i]) for i in range(len(cache.shape))]
        pair = []
        for tensor in (cache, second):
            host = ttnn.allocate_tensor_on_host(ttnn.Shape([count] + cache_dims[1:]), tensor.dtype, tensor.layout, mesh)
            ttnn.qwen_read_blocks(tensor, host, selected)
            pair.append(ttnn.to_torch(host, mesh_composer=composer))
        same_values = bool((pair[0].contiguous().view(torch.uint8) == pair[1].contiguous().view(torch.uint8)).all())
        # The teeth: one changed byte must not read back as the original.
        changed = out.copy()
        changed.reshape(-1)[0] ^= 0xFF
        ttnn.qwen_write_blocks_raw(second, changed, selected)
        again = np.empty_like(out)
        ttnn.qwen_read_blocks_raw(second, again, selected)
        teeth = not bool(np.array_equal(again, out))
        ttnn.qwen_write_blocks_raw(second, out, selected)
        report[name] = dict(blocks=count, bytes_equal=same_bytes, values_equal=same_values, teeth=teeth)
        for field, ok in (('raw bytes written and read back differ', same_bytes), ('the raw-written cache unpacks to other values', same_values),
                          ('a changed byte read back as the original (the comparison has no teeth)', teeth)):
            if not ok:
                problems.append('%s: %s' % (name, field))
    # A write touches only the blocks it names: blocks outside every set read back as they were before the first write.
    if outside:
        after = np.empty_like(before)
        ttnn.qwen_read_blocks_raw(second, after, outside)
        report['outside_checked'] = outside
        if not bool(np.array_equal(before, after)):
            problems.append('a raw write changed blocks it did not name (%s)' % outside)
    report['program_cache_growth'] = mesh.num_program_cache_entries() - programs
    if report['program_cache_growth']:
        problems.append('the raw ops grew the program cache by %d' % report['program_cache_growth'])
    rates = {}
    for size in RAW_SIZES:
        if size > blocks:
            continue
        ids = list(range(size))
        buffer = np.empty((devices, size, slice_bytes), dtype=np.uint8)
        begin = time.perf_counter()
        ttnn.qwen_read_blocks_raw(cache, buffer, ids)
        read_s = time.perf_counter() - begin
        begin = time.perf_counter()
        ttnn.qwen_write_blocks_raw(second, buffer, ids)
        write_s = time.perf_counter() - begin
        moved = buffer.nbytes
        rates[str(size)] = dict(bytes=moved, read_ms=round(read_s * 1000, 2), write_ms=round(write_s * 1000, 2),
                                read_gbps=round(moved / read_s / 1e9, 3), write_gbps=round(moved / write_s / 1e9, 3))
    report['rates'] = rates
    ttnn.deallocate(second)
    report['status'] = 'ok' if not problems else 'FAIL'
    return report, problems


def run(ttnn, mesh, devices, blocks, heads_per_chip):
    """The checks on an open mesh -> the report dict (ok, problems, one entry per block set, the cost figures)."""
    problems, report = [], {}
    heads = heads_per_chip * devices
    torch.manual_seed(1)
    values = (torch.randn(blocks, heads, BLOCK, HEAD_DIM) * 3).to(torch.bfloat16)
    cache = ttnn.as_tensor(torch.zeros(blocks, heads_per_chip, BLOCK, HEAD_DIM, dtype=torch.bfloat16),
                           device=mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    source = ttnn.from_torch(values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                             mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(source, cache)
    ttnn.deallocate(source)
    del values
    composer = ttnn.ConcatMeshToTensor(mesh, dim=1)
    want_shape = (blocks, heads, BLOCK, HEAD_DIM)
    begin = time.perf_counter()
    packed = ttnn.from_device(cache)
    report['whole_device_read_s'] = round(time.perf_counter() - begin, 3)
    begin = time.perf_counter()
    whole = ttnn.to_torch(packed, mesh_composer=composer)
    report['whole_unpack_s'] = round(time.perf_counter() - begin, 3)
    del packed
    report['whole_read_s'] = round(report['whole_device_read_s'] + report['whole_unpack_s'], 3)
    if tuple(whole.shape) != want_shape:
        problems.append('the whole-cache read has shape %s, not %s: the cache was not allocated like production' % (
            tuple(whole.shape), want_shape))
    report['whole_bytes_fp32'] = whole.numel() * whole.element_size()
    programs = mesh.num_program_cache_entries()
    sets = block_sets(blocks)
    for name, selected in sets.items():
        # ttnn.Shape supports only __getitem__(int), __len__ and __iter__: no slice (KVQ v560 died on cache.shape[1:]).
        cache_dims = [int(cache.shape[i]) for i in range(len(cache.shape))]
        host = ttnn.allocate_tensor_on_host(ttnn.Shape([len(selected)] + cache_dims[1:]),
                                            cache.dtype, cache.layout, mesh)
        begin = time.perf_counter()
        ttnn.qwen_read_blocks(cache, host, selected)
        read_s = time.perf_counter() - begin
        begin = time.perf_counter()
        got = ttnn.to_torch(host, mesh_composer=composer)
        unpack_s = time.perf_counter() - begin
        want = whole.index_select(0, torch.as_tensor(selected, dtype=torch.long))
        if tuple(got.shape) != (len(selected),) + want_shape[1:]:
            problems.append('%s: the region read unpacked to %s, not %s (a chip shard was lost)' % (
                name, tuple(got.shape), (len(selected),) + want_shape[1:]))
        same = got.shape == want.shape and bool((got.contiguous().view(torch.uint8) == want.contiguous().view(torch.uint8)).all())
        report[name] = dict(blocks=len(selected), read_ms=round(read_s * 1000, 2), unpack_ms=round(unpack_s * 1000, 2), equal=same)
        if not same:
            problems.append('%s: the region read differs from the whole-cache read' % name)
    report['program_cache_growth'] = mesh.num_program_cache_entries() - programs
    if report['program_cache_growth']:
        problems.append('the region read grew the program cache by %d' % report['program_cache_growth'])
    one = report.get('one')
    biggest = max((report[name] for name in sets), key=lambda entry: entry['blocks'])
    # The volume rules are timing rules: they mean something only when the whole-cache device read is long enough to time
    # (the production pool's is seconds; a toy pool's is not), and the report says whether they were applied.
    report['volume_checked'] = report['whole_device_read_s'] >= MIN_TIMED_WHOLE_READ_S
    if report['volume_checked']:
        if one and one['read_ms'] / 1000.0 > SCALING_FACTOR * report['whole_device_read_s'] + 0.05:
            problems.append('one block took %.1f ms against a whole-cache device read of %.3f s: the read does not follow the row' % (
                one['read_ms'], report['whole_device_read_s']))
        if biggest['read_ms'] / 1000.0 > report['whole_device_read_s']:
            problems.append('the largest set (%d blocks) took longer than the whole-cache device read' % biggest['blocks'])
    report['read_ms_per_block_largest'] = round(biggest['read_ms'] / biggest['blocks'], 4)
    raw, raw_problems = run_raw(ttnn, mesh, devices, cache, blocks, heads_per_chip, composer)
    report['raw'] = raw
    problems.extend('raw: ' + problem for problem in raw_problems)
    report['ok'] = not problems
    report['problems'] = problems
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=4096)
    parser.add_argument('--heads-per-chip', type=int, default=1)
    options = parser.parse_args(argv)

    import ttnn

    problem, _ = ensure_extension(ttnn)
    if problem:
        print(json.dumps({'ok': False, 'problem': problem + ': the graft is not in the build'}))
        return 1
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, options.devices))
    try:
        report = run(ttnn, mesh, options.devices, options.blocks, options.heads_per_chip)
    finally:
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
