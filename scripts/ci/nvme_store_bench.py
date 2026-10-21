"""A micro-benchmark of the NVMe cold tier's store on a file or a raw namespace: what the tier's spill and restore would get from the drive, without an engine or a card.

    python3 nvme_store_bench.py PATH [--gib 8] [--block-bytes 2228224] [--batch-blocks 256] [--threads 4] [--verify sample|all|off]

It writes `--gib` GiB of KV-block-sized objects through NvmeStore.put_batch (the spill's path: hashing, aligned writes, the journal), reads them back in session-sized
runs through read_many (the restore's path: merged runs, reader threads, the digest check), and reads a random half in shuffled order (a restore of blocks the allocator
scattered). It prints GB/s for each and the store's counters.

SAFETY. A regular file is created or reused (sparse, `--gib` plus the metadata). A block device is REFUSED unless both --format-force and --yes-destroy-everything are given:
the run formats it, and everything on it is gone. Run it on the namespace under test only, with the tier's engine stopped (the store takes an exclusive lock).

Stdlib only.
"""

import argparse
import hashlib
import mmap
import os
import random
import stat
import sys
import time

import qwen_nvme_store as store_module
from qwen_nvme_store import ALIGN, KV, NvmeStore, PutItem

BLOCK_BYTES = 2228224          # the production KV block: 32 cache tensors x 4 chips x 17,408 bytes (544 pages)


def make_items(start, count, block_bytes, arena):
    """`count` items whose bytes are the arena's (one aligned buffer, shared: the cost measured is the store's, not the generator's) under distinct keys."""
    fp = b'benchfp0'
    items = []
    for index in range(start, start + count):
        key = hashlib.sha256(b'bench%d' % index).digest()
        items.append(PutItem(KV, key, [arena[:block_bytes]], 'bench', b'bench-chain', index, fp))
    return items


def run(path, gib=8.0, block_bytes=BLOCK_BYTES, batch_blocks=256, threads=4, verify='sample', format_policy='auto', sample_every=16, out=print, rng=None):
    """-> dict of the measured rates and counters. Raises on a refused path."""
    rng = rng or random.Random(1)
    block_bytes = store_module.roundup(block_bytes)
    total_blocks = max(1, int(gib * (1 << 30)) // block_bytes)
    size = None
    if not os.path.exists(path):
        size = int(total_blocks * block_bytes * 1.1) + (64 << 20)
    elif not stat.S_ISBLK(os.stat(path).st_mode) and os.path.getsize(path) < total_blocks * block_bytes * 1.05:
        size = int(total_blocks * block_bytes * 1.1) + (64 << 20)
    store = NvmeStore.open(path, format=format_policy, size=size, threads=threads, free_floor_bytes=0)
    arena_map = mmap.mmap(-1, store_module.roundup(block_bytes))
    arena = memoryview(arena_map)
    arena[:] = os.urandom(len(arena))
    result = dict(path=path, blocks=total_blocks, block_bytes=block_bytes, threads=threads, verify=verify)
    try:
        began = time.perf_counter()
        for low in range(0, total_blocks, batch_blocks):
            count = min(batch_blocks, total_blocks - low)
            stored = store.put_batch(make_items(low, count, block_bytes, arena))
            if not all(stored):
                raise RuntimeError('the store refused a batch at block %d: the device is too small for %d blocks' % (low, total_blocks))
        written = time.perf_counter() - began
        result['write_gbps'] = total_blocks * block_bytes / written / 1e9
        result['write_s'] = written
        out('write   %7.2f GB/s  (%d blocks of %d bytes in %.2f s: hashing, aligned writes, the journal)' % (result['write_gbps'], total_blocks, block_bytes, written))

        def read(order, label):
            entries = store.pin([(KV, hashlib.sha256(b'bench%d' % index).digest()) for index in order])
            views = [memoryview(bytearray(block_bytes)) for _ in entries]
            chosen = set(id(entry) for position, entry in enumerate(entries) if verify == 'all' or (verify == 'sample' and position % sample_every == 0))

            def check(entry, view):
                return hashlib.sha256(view).digest() == entry.digest if id(entry) in chosen else True

            began = time.perf_counter()
            results = store.read_many(entries, views, verify=check if chosen else None)
            took = time.perf_counter() - began
            store.unpin(entries)
            bad = sum(1 for item in results if item is not None)
            rate = len(entries) * block_bytes / took / 1e9
            out('%-8s%7.2f GB/s  (%d blocks in %.2f s, %d failed, verify %s)' % (label, rate, len(entries), took, bad, verify))
            return rate, bad

        # (the read buffers are heap memory, as the heap-mode fallback's are; the production restore reads into aligned slab slots, which skips a bounce)
        session = min(total_blocks, 3968)             # a 254k-token session
        result['read_sequential_gbps'], bad = read(range(session), 'seq')
        order = list(range(total_blocks))
        rng.shuffle(order)
        result['read_shuffled_gbps'], bad2 = read(order[:min(total_blocks // 2 + 1, 4096)], 'shuffled')
        result['read_failures'] = bad + bad2
        result.update(store.snapshot_stats())
    finally:
        arena.release()
        arena_map.close()
        store.close()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('path')
    parser.add_argument('--gib', type=float, default=8.0)
    parser.add_argument('--block-bytes', type=int, default=BLOCK_BYTES)
    parser.add_argument('--batch-blocks', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--verify', choices=('sample', 'all', 'off'), default='sample')
    parser.add_argument('--format-force', action='store_true', help='initialize the device whatever it holds (a block device needs this and --yes-destroy-everything)')
    parser.add_argument('--yes-destroy-everything', action='store_true')
    options = parser.parse_args(argv)
    if os.path.exists(options.path) and stat.S_ISBLK(os.stat(options.path).st_mode):
        if not (options.format_force and options.yes_destroy_everything):
            parser.error('%s is a block device: this run formats it and everything on it is lost; pass --format-force and --yes-destroy-everything if that is the point' % options.path)
    run(options.path, options.gib, options.block_bytes, options.batch_blocks, options.threads, options.verify, 'force' if options.format_force else 'auto')
    return 0


if __name__ == '__main__':
    sys.exit(main())
