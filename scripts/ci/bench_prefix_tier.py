"""CPU-side costs of the host KV tier at the production block geometry (stdlib and numpy; no device, no vLLM).

    python3 scripts/ci/bench_prefix_tier.py [--blocks 512]

One block of the paged KV pool is 32 cache tensors x 4 chips x 17,408 bytes = 2,228,224 bytes. Every device call is a no-op here, so what is timed is the
host's own work: the model graft's adapter (qwen_prefix_model_patch._QwenKvTierIO) staging the blocks into and out of host memory, the store's puts, gets,
evictions and digests. The device's time is NOT in these numbers (the Q1 job of references/tp4-prefix-tiers-jobs prints it); the host's rates move with its
load and memory pressure, so run it twice. 512 blocks is 1.14 GB per buffer; 2,048 is one 128k session and 3,968 one 254k session."""

import argparse
import ast
import hashlib
import os
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import qwen_prefix_model_patch as patcher  # noqa: E402
import qwen_prefix_registry as registry  # noqa: E402

LAYERS, SLICE_BYTES, CHIPS = 16, 17408, 4


def adapter_class():
    """The model graft's _QwenKvTierIO, built from its source in the patch (the model file is a string there) over a ttnn whose raw ops do nothing."""
    import numpy  # noqa: F401  (the adapter imports it lazily; fail here, with the name)
    source = patcher.MODEL_ADAPTER
    tree = ast.parse(source)
    segment = [ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_QwenKvTierIO'][0]
    fake = SimpleNamespace(qwen_block_bytes=lambda cache: SLICE_BYTES, qwen_read_blocks_raw=lambda cache, out, ids: None,
                           qwen_write_blocks_raw=lambda cache, data, ids: None)
    namespace = {'ttnn': fake, 'os': os}
    exec(segment, namespace)
    cache = SimpleNamespace(shape=(19968, 1, 64, 256), dtype='bfloat8_b')
    model = SimpleNamespace(_paged_kv_caches=[(cache, cache) for _ in range(LAYERS)], num_devices=CHIPS)
    return namespace['_QwenKvTierIO'](model)


def rate(size, seconds):
    return size / max(seconds, 1e-9) / 1e9


def run(blocks, out=print):
    io = adapter_class()
    ids = list(range(blocks))
    size = blocks * io.block_bytes
    out('block_bytes %d tensors %d blocks %d (%.2f GB)' % (io.block_bytes, len(io.caches), blocks, size / 1e9))
    began = time.perf_counter()
    payloads = io.read_blocks(ids)
    out('adapter read, fresh arrays: %.3f s  %.2f GB/s' % (time.perf_counter() - began, rate(size, time.perf_counter() - began)))
    del payloads
    slab = registry.TierStore(size, digest_inline=True)
    slab.configure(io.block_bytes)
    slots = slab.reserve(blocks)
    views = [slot.view for slot in slots]
    began = time.perf_counter()
    io.read_blocks(ids, into=views)
    first = time.perf_counter() - began
    out('adapter read, into slab (first fill, pages not yet resident): %.3f s  %.2f GB/s' % (first, rate(size, first)))
    began = time.perf_counter()
    io.read_blocks(ids, into=views)
    warm = time.perf_counter() - began
    out('adapter read, into slab (slots resident): %.3f s  %.2f GB/s' % (warm, rate(size, warm)))
    began = time.perf_counter()
    io.write_blocks(ids, views)
    written = time.perf_counter() - began
    out('adapter write from slab views: %.3f s  %.2f GB/s' % (written, rate(size, written)))
    slab.abort(slots)
    store = registry.TierStore(size * 2, digest_inline=False)
    began = time.perf_counter()
    for index, view in enumerate(views):
        store.put(b'k%07d' % index, view, 'tenant', b'chain', index)
    put = time.perf_counter() - began
    out('store put %d blocks: %.4f s (%.1f us/block, no copy; digests queued)' % (blocks, put, put / blocks * 1e6))
    began = time.perf_counter()
    while any(record.digest is None for record in list(store.records.values())):
        time.sleep(0.005)
    digest = time.perf_counter() - began
    out('digest worker finished %.3f s after the last put (one thread, off the calling thread): %.2f GB/s' % (digest, rate(size, digest)))
    began = time.perf_counter()
    hashlib.sha256(views[0]).hexdigest()
    one = time.perf_counter() - began
    out('sha256 of one block: %.3f ms (%.2f GB/s)' % (one * 1000, rate(io.block_bytes, one)))
    began = time.perf_counter()
    for key in list(store.records):
        store.get(key)
    out('store get %d blocks: %.1f us/block' % (blocks, (time.perf_counter() - began) / blocks * 1e6))
    began = time.perf_counter()
    for record in list(store.records.values()):
        store.verify(record)
    verify = time.perf_counter() - began
    out('verify (sha256 + compare) %d blocks: %.3f s = %.2f GB/s' % (blocks, verify, rate(size, verify)))
    small = registry.TierStore(10 * io.block_bytes)
    began = time.perf_counter()
    for index, view in enumerate(views):
        small.put(b'e%07d' % index, view, 'tenant', b'chain', index)
    out('store put with eviction %d blocks into a 10-block store: %.1f us/block' % (blocks, (time.perf_counter() - began) / blocks * 1e6))
    store.close()
    small.close()
    slab.close()
    return dict(block_bytes=io.block_bytes, read_resident_gbps=rate(size, warm), write_gbps=rate(size, written))


def paced_adapter(device_write_gbps, device_read_gbps):
    """The adapter over raw ops that take the time a device at those rates would (a sleep of bytes / rate: the host is free meanwhile, as it is during a real call, which
    releases the GIL). Rates of 0 make the ops free. -> (adapter, ops (read_raw, write_raw) for the threaded wrapper)."""
    import numpy

    def pace(array, rate):
        if rate:
            time.sleep(array.nbytes / (rate * 1e9))

    def read_raw(cache, out, ids):
        pace(out, device_read_gbps)

    def write_raw(cache, data, ids):
        pace(data, device_write_gbps)

    source = patcher.MODEL_ADAPTER
    tree = ast.parse(source)
    segment = [ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_QwenKvTierIO'][0]
    fake = SimpleNamespace(qwen_block_bytes=lambda cache: SLICE_BYTES, qwen_read_blocks_raw=read_raw, qwen_write_blocks_raw=write_raw)
    namespace = {'ttnn': fake, 'os': os}
    exec(segment, namespace)
    cache = SimpleNamespace(shape=(19968, 1, 64, 256), dtype='bfloat8_b')
    model = SimpleNamespace(_paged_kv_caches=[(cache, cache) for _ in range(LAYERS)], num_devices=CHIPS)
    del numpy
    return namespace['_QwenKvTierIO'](model), (read_raw, write_raw)


def run_threads(blocks, threads, write_gbps, read_gbps, chunk=0, out=print):
    """The adapter's read_blocks / write_blocks against qwen_tier_upload.ThreadedTierIO on the same paced device. The staging copies are real (numpy, this host); the device
    time is SIMULATED at the rates given (Q1's measured 22.8 GB/s write and 13.6 GB/s read by default), so the table is the host side of the overlap, not a card number."""
    import qwen_tier_upload as upload

    io, ops = paced_adapter(write_gbps, read_gbps)
    wrapped = upload.ThreadedTierIO(io, threads, chunk, ops=ops)
    ids = list(range(blocks))
    size = blocks * io.block_bytes
    out('threads %d, call size %d blocks, device simulated at write %.1f / read %.1f GB/s, %d blocks (%.2f GB)' % (threads, chunk or io.CHUNK_BLOCKS, write_gbps, read_gbps, blocks, size / 1e9))
    slab = registry.TierStore(size, digest_inline=True)
    slab.configure(io.block_bytes)
    slots = slab.reserve(blocks)
    views = [slot.view for slot in slots]
    result = {}
    try:
        for name, call in (('read into the slab', lambda target: target.read_blocks(ids, into=views)), ('write from the slab', lambda target: target.write_blocks(ids, views))):
            call(io)                # touch every page once: the slab's first fill is page-fault bound and is not what is compared
            rows = []
            for label, target in (('adapter ', io), ('threaded', wrapped)):
                best = None
                for _ in range(3):
                    began = time.perf_counter()
                    call(target)
                    took = time.perf_counter() - began
                    best = took if best is None else min(best, took)
                rows.append(best)
                out('  %-20s %s: %.3f s  %.2f GB/s' % (name, label, best, rate(size, best)))
            out('  %-20s speedup x%.2f' % (name, rows[0] / rows[1]))
            result[name] = rows
    finally:
        wrapped.close()
        slab.abort(slots)
        slab.close()
    return result


def run_verify_threads(blocks, threads, out=print):
    """sha256 of `blocks` blocks sequentially against qwen_tier_upload.Verifier on `threads` workers (the restore's digest check at verify=all; sample is one block in 16)."""
    import qwen_tier_upload as upload

    io = adapter_class()
    store = registry.TierStore(blocks * io.block_bytes * 2, digest_inline=True)
    store.configure(io.block_bytes)
    slots = store.reserve(blocks)
    for index, slot in enumerate(slots):
        slot.view[:64] = bytes([index % 251]) * 64
        store.put(b'v%07d' % index, slot.view, 'tenant', b'chain', index)
    records = [store.get(b'v%07d' % index) for index in range(blocks)]
    verifier = upload.Verifier(threads)
    try:
        began = time.perf_counter()
        sequential = [store.verify(record) for record in records]
        seq = time.perf_counter() - began
        began = time.perf_counter()
        parallel = verifier.verdicts(store, records)
        par = time.perf_counter() - began
    finally:
        verifier.close()
        store.close()
    out('digest check of %d blocks: sequential %.3f s (%.2f GB/s), %d threads %.3f s (%.2f GB/s), x%.2f, verdicts %s' % (
        blocks, seq, rate(blocks * io.block_bytes, seq), threads, par, rate(blocks * io.block_bytes, par), seq / max(par, 1e-9),
        'equal' if sequential == parallel else 'DIFFER'))
    return seq, par


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--blocks', type=int, default=512)
    parser.add_argument('--threads', type=int, default=0, help='also compare the adapter with the threaded wrapper (qwen_tier_upload) on this many staging threads, the device simulated')
    parser.add_argument('--write-gbps', type=float, default=22.8, help='the simulated device write rate (Q1: 22.8)')
    parser.add_argument('--read-gbps', type=float, default=13.6, help='the simulated device read rate (Q1: 13.6)')
    parser.add_argument('--chunk', type=int, default=0, help='blocks per device call for the threaded wrapper (0: the adapter\'s 512)')
    options = parser.parse_args(argv)
    if options.blocks < 1:
        parser.error('--blocks must be at least 1')
    if options.threads:
        run_threads(options.blocks, options.threads, options.write_gbps, options.read_gbps, options.chunk)
        run_verify_threads(min(options.blocks, 256), options.threads)
        return 0
    run(options.blocks)
    return 0


if __name__ == '__main__':
    sys.exit(main())
