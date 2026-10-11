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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--blocks', type=int, default=512)
    options = parser.parse_args(argv)
    if options.blocks < 1:
        parser.error('--blocks must be at least 1')
    run(options.blocks)
    return 0


if __name__ == '__main__':
    sys.exit(main())
