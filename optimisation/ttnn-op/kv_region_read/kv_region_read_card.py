"""Card check for ttnn.qwen_read_blocks (optimisation/sim/kv-region-read.patch): the region read of a paged KV
cache against the whole-cache read, byte for byte, on a mesh, and what it costs.

NOT RUN YET. The patch is a draft against the pinned tt-metal tree; it has been neither compiled nor
executed. Run this once on a graft build, in a card window (never beside production):

    python3 kv_region_read_card.py --devices 4 --blocks 4096 --heads-per-chip 1

Prints one JSON object and exits 1 on any mismatch. What it settles (the questions the patch leaves open):
  * the host tensor ttnn.allocate_tensor_on_host(shape=[n, heads_per_chip, 64, 256]) really is the per-device shard
    shape qwen_read_blocks expects (the read refuses a mismatch with a TT_FATAL, so a wrong guess fails loud);
  * several region reads of one shard in a single enqueue_read_shards land where the host offsets say;
  * the unpacked values equal the whole-cache read's, for runs, singletons and a shuffled order;
  * the program cache does not grow (the read compiles nothing) - the property the audit exists to keep;
  * the cost per block and per read call, which the W-1 minutes are based on (docs/prefix-audit-cost.md).
"""

import argparse
import json
import sys
import time

import torch

BLOCK = 64
HEAD_DIM = 256


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=4096)
    parser.add_argument('--heads-per-chip', type=int, default=1)
    options = parser.parse_args(argv)

    import ttnn

    if not hasattr(ttnn, 'qwen_read_blocks'):
        print(json.dumps({'ok': False, 'problem': 'this ttnn has no qwen_read_blocks: the graft is not in the build'}))
        return 1
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, options.devices))
    problems, report = [], {}
    try:
        heads = options.heads_per_chip * options.devices
        torch.manual_seed(1)
        values = (torch.randn(options.blocks, heads, BLOCK, HEAD_DIM) * 3).to(torch.bfloat16)
        cache = ttnn.from_torch(values, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        composer = ttnn.ConcatMeshToTensor(mesh, dim=1)
        begin = time.perf_counter()
        whole = ttnn.to_torch(cache, mesh_composer=composer)
        report['whole_read_s'] = round(time.perf_counter() - begin, 3)
        report['whole_bytes_fp32'] = whole.numel() * whole.element_size()
        programs = mesh.num_program_cache_entries()
        for name, blocks in block_sets(options.blocks).items():
            host = ttnn.allocate_tensor_on_host(ttnn.Shape([len(blocks)] + [int(d) for d in cache.shape[1:]]),
                                                cache.dtype, cache.layout, mesh)
            begin = time.perf_counter()
            ttnn.qwen_read_blocks(cache, host, blocks)
            read_s = time.perf_counter() - begin
            begin = time.perf_counter()
            got = ttnn.to_torch(host, mesh_composer=composer)
            unpack_s = time.perf_counter() - begin
            want = whole.index_select(0, torch.as_tensor(blocks, dtype=torch.long))
            same = got.shape == want.shape and bool((got.contiguous().view(torch.uint8) == want.contiguous().view(torch.uint8)).all())
            report[name] = dict(blocks=len(blocks), read_ms=round(read_s * 1000, 2), unpack_ms=round(unpack_s * 1000, 2), equal=same)
            if not same:
                problems.append('%s: the region read differs from the whole-cache read' % name)
        report['program_cache_growth'] = mesh.num_program_cache_entries() - programs
        if report['program_cache_growth']:
            problems.append('the region read grew the program cache by %d' % report['program_cache_growth'])
    finally:
        ttnn.close_mesh_device(mesh)
    report['ok'] = not problems
    report['problems'] = problems
    print(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
