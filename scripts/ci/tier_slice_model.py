"""The sliced restore's cost model (stdlib only; docs/prefix-store-hygiene.md "Sliced restore").

    python3 scripts/ci/tier_slice_model.py            the table for the measured card figures below
    python3 scripts/ci/tier_slice_model.py --cpu      also times this host's digest and compare of one block (the part of a slice that is CPU work)

A restore of N blocks costs the engine W = N * c milliseconds of work (c: the cost per block of the digest check, the allocation and spill, the raw write and, in an audited arm, the
read-back and compare). Whole, in the trim, that is one stall of W for every seat. Sliced with a budget of S milliseconds a step (and a block cap K), the engine does the same work in
ceil(W / S) steps (the first of them a probe of SLICE_PROBE_BLOCKS blocks, then as many as S allows at the measured c, at most K), and between two steps the seats get a round of R
milliseconds. So:

  a seat's longest wait caused by the restore    max slice + R      (the slice is S, or the probe's cost before c is known)
  the returning request's extra time to first token   (slices - 1) * R     (a round between slices; the work W itself is paid either way)
  what slicing cannot touch                      the request's own tail prefill and engine build (T3: a device-resident hit at 254k took 4.65 s to its first token, nothing else running)

The figures are MEASURED (docs/prefix-store-hygiene.md and the TC1 run it cites); the table is an ESTIMATE of what the flag does with them. Nothing here has run on a card.
"""

import argparse
import hashlib
import sys
import time

PROBE_BLOCKS = 8
SPILL_MAX_BLOCKS = 512         # QWEN_PREFIX_HOST_TIER_SPILL_MAX_BLOCKS: the most one step's evictions may spill; the whole restore evicts all its blocks in one step and the rest are dropped
BLOCK_BYTES = 2228224          # the 2.23 MB attention-KV page of one 64-token block (T3: 8.56 GB / 3,841 blocks)

# Measured on the four-card mesh. Per block, in milliseconds.
RATES = {
    # T3, audits off, verify=sample (run 38084230415): 3,841 blocks, 8.56 GB restored in 961 ms (8.9 GB/s), the spill of the 512 displaced blocks it kept (1.14 GB) 142 ms.
    'unaudited': dict(per_block_ms=961.0 / 3841, spill_ms_per_block=142.0 / 512, round_ms=150.0, tail_s=4.65,
                      note='T3 (audits off, verify=sample): the restore 8.9 GB/s; a seat round about 150 ms (the fusion ship profile at 8 live); the hit alone 4.65 s to its first token'),
    # TC1 on the audited ship-4 twin (run 38133584894): 3,905 blocks; digests of every block 3.63 s before the restore began; the restore's 6,321 ms of which 888 ms the spill of 512
    # displaced blocks (the cap: the 3,393 others were dropped); a seat round 493 ms at the median.
    'audited': dict(per_block_ms=(3628.0 + 6321.0 - 888.0) / 3905, spill_ms_per_block=888.0 / 512, round_ms=493.0, tail_s=4.8,
                    note='TC1 (the audited twin: every digest, read-back and compare): digests 0.93 ms and the write plus read-back 1.39 ms a block, the spill 1.73 ms a block, a seat round 493 ms'),
}


def sizes(blocks, per_block_ms, slice_ms, cap):
    """The slice sizes the governor chooses when the cost per block is `per_block_ms` from the first slice on (a probe of PROBE_BLOCKS first)."""
    out, left = [], blocks
    while left > 0:
        count = PROBE_BLOCKS if not out else int(slice_ms / per_block_ms)
        count = max(1, min(cap, count, left))
        out.append(count)
        left -= count
    return out


def row(blocks, rates, slice_ms, cap):
    """-> dict: the slice sizes' summary and the two numbers that matter."""
    per_block = rates['per_block_ms'] + rates['spill_ms_per_block']          # sliced: each slice's allocation evicts, and spills, as many blocks as it restores
    plan = sizes(blocks, per_block, slice_ms, cap)
    work_ms = blocks * per_block
    whole_ms = blocks * rates['per_block_ms'] + min(blocks, SPILL_MAX_BLOCKS) * rates['spill_ms_per_block']      # whole: one allocation, one capped spill
    longest_ms = max(plan) * per_block
    return dict(blocks=blocks, slices=len(plan), first=plan[0], steady=plan[1] if len(plan) > 1 else plan[0], work_s=work_ms / 1000.0, whole_stall_s=whole_ms / 1000.0 + rates['round_ms'] / 1000.0,
                longest_slice_ms=longest_ms, seat_wait_s=(longest_ms + rates['round_ms']) / 1000.0, ttft_extra_s=(len(plan) - 1) * rates['round_ms'] / 1000.0)


def table(blocks_list=(1985, 3905), slice_ms_list=(100.0, 200.0, 400.0), cap=512):
    lines = []
    for name in ('unaudited', 'audited'):
        rates = RATES[name]
        lines.append('%s: %s' % (name, rates['note']))
        lines.append('  blocks  slice_ms  slices  first/steady  sliced work (s)  whole: seats wait (s)  sliced: seats wait (s)  returning TTFT extra (s)')
        for blocks in blocks_list:
            for slice_ms in slice_ms_list:
                r = row(blocks, rates, slice_ms, cap)
                lines.append('  %6d  %8.0f  %6d  %5d/%-6d  %16.2f  %21.2f  %22.2f  %24.2f' % (
                    blocks, slice_ms, r['slices'], r['first'], r['steady'], r['work_s'], r['whole_stall_s'], r['seat_wait_s'], r['ttft_extra_s']))
        lines.append('  and what the flag cannot slice: the returning request\'s tail prefill and engine build, about %.2f s here (the seats\' wait whatever the restore does)' % rates['tail_s'])
    return lines


def cpu_cost(repeat=3):
    """(digest ms, compare ms) of one 2.23 MB block on this host: sha256 of the block, and the audit's bytes(a) != bytes(b) over two copies."""
    block = bytes(range(256)) * (BLOCK_BYTES // 256)
    view = memoryview(bytearray(block))
    began = time.perf_counter()
    for _ in range(repeat * 20):
        hashlib.sha256(view).hexdigest()
    digest = (time.perf_counter() - began) * 1000.0 / (repeat * 20)
    began = time.perf_counter()
    for _ in range(repeat * 20):
        bytes(view) != bytes(view)
    compare = (time.perf_counter() - began) * 1000.0 / (repeat * 20)
    return digest, compare


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cpu', action='store_true', help='time this host\'s digest and compare of one block')
    options = parser.parse_args(argv)
    for line in table():
        out(line)
    if options.cpu:
        digest, compare = cpu_cost()
        out('this host, one %.2f MB block: sha256 %.3f ms (%.2f GB/s), audit compare (two copies and a memcmp) %.3f ms' % (
            BLOCK_BYTES / 1e6, digest, BLOCK_BYTES / digest / 1e6, compare))
    return 0


if __name__ == '__main__':
    sys.exit(main())
