"""What the three K-split kernels (reader, writer, compute) and the host planner share: the device-side integer arithmetic of the slices, as C++ text, and the
compile-time / runtime argument layout. One source: make_ks_reader.py, make_ks_writer.py and make_ks_compute.py paste RANGE_BLOCK byte for byte, and
test_ksplit_sources.py holds the three copies equal and equal to ksplit_planner's Python functions (it executes the C++ with g++ on a grid of inputs).
"""

NL = chr(10)


def lines(*parts):
    return ''.join(part + NL for part in parts)


MIN_BLOCKS_PER_SLICE = 4

# The slice arithmetic. A q chunk streams B K blocks (B = ceil(q_high / Sk_chunk_t) = chunk_start / 128 + q + 1); E = max(1, min(S, B / 4)) slices take part; slice s takes
# [s * B / E, (s + 1) * B / E), so the ranges are contiguous, disjoint, cover [0, B) and the last one holds the diagonal block B - 1. A slice s >= E is idle. Each kernel
# computes this from the chunk start it reads from the same device tensor, so no host value can disagree with a kernel.
RANGE_BLOCK = lines(
    '// [QWEN-SDPA-KS] the slices of one q chunk\'s K stream: the same integer arithmetic in the reader, the writer and the compute kernel (ksplit_planner.py holds the Python twin).',
    'namespace qwen_ks {',
    'constexpr uint32_t kMinBlocksPerSlice = %d;' % MIN_BLOCKS_PER_SLICE,
    'FORCE_INLINE uint32_t effective_slices(uint32_t blocks, uint32_t slices) {',
    '    const uint32_t by_blocks = blocks / kMinBlocksPerSlice;',
    '    const uint32_t taking_part = by_blocks < slices ? by_blocks : slices;',
    '    return taking_part > 0 ? taking_part : 1;',
    '}',
    'FORCE_INLINE uint32_t slice_begin(uint32_t blocks, uint32_t slice, uint32_t eff) { return (slice * blocks) / eff; }',
    '}  // namespace qwen_ks')

# The larger-than-any-K constants of the RING instantiation (compute kernel): kv_global_tile_for_local<true, 0, N, N> is the identity for K tile indices below N.
KV_IDENTITY_TILES = 1 << 20
KV_NEVER_TILES = 0xFFFFFFFF
