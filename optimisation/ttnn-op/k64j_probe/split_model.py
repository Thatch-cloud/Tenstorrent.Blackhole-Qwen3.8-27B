"""A pure-Python mirror of how the decode SDPA splits one user's keys over the cores of a KV head.

WHY. K64j (c2-serve-for-real-plan.md section 2.3) wants the served non-causal replay call to take its extent at
RUN time (a cur_pos word) instead of baking it into the program (St * 32 - 1). Risk R1 asks whether the runtime
position reproduces the compile-time split exactly. The split is three functions of tt-metal 9f9cd4fd, copied
here line for line so a CPU test can enumerate every family and core count:

  rt_args_common.hpp (sha256 1b52c60d...)   nearest_n, nearest_pow_of_2_up_to_8, get_workload_for_core,
                                            get_dynamic_Sk_chunk_t (the reader, the writer and the compute
                                            kernel all call these three with their own cur_pos)
  sdpa_decode_device_operation.hpp          get_tree_reduction_params (host side; the writer and the compute
                                            kernel then drop the children whose core has no chunk)
  sdpa_decode_program_factory.cpp 3e0a69af  the cores per head (:195-206) and max_dynamic_chunk_size (:388-389)

WHAT IT SHOWS (test_k64j_probe.SplitModelTests):
  - with a fixed chunk (k_chunk_size 256, every served replay config: attention_replay.py:53-54,
    pooled_attention_replay.py:199-200), get_dynamic_Sk_chunk_t returns the compile-time chunk under
    `if constexpr`, so max_dynamic_chunk_size never reaches the split (served value 8 = dst_size with
    fp32_dest_acc_en false, the sdpa_decode.cpp:77 default; it matters only to the native k_chunk_size 0 call);
  - get_workload_for_core depends on cur_pos only through nearest_n(cur_pos + 1, 256), so every cur_pos in
    [E - 256, E - 1] gives the split of the compile-time call at capacity E (cur_pos_base = St * 32 - 1 = E - 1):
    same chunk count, same per-core ranges, same active tree;
  - the cores per head depend on B (16 for B <= 3 on the 11 x 10 grid, 13 at B = 4): a runtime-position call is
    the compile-time call's twin only at the same B;
  - the writer uses its own cur_pos for has_local_data and the active children; a writer left on the
    compile-time St while the reader and compute run a smaller runtime extent waits for children that have no
    chunk - a hang - exactly when E / 256 < cores_per_head (E < 4,096 at 16 cores), and it never skips a user.

No ttnn, no torch. Python 3.7-compatible. split() and tree_params() are memoised: treat what they return as
read-only.
"""

import functools

TILE = 32
K_CHUNK = 256                   # k_chunk_size of every served replay config
SK_CHUNK_T = K_CHUNK // TILE    # 8 tiles
MAX_TREE_REDUCTION_ROUNDS = 6
UINT32_MAX = 0xFFFFFFFF
GRID_CORES = 11 * 10            # p150a compute_with_storage_grid_size
KV_HEADS = 2                    # per chip
MAX_CORES_PER_HEAD_BATCH = 16   # SDPAProgramConfig's default; every probe and served call leaves it


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_of_2_up_to_8(maximum, x):
    """rt_args_common.hpp:13-33 (only three shift steps: exact up to 8, which is all the kernels use)."""
    if x == 0:
        return 1
    x -= 1
    x |= x >> 1
    x |= x >> 2
    result = x + 1
    return maximum if result > maximum else result


def dynamic_chunk_tiles(sk_chunk_t, max_size, cur_pos):
    """get_dynamic_Sk_chunk_t<Sk_chunk_t, max_size>(cur_pos) (rt_args_common.hpp:95-108)."""
    if sk_chunk_t == 0:
        seq_len_in_tiles = cur_pos // TILE + 1
        return nearest_pow_of_2_up_to_8(max_size, seq_len_in_tiles)
    return sk_chunk_t


def max_dynamic_chunk_size(fp32_dest_acc_en=False):
    """sdpa_decode_program_factory.cpp:388-389: dst_size, 4 with fp32 accumulation in DST, else 8."""
    return 4 if fp32_dest_acc_en else 8


def workload(cur_pos, core_num, cores_per_head, k_chunk_size, sliding_window=None):
    """get_workload_for_core (rt_args_common.hpp:35-93): (PSt, k_num_chunks, k_chunk_start, k_chunk_end,
    window_start_unaligned, window_start_chunk). cur_batch is unused by the function."""
    if sliding_window:
        window_end = cur_pos + 1
        window_start_unaligned = window_end - sliding_window if window_end > sliding_window else 0
        window_start = (window_start_unaligned // k_chunk_size) * k_chunk_size
        valid_seq_len = nearest_n(window_end, k_chunk_size) - window_start
    else:
        valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size)
        window_start = window_start_unaligned = 0
    pst = valid_seq_len // TILE
    window_start_chunk = window_start // k_chunk_size
    num_chunks = valid_seq_len // k_chunk_size
    if cores_per_head > num_chunks:
        chunks_per_core = 1 if core_num < num_chunks else 0
        start = window_start_chunk + (num_chunks - core_num - 1) * chunks_per_core
        end = window_start_chunk + (num_chunks - core_num) * chunks_per_core
    else:
        chunks_per_core = num_chunks // cores_per_head
        residuals = num_chunks % cores_per_head
        reversed_core = cores_per_head - core_num - 1
        start = window_start_chunk + reversed_core * chunks_per_core + min(residuals, reversed_core)
        end = start + chunks_per_core + (1 if reversed_core < residuals else 0)
    return pst, num_chunks, start, end, window_start_unaligned, window_start_chunk


def cores_per_head(batches, kv_heads=KV_HEADS, grid_cores=GRID_CORES, max_cores=MAX_CORES_PER_HEAD_BATCH):
    """sdpa_decode_program_factory.cpp:195-206 (num_heads_per_core is 1 for every shape here)."""
    if batches < 1 or batches > grid_cores:
        raise ValueError('batches must be 1..%d' % grid_cores)
    uncapped = min(grid_cores, max_cores * batches * kv_heads) // batches
    return max(1, uncapped // kv_heads)


@functools.lru_cache(maxsize=None)
def tree_params(core_id, cores):
    """get_tree_reduction_params (sdpa_decode_device_operation.hpp:38-91): dict(is_root, parent, send_at_round,
    children (per round, None when absent), num_rounds)."""
    children = [None] * MAX_TREE_REDUCTION_ROUNDS
    if cores <= 1:
        return dict(is_root=True, parent=None, send_at_round=None, children=children, num_rounds=0)
    vid = (cores - 1) - core_id
    num_rounds = (cores - 1).bit_length()
    is_root = core_id == 0
    for r in range(num_rounds):
        mask = (2 << r) - 1
        if vid & mask == mask:
            child_vid = vid - (1 << r)
            if child_vid < cores:
                children[r] = (cores - 1) - child_vid
    parent = send_at_round = None
    if not is_root:
        trailing_ones = 0
        while vid >> trailing_ones & 1:
            trailing_ones += 1
        parent_vid = vid + (1 << trailing_ones)
        parent = (cores - 1) - parent_vid if parent_vid < cores else 0
        send_at_round = trailing_ones
    if is_root:
        for c in range(1, cores):
            cv = (cores - 1) - c
            t = 0
            while cv >> t & 1:
                t += 1
            if cv + (1 << t) >= cores and children[t] is None:
                children[t] = c
    return dict(is_root=is_root, parent=parent, send_at_round=send_at_round, children=children,
                num_rounds=num_rounds)


@functools.lru_cache(maxsize=None)
def split(cur_pos, cores, sk_chunk_t=SK_CHUNK_T, max_size=8):
    """The whole head's split at one cur_pos, as the reader and the compute kernel compute it: the chunk count
    and each core's [start, end) chunk range, plus the writer's view of the tree (the children it waits for:
    child_id < k_num_chunks, sdpa writer_decode_all.cpp:173-183)."""
    chunk = dynamic_chunk_tiles(sk_chunk_t, max_size, cur_pos) * TILE
    ranges = []
    num_chunks = None
    for core in range(cores):
        _pst, num_chunks, start, end, _wsu, _wsc = workload(cur_pos, core, cores, chunk)
        ranges.append((start, end))
    waits = {}
    for core in range(cores):
        if ranges[core][0] == ranges[core][1]:
            continue                                   # has_local_data false: the core leaves the tree
        active = [child for child in tree_params(core, cores)['children']
                  if child is not None and child < num_chunks]
        waits[core] = tuple(active)
    return dict(chunk=chunk, num_chunks=num_chunks, ranges=tuple(ranges), waits=waits)


def compile_time_cur_pos(capacity):
    """The non-causal kernels' cur_pos_base = St * 32 - 1 (reader_decode_qwen.cpp:125, writer :118, compute :143)."""
    if capacity <= 0 or capacity % TILE:
        raise ValueError('capacity must be a positive multiple of %d' % TILE)
    return capacity - 1


def extent(start):
    """E = (start // 256 + 1) * 256: the 256-key family a replay block starting at `start` is served from
    (packed_verifier.py:490-493,533)."""
    return (start // K_CHUNK + 1) * K_CHUNK


def extent_cur_pos(start):
    """K64j's runtime cur_pos for a block at `start`: E - 1, which is also start | 255."""
    return extent(start) - 1


def same_split(runtime_cur_pos, capacity, cores, sk_chunk_t=SK_CHUNK_T, max_size=8):
    """Whether a runtime cur_pos reproduces the compile-time call at `capacity` (every field split() returns)."""
    return split(runtime_cur_pos, cores, sk_chunk_t, max_size) == split(compile_time_cur_pos(capacity), cores,
                                                                         sk_chunk_t, max_size)


def stale_writer_hangs(runtime_cur_pos, capacity, cores):
    """A writer left on the compile-time cur_pos (capacity - 1) beside a reader and a compute kernel on the
    runtime one. The writer decides has_local_data and its active children from ITS cur_pos
    (writer_decode_all.cpp:146-183), so on a core that has a chunk at the stale extent and none at the runtime
    one it waits for compute output that never comes (cb_out_worker, :343-345, or cb_out at the root), and a
    parent waits for a child's semaphore that is never raised (:266-279). Returns the blocked writers as sorted
    (core, why) pairs; empty means the stale writer's tree happens to be the live one (every core has a chunk at
    both extents). A skipped user (UINT32_MAX) always blocks such a writer: the reader and the compute kernel
    return at once while it reduces a full tree."""
    stale = split(compile_time_cur_pos(capacity), cores)
    if runtime_cur_pos == UINT32_MAX:
        live_ranges = [(0, 0)] * cores
    else:
        live_ranges = split(runtime_cur_pos, cores)['ranges']
    blocked = set()
    for core, children in stale['waits'].items():
        if live_ranges[core][0] == live_ranges[core][1]:
            blocked.add((core, 'waits for compute output (no chunk at the runtime extent)'))
        for child in children:
            if live_ranges[child][0] == live_ranges[child][1]:
                blocked.add((core, 'waits for child %d (no chunk at the runtime extent)' % child))
    return sorted(blocked)


def families(capacity):
    """Every 256-key family a runtime-extent call at the maximum `capacity` can serve: E = 256 .. capacity."""
    if capacity % K_CHUNK:
        raise ValueError('capacity must be a multiple of %d' % K_CHUNK)
    return list(range(K_CHUNK, capacity + 1, K_CHUNK))
