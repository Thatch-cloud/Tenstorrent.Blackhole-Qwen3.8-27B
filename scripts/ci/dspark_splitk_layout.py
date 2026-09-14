"""Host oracle for folding draft query rows into heads for shared-KV split-K decode."""


def fold_query(query):
    if tuple(query.shape) != (1, 16, 32, 128):
        raise ValueError('Complete padded draft query [1,16,32,128] required')
    return query.reshape(4, 4, 32, 128).permute(0, 2, 1, 3).reshape(1, 1, 512, 128).contiguous()


def unfold_output(output):
    if tuple(output.shape) != (1, 1, 512, 128):
        raise ValueError('Complete folded draft output required')
    return output.reshape(4, 32, 4, 128).permute(0, 2, 1, 3).reshape(1, 16, 32, 128).contiguous()


def fold_mask(mask):
    if mask.ndim != 4 or tuple(mask.shape[:3]) != (1, 1, 32) or mask.shape[3] % 32:
        raise ValueError('Aligned complete per-query draft mask required')
    return mask.repeat_interleave(4, dim=2).repeat(1, 1, 4, 1).contiguous()


def scheduling(grid=(8, 8), max_cores_per_kv_head=16):
    if grid not in ((8, 8), (11, 10)) or max_cores_per_kv_head not in (16, 24):
        raise ValueError('Explicit Blackhole comparison geometry required')
    cores = grid[0] * grid[1]
    workers = min(cores, 4 * max_cores_per_kv_head) // 4
    return dict(prefill_query_work_items=16, configured_grid_cores=cores,
        decode_workers_per_kv_head=workers, decode_active_cores=workers * 4,
        folded_query_heads=512, kv_heads=4, batch_lanes=4, query_heads_per_lane=128,
        kv_heads_per_lane=1, copies_of_kv=1, hardware_qualified=False)
