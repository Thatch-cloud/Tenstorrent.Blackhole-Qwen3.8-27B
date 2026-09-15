"""One proposed combined-runtime ladder geometry; planning is not admission."""


CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)


def geometry(context):
    if type(context) is not int or context not in CONTEXTS:
        raise ValueError('Explicit 4K through 262K ladder context required')
    capacity = context + 1024
    storage_keys = ((capacity + 15 + 63) // 64) * 64
    native_keys = ((storage_keys + 2047) // 2048) * 2048
    return dict(context=context, output_budget=256, streams=1, verifier_rows=16,
        draft_proposals=15, capacity=capacity, target_page_rows=64,
        target_page_count=capacity // 64, target_cache_blocks=capacity // 64 + 8,
        storage_keys=storage_keys, native_keys=native_keys,
        extra_masked_keys=native_keys - storage_keys, key_chunk_size=256,
        splitk_workers_per_lane=8, key_chunks_per_worker=native_keys // 256 // 8,
        draft_history_bank_bytes_per_chip=2 * 5 * 2 * 4 * capacity * 128 * 2,
        memory_scope='Two BF16 draft history banks only; excludes weights, target KV, traces and scratch',
        runtime_admitted=False, hardware_fit_qualified=False, performance_qualified=False)


def require_reference(reference):
    expected = geometry(65536)
    if any(reference.get(key) != expected[key] for key in ('context', 'capacity', 'storage_keys', 'native_keys')):
        raise ValueError('The proposed ladder must preserve the exact admitted 64K padded geometry')
    return expected


def ladder():
    return tuple(geometry(context) for context in CONTEXTS)
