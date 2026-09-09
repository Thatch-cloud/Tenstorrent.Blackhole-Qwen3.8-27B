"""Native widened rotary composition for DSpark proposal heads, not target-model arithmetic."""


POLICY = 'FP32-intermediate native rotary with BF16 tables/output; proposal-only, CPU rtol=atol=0.01'
CASES = ((16, 7, 32), (4, 39, 64), (4, 4103, 4128))


def execute(operations, heads, cosine, sine, owned):
    shape = tuple(heads.shape)
    if (len(shape) != 4 or shape[0] != 1 or shape[1] not in (4, 16) or shape[-1] != 128
            or not 32 <= shape[2] <= 262144 or shape[2] % 32
            or tuple(cosine.shape) != (1, 1, shape[2], 128) or sine.shape != cosine.shape):
        raise ValueError('Padded local DSpark query/key heads and matching absolute-position tables required')
    if any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT
            or value.memory_config() != operations.DRAM_MEMORY_CONFIG for value in (heads, cosine, sine)):
        raise ValueError('Caller-owned interleaved DRAM BF16 tile inputs required')

    def retain(value):
        owned.append(value)
        return value

    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    wide = [retain(operations.typecast(value, operations.float32)) for value in (heads, cosine, sine)]
    rotated = retain(operations.experimental.rotary_embedding_hf(*wide, is_decode_mode=False,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    return retain(operations.typecast(rotated, operations.bfloat16))


def fixtures(rotary, heads, live_rows, padded_rows):
    import torch

    if (heads, live_rows, padded_rows) not in CASES:
        raise ValueError('Declared query, short-key or 4K-key test geometry required')
    generator = torch.Generator().manual_seed(382600 + live_rows)
    original = torch.randn(2, heads, padded_rows, 128, generator=generator).bfloat16()
    changed = torch.randn(2, heads, padded_rows, 128, generator=generator).bfloat16()
    patterns = []
    for start, values in ((0, original), (8190, original), (262144 - live_rows, changed)):
        tables = [torch.ones(1, 1, padded_rows, 128, dtype=torch.bfloat16),
            torch.zeros(1, 1, padded_rows, 128, dtype=torch.bfloat16)]
        for table, valid in zip(tables, rotary.tables(start, live_rows), strict=True):
            table[:, :, :live_rows] = valid
        patterns.append([values.clone(), *tables])
    padding_changed = [value.clone() for value in patterns[2]]
    for value in padding_changed:
        value[..., live_rows:, :] = (torch.randn(value[..., live_rows:, :].shape, generator=generator) * 3).bfloat16()
    return [*patterns, padding_changed]
