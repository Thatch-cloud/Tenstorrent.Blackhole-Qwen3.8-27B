"""Explicit FP32 native RMS output followed by the required DSpark BF16 rounding."""

from dspark_projection import compute_config, require_tensor


POLICY = 'FP32 native unweighted RMS input/output; explicit BF16 cast before learned BF16 gamma'
COMPOSED_POLICY = 'FP32 square/sum/rsqrt/product composition; explicit BF16 cast before learned BF16 gamma'


def normalize(operations, narrowed, gamma, retain, *, composed=False):
    if not callable(retain) or type(composed) is not bool:
        raise ValueError('Explicit transient ownership required')
    require_tensor(operations,narrowed,(1,1,32,5120),operations.bfloat16)
    require_tensor(operations,gamma,(1,1,1,5120),operations.bfloat16)
    widened = retain(operations.typecast(narrowed,operations.float32))
    if composed:
        memory = operations.DRAM_MEMORY_CONFIG
        squared = retain(operations.multiply(widened,widened,fast_and_approximate_mode=False,memory_config=memory))
        summed = retain(operations.sum(squared,dim=3,keepdim=True,
            compute_kernel_config=compute_config(operations),memory_config=memory))
        variance = retain(operations.multiply(summed,1/5120,fast_and_approximate_mode=False,memory_config=memory))
        stabilized = retain(operations.add(variance,1e-6,dtype=operations.float32,memory_config=memory))
        reciprocal = retain(operations.rsqrt(stabilized,fast_and_approximate_mode=False,memory_config=memory))
        normalized = retain(operations.multiply(widened,reciprocal,fast_and_approximate_mode=False,memory_config=memory))
    else:
        normalized = retain(operations.rms_norm(widened,epsilon=1e-6,weight=None,
            compute_kernel_config=compute_config(operations),memory_config=operations.DRAM_MEMORY_CONFIG))
    require_tensor(operations,normalized,(1,1,32,5120),operations.float32)
    rounded = retain(operations.typecast(normalized,operations.bfloat16))
    context = retain(operations.mul(rounded,gamma,dtype=operations.bfloat16,memory_config=operations.DRAM_MEMORY_CONFIG))
    return dict(wide_norm=normalized,unweighted_norm=rounded,context=context)
