"""Explicit BF16 residual rounding through a native FP32 sum, without a host arithmetic fallback."""

from dspark_projection import require_tensor


POLICY = 'Native FP32 sum of BF16 operands followed by explicit BF16 cast'
WIDE_POLICY = 'Native FP32 sum of explicitly widened BF16 operands followed by explicit BF16 cast'


def add(operations, first, second, retain, *, widen_inputs=False):
    if not callable(retain) or type(widen_inputs) is not bool:
        raise ValueError('Explicit residual ownership required')
    for value in (first,second):
        require_tensor(operations,value,(1,1,32,5120),operations.bfloat16)
    if widen_inputs:
        first,second = [retain(operations.typecast(value,operations.float32)) for value in (first,second)]
    widened = retain(operations.add(first,second,dtype=operations.float32,memory_config=operations.DRAM_MEMORY_CONFIG))
    return retain(operations.typecast(widened,operations.bfloat16))
