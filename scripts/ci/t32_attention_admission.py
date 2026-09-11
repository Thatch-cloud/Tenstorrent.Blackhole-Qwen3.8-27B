"""Require the simulator-validated attention kernel for experimental captured T32 drafting."""

import os

from native_draft_sdpa import audit_active_kernel
from t32_ci_runtime import fingerprints


PATCHED = {
    'compute_common.hpp': '6e5c1cf9404c436ee9ceb23b4e134aaea1fd49df09c2eba5bc68c6a16cb4d061',
    'sdpa.cpp': 'bbfc37121a1c4461a7c385cf5257edacc9d9694ade001fa20d9f3e3277721b5b',
}


def require_active():
    required = ('QWEN_SIM_ONLY', 'QWEN_PRECISE_DRAFT_ACTIVE', 'QWEN_T32_SFPU_SUM')
    forbidden = ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED', 'QWEN_T32_FP32_BUILD',
                 'QWEN_T32_PRECISE_RECIP', 'QWEN_T32_EXPLICIT_PACK')
    if (any(os.environ.get(name) != '1' for name in required)
            or any(os.environ.get(name) == '1' for name in forbidden)
            or os.environ.get('QWEN_T32_NUMERATOR_TAP', '0') != '0'):
        raise ValueError('Captured T32 drafting requires the validated simulator SFPU sum, without diagnostic taps')
    root = os.environ.get('TT_METAL_HOME')
    if not root:
        raise ValueError('Explicit runtime root required')
    audit = audit_active_kernel(root)
    if audit.get('patched') != PATCHED:
        raise ValueError('T32 attention differs from simulator run 34653659471')
    return dict(component_run=34653659471, kernel=audit, runtime=fingerprints(root),
                full_request_qualified=False)
