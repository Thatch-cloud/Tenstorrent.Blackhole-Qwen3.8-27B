"""Admit approximate T32 proposals only inside a source-bound hardware experiment."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from dflash_combined_sim_runtime import binary_hashes, BINARIES, BINARY_SHA256
from dflash_t16_native_attention_gate import PACKER, SIMULATOR_PACKER, native_hashes, hashes
from dflash_t32_cache_gate import qualify as qualify_cache


REPORTS = {
    31: 'aeb9e55a2426c31ea695bd191124abb233ca652dc6252bccc8c4e1cf858b8408',
    2048: '3e4e153046d2ebcebb3f9d822b081a04b1e8453a2890cee8499c6644d1bc6857',
}
CACHE_SHA256 = '2ad1b72e02ab978b82cfc252d8226a98a2c1d9a3296a15c13b85da418164e7a6'
CACHE_SOURCES = ('dflash-t32-cache-replay-probe.py', 'dflash_t32_cached_adapter.py',
    'dflash_t32_cache_gate.py', 'attention_batch.py', 'gdn_multitoken_conv.py',
    'dflash_device.py', 'dflash_proposal_trace.py', 'draft_kv_history.py',
    'dflash_proposal_inputs.py', 'dflash_attention_mask.py',
    'dflash_t32_native_attention.py', 'dflash_t16_native_attention.py')
_ACTIVE = ContextVar('dflash_t32_native_admission', default=None)


def require_active():
    if (any(os.environ.get(name) != '1' for name in ('QWEN_T32_COMBINED_EXPERIMENT',
            'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS', 'QWEN_FROZEN_COMBINED_RUNTIME'))
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('QWEN_SIM_ONLY') == '1'
            or os.environ.get('TT_METAL_DEVICE_PROFILER')):
        raise ValueError('Explicit allocated unprofiled T32 combined experiment required')
    admission = _ACTIVE.get()
    if admission is None:
        raise ValueError('Source-bound T32 proposal admission required')
    return admission


def validate_record(admission):
    if (not isinstance(admission, dict) or admission.get('block_rows') != 32
            or admission.get('reports') != {str(context): digest for context, digest in REPORTS.items()}
            or admission.get('cache_report_sha256') != CACHE_SHA256
            or admission.get('approximate_proposals') is not True
            or admission.get('accuracy_qualified') is not False
            or admission.get('target_attention_changed') is not False
            or admission.get('runtime_binaries') != dict.fromkeys(BINARIES, BINARY_SHA256)):
        raise ValueError('Explicit source-bound T32 proposal and cache record required')
    from dflash_t32_native_attention_gate import SOURCES

    directory = Path(__file__).parent
    if (admission.get('sources') != hashes(directory, SOURCES)
            or admission.get('cache_sources') != hashes(directory, CACHE_SOURCES)):
        raise ValueError('T32 request source record differs from current implementation')


def admit(attention_evidence, cache_evidence, directory, runtime):
    from dflash_t32_native_attention_gate import SOURCES, qualify

    attention_evidence, cache_evidence = Path(attention_evidence), Path(cache_evidence)
    sources = hashes(directory, SOURCES)
    native = native_hashes(runtime)
    binaries = binary_hashes(runtime)
    for context, expected in REPORTS.items():
        raw = (attention_evidence / f'dflash-t32-{context}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Pinned T32 native-attention report required')
        qualify(json.loads(raw), context, sources, native)
    raw = (cache_evidence / 'dflash-t32-cache.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != CACHE_SHA256:
        raise ValueError('Pinned T32 cache replay report required')
    if (cache_evidence / 'dflash-t32-cache.exit-status').read_text().strip() != '0':
        raise ValueError('Successful T32 cache simulator exit required')
    cached_sources = hashes(directory, CACHE_SOURCES)
    qualify_cache(json.loads(raw), cached_sources, {**native, PACKER: SIMULATOR_PACKER}, binaries)
    return dict(reports={str(context): digest for context, digest in REPORTS.items()},
        cache_report_sha256=CACHE_SHA256, sources=sources, cache_sources=cached_sources,
        native_sources=native, runtime_binaries=binaries, block_rows=32,
        approximate_proposals=True, accuracy_qualified=False, target_attention_changed=False)


@contextmanager
def scoped_native_t32(attention_evidence, cache_evidence, directory, runtime):
    import dflash_t32_cached_adapter
    import dflash_t32_native_attention

    if _ACTIVE.get() is not None:
        raise ValueError('Nested T32 proposal admission is unsupported')
    admission = admit(attention_evidence, cache_evidence, directory, runtime)
    token = _ACTIVE.set(admission)

    def attention(operations, query, key, value, mask, *, mask_validated=False):
        require_active()
        return dflash_t32_native_attention.native_attention(operations, query, key, value, mask,
            mask_validated=mask_validated)

    try:
        require_active()
        with patch.object(dflash_t32_cached_adapter, 'require_simulator', require_active), \
                patch.object(dflash_t32_native_attention, 'attention', attention):
            yield admission
    finally:
        _ACTIVE.reset(token)
        if admit(attention_evidence, cache_evidence, directory, runtime) != admission:
            raise ValueError('T32 proposal source admission changed during request')
