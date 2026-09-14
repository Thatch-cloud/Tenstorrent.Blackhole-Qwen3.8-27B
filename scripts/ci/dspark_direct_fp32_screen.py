"""Audited folded-T16 combined request with qualified direct FP32 staging."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import native_draft_sdpa
import target_t16_64k_screen as baseline
from dspark_direct_fp32_request_gate import qualify
from dspark_direct_fp32_stage import staging_scope


BASE_SUMMARIZE = baseline.summarize_screen


def summarize_screen(requests, *, expected, admission, sources):
    for value in requests:
        actual = value.get('native_attention_kernel', {})
        if actual.get('original') != native_draft_sdpa.SOURCE_HASHES or actual.get('patched') != expected:
            raise ValueError('Every request must compile the actual qualified direct-staging kernel')
    result = BASE_SUMMARIZE(requests)
    result.update(direct_fp32_stage=True, direct_fp32_hardware_admission=admission,
        direct_fp32_sources=sources)
    return result


@contextmanager
def screen_scope(directory):
    directory = Path(directory)
    admission = qualify(directory, directory / 'dspark-direct-fp32-stage-hardware.json')
    dependencies = ('dspark_direct_fp32_stage.py', 'dspark_direct_fp32_request_gate.py', Path(__file__).name)
    sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in dependencies}

    def summarize(requests):
        root = Path(os.environ['TT_METAL_HOME']) / native_draft_sdpa.KERNEL_DIRECTORY
        original = {name: (root / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        expected = {name: hashlib.sha256(source).hexdigest()
            for name, source in native_draft_sdpa.patched_sources(original).items()}
        return summarize_screen(requests, expected=expected, admission=admission, sources=sources)

    with staging_scope(), patch.object(baseline, 'summarize_screen', summarize), baseline.screen_scope(directory):
        yield
