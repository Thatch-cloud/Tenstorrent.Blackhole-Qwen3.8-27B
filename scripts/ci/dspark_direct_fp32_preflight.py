"""Compose the actual admitted request kernels before loading any model weights."""

import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import native_draft_sdpa
from dspark_attention_header_boundary import PREFIX, original_header
from dspark_direct_fp32_stage import REPLACEMENT


def validate_active_sources(root):
    directory = Path(root) / native_draft_sdpa.KERNEL_DIRECTORY
    original = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
    patched = native_draft_sdpa.patched_sources(original)
    header = patched['compute_common.hpp']
    if os.environ.get('QWEN_DSPARK_CENTER_TILE_FILL') == '1':
        if b'fill_tile(0, 0.0f);' not in header or b'scratch[index] = 0;' in header:
            raise ValueError('Center tile-fill must replace the actual scalar scratch clear')
    if os.environ.get('QWEN_DSPARK_NORMALIZATION_DIRECT_STAGE') == '1':
        if b'qwen_stage_score_tile(reciprocal_cb, scratch_cb, false);' not in header:
            raise ValueError('Normalization staging must reach the actual request kernel')
    if (REPLACEMENT.encode() not in header
            or header.count(b'qwen_stage_score_tile(in0_cb, QWEN_SCORE_SCRATCH_CB, true);') != 1):
        raise ValueError('The actual request scope must install the complete qualified staging arithmetic')
    if not header.startswith(PREFIX.encode()) or original_header(header.decode()).encode() != original['compute_common.hpp']:
        raise ValueError('The target attention branch must retain the pinned native header')
    return dict(passed=True, scope=__doc__, device_execution=False, model_weights_loaded=False,
        original=native_draft_sdpa.SOURCE_HASHES,
        patched={name: hashlib.sha256(source).hexdigest() for name, source in patched.items()},
        performance_qualified=False)


def main():
    if os.environ.get('QWEN_DSPARK_DIRECT_FP32_STAGE') != '1':
        raise ValueError('Explicit direct-staging request preflight required')
    from dspark_64k_entry import run
    started = perf_counter()
    with patch.object(sys, 'argv', [__file__, '--captured-publication', '--norm-scatter-variants']):
        report = run(lambda: validate_active_sources(os.environ['TT_METAL_HOME']))
    report['elapsed_seconds'] = perf_counter() - started
    Path('/experiment/results/direct-fp32-source-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(stage='direct-fp32-source-preflight', **report)), flush=True)


if __name__ == '__main__':
    main()
