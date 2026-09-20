"""Dump the image's model decode path so its 32-row (one tile) assumptions can be enumerated.

Run 35502452429 (image v48, four users in one 64-row packed block) died at attach in
the block's warm forward, before any of the block's own adapters ran:

    model.py:938 _forward_decode -> layer.py:188 attention_norm
    -> distributed_norm.py:85 all_gather_async
    TT_FATAL tensor_spec.cpp:161: Shard height 32 must match physical height 64 for width sharded

The model's decode mode builds width-sharded specs one tile (32 rows) high. A
64-row block needs every decode-mode config on the block's path (norms, residual
adds, MLP program configs, LM head, embeddings, RoPE) to accept two tiles, or the
block must route around each. The sources live only in the image, so this probe
prints them, with line numbers, for the analysis. CPU only: no device, no
weights, no imports from the model.
"""

import hashlib
import io
import re
import sys
from pathlib import Path

TT = Path('/opt/tt-metal')
QWEN = TT / 'models/demos/blackhole/qwen36/tt'
FULL = [
    TT / 'models/tt_transformers/tt/distributed_norm.py',
    QWEN / 'layer.py',
]
# (path, regex naming the function or region to print in full)
REGIONS = [
    (QWEN / 'model.py', r'def _forward_decode\b'),
    (QWEN / 'model.py', r'def _final_norm_decode\b'),
    (QWEN / 'model.py', r'def _lm_head\b'),
    (QWEN / 'model_config.py', r'def get_norm_config\b'),
    (QWEN / 'model_config.py', r'def get_model_config\b'),
]
GREPS = [
    (QWEN / 'model_config.py', r'shard|Shard|core_grid|per_core_M|tile_padded_batch|max_batch_size|DECODE|decode'),
    (QWEN / 'mlp.py', r'shard|Shard|core_grid|per_core_M|mode|decode|program_config'),
    (QWEN / 'attention.py', r'shard|Shard|core_grid|per_core_M|mode ==|decode|program_config'),
    (QWEN / 'gdn.py', r'shard|Shard|core_grid|per_core_M|mode ==|decode|program_config'),
    (TT / 'models/common/rmsnorm.py', r'shard|Shard|core_grid|mode|program_config|def forward'),
]


def show(label, value):
    print('%-44s %s' % (label, value))


def read(path):
    return io.open(path, encoding='utf-8', errors='replace').read()


def header(path, text):
    print()
    print('=' * 100)
    print('%s  sha256=%s  lines=%d' % (path, hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], text.count('\n') + 1))
    print('=' * 100)


def region(text, pattern):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped and (len(lines[j]) - len(lines[j].lstrip())) <= indent and not stripped.startswith(('#', '"""', "'''", ')', ']', '}')):
                    break
                j += 1
            return i, lines[i:j]
    return None, []


def main():
    listing = sorted(p.name for p in QWEN.glob('*.py')) if QWEN.is_dir() else []
    show('qwen36/tt modules', ' '.join(listing) or 'MISSING')
    for path in FULL:
        if not path.is_file():
            show('missing', path)
            continue
        text = read(path)
        header(path, text)
        for i, line in enumerate(text.split('\n'), 1):
            print('%5d  %s' % (i, line))
    for path, pattern in REGIONS:
        if not path.is_file():
            show('missing', path)
            continue
        text = read(path)
        start, lines = region(text, pattern)
        header('%s :: %s' % (path, pattern), text)
        if start is None:
            print('  (no match)')
            continue
        for i, line in enumerate(lines, start + 1):
            print('%5d  %s' % (i, line))
    for path, pattern in GREPS:
        if not path.is_file():
            show('missing', path)
            continue
        text = read(path)
        header('%s :: grep %s' % (path, pattern), text)
        for i, line in enumerate(text.split('\n'), 1):
            if re.search(pattern, line):
                print('%5d  %s' % (i, line.rstrip()))
    print()
    print('VERDICT')
    print('  The decode-path sources above are what a 64-row block must satisfy; enumerate every')
    print('  width-sharded spec, core grid and program config that assumes one 32-row tile.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
