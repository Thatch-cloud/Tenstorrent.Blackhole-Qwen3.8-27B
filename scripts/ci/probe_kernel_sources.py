"""Dump the two ttnn ops whose C++ limits cap the four-user 64-row round.

Per-user parity under four-way batching needs the 64-row verify round to run every
op once. Two ops cannot, and no Python model graft reaches them:

  1. ttnn.transformer.attn_decode_prep at batch 64 HANGS on the device (run
     35507675630, TT_METAL_WATCHER: its reader/writer/compute kernels resident,
     23 worker cores per chip stuck at cb_wait_front). The op is the project's own
     (optimisation/ttnn-op/test_attn_prep.py tests batch 1, 3, 8, 32 - never above).
  2. ttnn.experimental.nlp_concat_heads_decode refuses more than 32 users:
     TT_FATAL input_shape[1] <= 32 at nlp_concat_heads_decode_device_operation.cpp:39.

Both stay two-call on the 16 full-attention layers until their kernels handle two
batch tiles. This probe prints, from the image, every .cpp/.hpp/.h under both op
directories in full with line numbers and a sha256, so the kernel changes can be
written against the real source (device operation validation, program factory,
reader/writer/compute kernels, CB sizing, per-core work split). CPU only: no
device, no build, no imports.
"""

import hashlib
import io
import sys
from pathlib import Path

TT = Path('/opt/tt-metal')
OPS = [
    TT / 'ttnn/cpp/ttnn/operations/transformer/attn_prep',
    TT / 'ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode',
]
# The nanobind and sources.cmake registration lines matter for a graft rebuild.
GREPS = [
    (TT / 'ttnn/cpp/ttnn/operations/transformer/sources.cmake', r'attn_prep'),
    (TT / 'ttnn/cpp/ttnn/operations/transformer/CMakeLists.txt', r'attn_prep|GLOB'),
    (TT / 'ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp', r'attn_prep|bind_'),
    (TT / 'ttnn/cpp/ttnn/operations/experimental/transformer/CMakeLists.txt', r'nlp_concat_heads_decode|GLOB'),
]
SUFFIXES = ('.cpp', '.hpp', '.h', '.cmake', '.txt')


def show(label, value):
    print('%-44s %s' % (label, value))


def read(path):
    return io.open(path, encoding='utf-8', errors='replace').read()


def header(path, text):
    print()
    print('=' * 100)
    print('%s  sha256=%s  lines=%d' % (
        path, hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], text.count('\n') + 1))
    print('=' * 100)


def dump(path, text):
    header(path, text)
    for i, line in enumerate(text.split('\n'), 1):
        print('%5d  %s' % (i, line))


def main():
    found = 0
    for op in OPS:
        if not op.is_dir():
            show('MISSING op directory', op)
            continue
        files = sorted(p for p in op.rglob('*') if p.is_file() and p.suffix in SUFFIXES)
        show('%s files (%d)' % (op.name, len(files)), ' '.join(str(p.relative_to(op)) for p in files))
        for path in files:
            dump(path, read(path))
            found += 1
    import re
    for path, pattern in GREPS:
        if not path.is_file():
            show('missing', path)
            continue
        text = read(path)
        header('%s :: grep %s' % (path, pattern), text)
        hits = 0
        for i, line in enumerate(text.split('\n'), 1):
            if re.search(pattern, line):
                print('%5d  %s' % (i, line.rstrip()))
                hits += 1
        if not hits:
            print('  (no match)')
    print()
    print('VERDICT')
    if found:
        print('  %d source files dumped for the two C++-bounded ops; write the batch-64 kernel' % found)
        print('  changes against these lines and rebuild the two .so via the ttbuild graft.')
    else:
        print('  Neither op directory exists at the expected path; locate them (find /opt/tt-metal')
        print('  -name "*attn_prep*" -o -name "*nlp_concat_heads_decode*") and re-run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
