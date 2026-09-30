"""Dump the sdpa_decode op the packed verify's attention runs on, for the one-pass kernel work.

The four-user packed verify reads each user's whole KV cache four times per layer: the
replay reader hands paged_scaled_dot_product_attention_decode a batch-3 and a batch-1
bundle of 4-row groups over the full capture family, and the program factory gives
every batch entry its own cores with no KV sharing (sdpa_decode_program_factory.cpp
:194-208 at sha 05708e6d; K multicast exists only on the MLA path). At 4 x 131k that
is ~128 ms of a 246 ms verify (docs/four-streams-131k-feasibility-2026-09-23.md).

Two kernel changes are planned against this op: read the dense mask only in the last
k-chunk, and read each k-chunk once for all of a user's row groups. Both need the
reader, compute and writer kernels and the factory exactly as the serving image has
them - the reader has never been available outside the image. This prints every
source file under the op directory with line numbers and a sha256, checks the factory
against sdpa_tree_scratch's audited hashes (original and tree-scratch patched), and
greps the registration lines a graft rebuild needs. CPU only: no device, no build.
"""

import hashlib
import io
import re
import sys
from pathlib import Path

TT = Path('/opt/tt-metal')
OP = TT / 'ttnn/cpp/ttnn/operations/transformer/sdpa_decode'
SHARED = [
    TT / 'ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp',
    TT / 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp',
    TT / 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp',
]
GREPS = [
    (TT / 'ttnn/cpp/ttnn/operations/transformer/sources.cmake', r'sdpa_decode'),
    (TT / 'ttnn/cpp/ttnn/operations/transformer/CMakeLists.txt', r'sdpa_decode|GLOB'),
    (TT / 'ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp', r'sdpa_decode|bind_'),
]
SUFFIXES = ('.cpp', '.hpp', '.h')
AUDITED = {
    'device/sdpa_decode_program_factory.cpp': {
        '05708e6d9ddeddfdf13303d8f8fa391941d73b742ea3a380beeb0883ce8d4792': 'original (audited)',
        '3e0a69af9563ae8899db1363286d6dc6cdc1744e0154887dc91a630b16084a4a': 'tree-scratch patched',
    },
    'device/kernels/dataflow/writer_decode_all.cpp': {
        '734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8': 'original (audited)'},
    'device/kernels/compute/sdpa_flash_decode.cpp': {
        'd24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f': 'original (audited)'},
}


def read(path):
    return io.open(path, encoding='utf-8', errors='replace').read()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path):
    text = read(path)
    print()
    print('=' * 100)
    print('%s  sha256=%s  lines=%d' % (path, sha(path), text.count('\n') + 1))
    print('=' * 100)
    for i, line in enumerate(text.split('\n'), 1):
        print('%5d  %s' % (i, line))


def main():
    if not OP.is_dir():
        print('MISSING op directory %s' % OP)
        print('VERDICT')
        print('  sdpa_decode not at the expected path; locate it with find /opt/tt-metal -name "*sdpa_decode*".')
        return 0
    files = sorted(p for p in OP.rglob('*') if p.is_file() and p.suffix in SUFFIXES)
    print('sdpa_decode files (%d): %s' % (len(files), ' '.join(str(p.relative_to(OP)) for p in files)))
    audit = {}
    for relative, known in AUDITED.items():
        path = OP / relative
        digest = sha(path) if path.is_file() else None
        audit[relative] = known.get(digest, 'UNKNOWN %s' % (digest or 'missing'))
        print('audit %-50s %s' % (relative, audit[relative]))
    for path in files:
        dump(path)
    for path in SHARED:
        if path.is_file():
            dump(path)
        else:
            print('shared header missing: %s' % path)
    for path, pattern in GREPS:
        if not path.is_file():
            print('missing %s' % path)
            continue
        print()
        print('grep %s :: %s' % (path, pattern))
        for i, line in enumerate(read(path).split('\n'), 1):
            if re.search(pattern, line):
                print('%5d  %s' % (i, line.rstrip()))
    print()
    print('VERDICT')
    print('  %d sdpa_decode source files dumped; factory is %s.' % (
        len(files), audit['device/sdpa_decode_program_factory.cpp']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
