"""Dump the chunked-prefill SDPA op (transformer/sdpa) as served, for the K/V prefix-sharing kernel.

M1 (the chunked-SDPA isolation bench, card M, image 0648ca9a) found the prefill SDPA bytes-bound:
bf16 KV costs 1.77x bf8 while exp-approx and fp32-off barely move it, and one layer's 2048-row
chunk at 126k keys takes 44.4 ms - the 22.7 s quadratic term of a 131k prompt. With 6 query heads
per KV head and 16 query chunks each, 96 cores stream the same KV prefix independently; the
factory's K/V chain forwarding is gated off for causal and chunked calls
(sdpa_program_factory.cpp:880 in the served fd8c0676). This prints every source file under the
sdpa op directory with line numbers and a sha256 - factory, reader/writer/compute kernels and the
shared headers - and greps the registration lines a graft rebuild needs, so the sharing kernel is
written against the served source. CPU only: no device, no build.
"""

import hashlib
import io
import re
import sys
from pathlib import Path

TT = Path('/opt/tt-metal')
OP = TT / 'ttnn/cpp/ttnn/operations/transformer/sdpa'
GREPS = [
    (TT / 'ttnn/cpp/ttnn/operations/transformer/sources.cmake', r'sdpa'),
    (TT / 'ttnn/cpp/ttnn/operations/transformer/CMakeLists.txt', r'sdpa|GLOB'),
    (TT / 'ttnn/cpp/ttnn/operations/transformer/transformer_nanobind.cpp', r'sdpa|bind_'),
]
SUFFIXES = ('.cpp', '.hpp', '.h')
FACTORY = 'device/sdpa_program_factory.cpp'
KNOWN = {
    'fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783': 'combined (served)',
    'a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc': 'original',
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
        print('  sdpa not at the expected path; locate it with find /opt/tt-metal -name "*sdpa_program_factory*".')
        return 0
    files = sorted(p for p in OP.rglob('*') if p.is_file() and p.suffix in SUFFIXES)
    print('sdpa files (%d): %s' % (len(files), ' '.join(str(p.relative_to(OP)) for p in files)))
    factory = OP / FACTORY
    verdict = KNOWN.get(sha(factory), 'UNKNOWN %s' % sha(factory)) if factory.is_file() else 'missing'
    print('audit %s %s' % (FACTORY, verdict))
    for path in files:
        dump(path)
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
    print('  %d sdpa source files dumped; factory is %s.' % (len(files), verdict))
    return 0


if __name__ == '__main__':
    sys.exit(main())
