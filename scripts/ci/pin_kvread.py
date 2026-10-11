"""Move the qwen_kv_read extension's pinned sha256 to a fresh build (stdlib only; run on the rig host, from a checkout).

    python3 scripts/ci/pin_kvread.py --dir <build directory>            say what would change (exit 1 when the build is refused)
    python3 scripts/ci/pin_kvread.py --dir <build directory> --write    rewrite every file that carries the old pin

The extension (optimisation/ttnn-op/kv_region_read/build_kv_read.sh) is pinned by sha256 in the Dockerfile's RUN, build-c2-serving-image.sh, c2_image_provenance.py and
the overlay test, so an image never bakes bytes nobody built and checked. Version 2 (the raw block read and write of the host KV tier) is a new build with a new hash;
this moves all four in one command instead of four hand edits at the start of a card window. It refuses a build that is not this checkout's source (the directory's copy
of qwen_kv_read.cpp must be byte-identical to the checkout's), that has no MANIFEST.sha256 (the build writes it only after every check passed) or whose binary does not
carry the four op names (version 1 has no raw ops). Only the four files that ENFORCE the pin are rewritten (PIN_SITES); the files that merely quote the old hash - the version 1 qualification pack and the audit-cost
note, which describe the version 1 build - are listed and left alone, so they stay true.
"""

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CPP = ROOT / 'optimisation' / 'ttnn-op' / 'kv_region_read' / 'qwen_kv_read.cpp'
SHA256 = re.compile(r'[0-9a-f]{64}')
OP_NAMES = (b'qwen_read_blocks', b'qwen_read_blocks_raw', b'qwen_write_blocks_raw', b'qwen_block_bytes')
EXTENSION = 'qwen_kv_read.so'
SOURCE = 'qwen_kv_read.cpp'
MANIFEST = 'MANIFEST.sha256'
# The files that enforce the pin: the image build's RUN, the build script, the provenance check (g) and the overlay test that holds the three equal.
PIN_SITES = ('docker/qwen-c2-serving.Dockerfile', 'scripts/ci/build-c2-serving-image.sh', 'scripts/ci/c2_image_provenance.py', 'scripts/ci/test_c2_image_overlay.py')


def sha256_of(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_problems(directory, cpp=CPP):
    """Why the build in `directory` cannot be pinned, [] when it can."""
    directory = Path(directory)
    problems = []
    binary = directory / EXTENSION
    if not binary.is_file():
        return ['%s has no %s: run optimisation/ttnn-op/kv_region_read/build_kv_read.sh' % (directory, EXTENSION)]
    if not (directory / MANIFEST).is_file():
        problems.append('%s has no %s: the build writes it only after every check passed' % (directory, MANIFEST))
    built = directory / SOURCE
    if not built.is_file():
        problems.append('%s has no %s: it cannot be tied to a source' % (directory, SOURCE))
    elif sha256_of(built) != sha256_of(cpp):
        problems.append('%s is not this checkout\'s %s (the build is from another source: rebuild)' % (built, SOURCE))
    data = binary.read_bytes()
    absent = [name.decode('ascii') for name in OP_NAMES if name not in data]
    if absent:
        problems.append('%s does not carry %s: it is not version 2 of the extension' % (binary, ', '.join(absent)))
    return problems


def tracked_files(root, old):
    """The files under root that carry `old`: git's answer inside a checkout, else a walk of the text files."""
    try:
        result = subprocess.run(['git', '-C', str(root), 'grep', '-l', '-F', old], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if result.returncode in (0, 1):
            return sorted(root / line for line in result.stdout.decode('utf-8', 'replace').splitlines() if line)
    except (OSError, subprocess.SubprocessError):
        pass
    found = []
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and '.git' not in path.parts:
            try:
                if old in path.read_text(encoding='utf-8'):
                    found.append(path)
            except (UnicodeDecodeError, OSError):
                continue
    return found


def rewrite(files, old, new):
    """Replace `old` with `new` in each file (utf-8, newlines untouched). -> {path: occurrences}."""
    counts = {}
    for path in files:
        with open(str(path), 'r', encoding='utf-8', newline='') as handle:
            text = handle.read()
        if old not in text:
            continue
        counts[path] = text.count(old)
        with open(str(path), 'w', encoding='utf-8', newline='') as handle:
            handle.write(text.replace(old, new))
    return counts


def current_pin():
    sys.path.insert(0, str(HERE))
    import c2_image_provenance
    return c2_image_provenance.KVREAD_SHA256


def main(argv=None, root=ROOT, out=print, pin=None, cpp=CPP, sites=PIN_SITES):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dir', required=True, help='the build directory (build_kv_read.sh writes ~/opgraft-KVR by default)')
    parser.add_argument('--write', action='store_true', help='rewrite the files (default: say what would change)')
    options = parser.parse_args(argv)
    problems = build_problems(options.dir, cpp)
    if problems:
        for problem in problems:
            out('refused: %s' % problem)
        return 1
    new = sha256_of(Path(options.dir) / EXTENSION)
    old = pin or current_pin()
    if not SHA256.fullmatch(old) or not SHA256.fullmatch(new):
        out('refused: not a sha256 (old %r, new %r)' % (old, new))
        return 1
    if new == old:
        out('the build is already the pinned one (%s): nothing to move' % new)
        return 0
    carrying = tracked_files(root, old)
    files = [path for path in carrying if str(path.relative_to(root)) in sites]
    quoting = [path for path in carrying if path not in files]
    out('%s -> %s in %d files: %s' % (old[:12], new[:12], len(files), ', '.join(str(path.relative_to(root)) for path in files)))
    if quoting:
        out('left alone (they quote the old hash as history): %s' % ', '.join(str(path.relative_to(root)) for path in quoting))
    missing = [site for site in sites if not any(str(path.relative_to(root)) == site for path in files)]
    if missing:
        out('refused: %s should carry the old pin and does not: the pin sites have moved, update PIN_SITES' % ', '.join(missing))
        return 1
    if not options.write:
        out('(dry run: --write rewrites them)')
        return 0
    rewrite(files, old, new)
    left = [path for path in tracked_files(root, old) if path in files]
    if left:
        out('refused: the old pin is still in %s' % ', '.join(str(path) for path in left))
        return 1
    out('pinned %s: commit, run the CPU suite, push, then B0' % new)
    return 0


if __name__ == '__main__':
    sys.exit(main())
