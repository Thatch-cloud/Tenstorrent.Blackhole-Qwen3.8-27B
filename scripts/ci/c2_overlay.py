"""The C2 serving image's overlay: one manifest (docker/qwen-c2-overlay.txt), four readers.

The C2 image (docker/qwen-c2-serving.Dockerfile) is P8 plus files laid over P8's fast-path
tree. Those files used to be named three times - the CI build step's copy loop, the
Dockerfile's cp line and the Dockerfile's extra destinations - and a file named in one list
but not another either never arrived or failed the build (memory
serving-image-bundle-provenance). Now docker/qwen-c2-overlay.txt names them once and this
module is the only thing that reads it:

  stage   (CI build step, or by hand)  copies the manifest's sources, the Dockerfile, the model
                                       graft and these tools into a build context;
  check   (build-c2-serving-image.sh)  refuses a context whose overlay/ tree is not exactly the
                                       manifest's sources;
  install (the Dockerfile)             copies each source to its destinations inside the image
                                       and records every destination's sha256 before and after;
  tests   (the Dockerfile)             prints the overlaid scripts/ci test modules to run there.

Stdlib only; runs on the rig host, in the image (python 3.10) and in the CPU suite.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

MANIFEST = 'docker/qwen-c2-overlay.txt'
DOCKERFILE = 'docker/qwen-c2-serving.Dockerfile'
GRAFT = 'docker/qwen-c2-graft'
TOOLS = ('scripts/ci/c2_overlay.py', 'scripts/ci/c2_image_provenance.py')
PURELIB = '@purelib/'

# Where the base image keeps each source root: a source with no listed destination replaces
# the base image's copy of itself.
ROOTS = {
    'scripts/ci/': '/experiment-scripts/ci/',
    'speculative-decoding/harness/': '/speculative-decoding/harness/',
}

# Sources the overlay must never carry, with the reason.
FORBIDDEN = {
    'scripts/ci/frozen_combined_runtime.py':
        'its bytes are pinned by the native cache (serving_native_install.py:57, adapter_sha256); '
        'change its callers instead',
}


class ManifestError(ValueError):
    pass


class Entry(object):
    __slots__ = ('source', 'destinations', 'line')

    def __init__(self, source, destinations, line):
        self.source, self.destinations, self.line = source, tuple(destinations), line

    def __repr__(self):
        return 'Entry(%r, %r)' % (self.source, self.destinations)


def default_destination(source):
    for root, destination in ROOTS.items():
        if source.startswith(root):
            return destination + source[len(root):]
    raise ManifestError('%s is under none of %s' % (source, ', '.join(sorted(ROOTS))))


def parse_manifest(text):
    """[Entry] in manifest order. Refuses anything ambiguous rather than guessing."""
    entries, sources, destinations = [], set(), {}
    for number, raw in enumerate(text.replace('\r\n', '\n').split('\n'), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        source, listed = fields[0], fields[1:]
        where = 'line %d (%s)' % (number, source)
        path = PurePosixPath(source)
        if path.is_absolute() or '..' in path.parts or str(path) != source or source.endswith('/'):
            raise ManifestError('%s: a source is a plain repo-relative file path' % where)
        if not any(source.startswith(root) and len(source) > len(root) for root in ROOTS):
            raise ManifestError('%s: sources live under %s' % (where, ', '.join(sorted(ROOTS))))
        if source in FORBIDDEN:
            raise ManifestError('%s: never overlaid - %s' % (where, FORBIDDEN[source]))
        if source in sources:
            raise ManifestError('%s: listed twice' % where)
        sources.add(source)
        targets = listed or [default_destination(source)]
        for target in targets:
            name = target[len(PURELIB):] if target.startswith(PURELIB) else None
            if name is not None:
                if not name or '/' in name or name in ('.', '..'):
                    raise ManifestError('%s: %s names one file directly in site-packages' % (where, target))
            else:
                target_path = PurePosixPath(target)
                if not target_path.is_absolute() or '..' in target_path.parts or str(target_path) != target:
                    raise ManifestError('%s: destination %s is not an absolute path' % (where, target))
            if target in destinations:
                raise ManifestError('%s: destination %s is also %s\'s' % (where, target, destinations[target]))
            destinations[target] = source
        entries.append(Entry(source, targets, number))
    if not entries:
        raise ManifestError('the overlay manifest names no files')
    return entries


def read_manifest(path):
    with open(str(path), encoding='utf-8') as handle:
        return parse_manifest(handle.read())


def resolve(destination, purelib):
    if destination.startswith(PURELIB):
        return purelib.rstrip('/') + '/' + destination[len(PURELIB):]
    return destination


def in_image_tests(entries):
    """Overlaid scripts/ci test modules, by module name, in manifest order."""
    return [PurePosixPath(entry.source).stem for entry in entries
            if entry.source.startswith('scripts/ci/test_') and entry.source.endswith('.py')
            and '/' not in entry.source[len('scripts/ci/'):]]


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def stage(repo, out):
    """Build context for docker/qwen-c2-serving.Dockerfile from a checkout.

    out/ gets: Dockerfile, qwen-c2-overlay.txt, the tools, overlay/<source> for every manifest
    entry, overlay.sha256, and the v235 model graft (graft/, graft.sha256, source.sha256).
    build-c2-serving-image.sh adds what only the rig has."""
    repo, out = Path(repo), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ManifestError('%s is not empty' % out)
    entries = read_manifest(repo / MANIFEST)
    missing = [entry.source for entry in entries if not (repo / entry.source).is_file()]
    if missing:
        raise ManifestError('the manifest names files the checkout lacks: ' + ', '.join(missing))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(repo / DOCKERFILE), str(out / 'Dockerfile'))
    shutil.copyfile(str(repo / MANIFEST), str(out / 'qwen-c2-overlay.txt'))
    for tool in TOOLS:
        shutil.copyfile(str(repo / tool), str(out / PurePosixPath(tool).name))
    lines = []
    for entry in entries:
        target = out / 'overlay' / entry.source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(repo / entry.source), str(target))
        lines.append('%s  %s\n' % (sha256(target), entry.source))
    (out / 'overlay.sha256').write_text(''.join(lines), encoding='utf-8')
    shutil.copytree(str(repo / GRAFT / 'graft'), str(out / 'graft'))
    for name in ('graft.sha256', 'source.sha256'):
        shutil.copyfile(str(repo / GRAFT / name), str(out / name))
    return entries


def check_context(ctx):
    """Problems with a staged context, [] when its overlay/ tree is exactly the manifest's."""
    ctx = Path(ctx)
    try:
        entries = read_manifest(ctx / 'qwen-c2-overlay.txt')
    except (OSError, ManifestError) as error:
        return ['manifest: %s' % error]
    problems = []
    for name in ('Dockerfile',) + tuple(PurePosixPath(tool).name for tool in TOOLS):
        if not (ctx / name).is_file():
            problems.append('%s is missing from the context' % name)
    wanted = {entry.source for entry in entries}
    root = ctx / 'overlay'
    present = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()} \
        if root.is_dir() else set()
    problems.extend('%s is in the manifest but not in overlay/' % name for name in sorted(wanted - present))
    problems.extend('overlay/%s is not in the manifest' % name for name in sorted(present - wanted))
    recorded = ctx / 'overlay.sha256'
    if recorded.is_file():
        for line in recorded.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition('  ')
            if name in present and sha256(root / name) != digest:
                problems.append('overlay/%s differs from overlay.sha256' % name)
    else:
        problems.append('overlay.sha256 is missing from the context')
    return problems


def install(manifest, root, record=None, purelib=None, log=None, image_root=None):
    """Copy every source under root to its destinations; return and record what changed.

    Each destination's sha256 before (None when new) and after is recorded, so a build log
    shows exactly which of the base image's files the overlay changed. image_root, when
    given, is prefixed to every destination (a staging tree instead of /)."""
    import sysconfig

    purelib = purelib or sysconfig.get_paths()['purelib']
    log = log or (lambda text: sys.stdout.write(text + '\n'))
    root = Path(root)
    entries = read_manifest(manifest)
    rows = []
    for entry in entries:
        source = root / entry.source
        digest = sha256(source)
        for destination in entry.destinations:
            target = resolve(destination, purelib)
            if image_root:
                target = os.path.join(str(image_root), target.lstrip('/'))
            before = sha256(target) if os.path.isfile(target) else None
            parent = os.path.dirname(target)
            if not os.path.isdir(parent):
                raise ManifestError('%s: no directory %s in the image' % (entry.source, parent))
            shutil.copyfile(str(source), target)
            after = sha256(target)
            if after != digest:
                raise ManifestError('%s: %s holds %s after the copy, not %s' % (entry.source, target, after, digest))
            state = 'new' if before is None else ('same' if before == after else 'changed')
            rows.append(dict(source=entry.source, destination=destination, path=target, state=state,
                             before=before, after=after))
            log('[C2-OVERLAY] %-7s %s -> %s %s%s' % (state, entry.source, target, after[:16],
                                                     '' if before in (None, after) else ' (was %s)' % before[:16]))
    if record:
        with open(str(record), 'w', encoding='utf-8') as handle:
            json.dump(dict(manifest=str(manifest), files=rows), handle, indent=1, sort_keys=True)
            handle.write('\n')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command')
    command = commands.add_parser('stage')
    command.add_argument('--repo', default='.')
    command.add_argument('--out', required=True)
    command = commands.add_parser('check')
    command.add_argument('--context', required=True)
    command = commands.add_parser('install')
    command.add_argument('--manifest', required=True)
    command.add_argument('--root', required=True)
    command.add_argument('--record')
    command = commands.add_parser('tests')
    command.add_argument('--manifest', required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == 'stage':
        entries = stage(arguments.repo, arguments.out)
        print('staged %d overlay files into %s' % (len(entries), arguments.out))
    elif arguments.command == 'check':
        problems = check_context(arguments.context)
        for problem in problems:
            sys.stderr.write('[C2-OVERLAY] context: %s\n' % problem)
        if problems:
            return 1
        print('[C2-OVERLAY] context matches the manifest')
    elif arguments.command == 'install':
        rows = install(arguments.manifest, arguments.root, arguments.record)
        counts = {}
        for row in rows:
            counts[row['state']] = counts.get(row['state'], 0) + 1
        print('[C2-OVERLAY] installed %d destinations: %s' % (
            len(rows), ', '.join('%d %s' % (counts[state], state) for state in sorted(counts))))
    elif arguments.command == 'tests':
        modules = in_image_tests(read_manifest(arguments.manifest))
        if not modules:
            sys.stderr.write('the manifest overlays no test modules\n')
            return 1
        print(' '.join(modules))
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
