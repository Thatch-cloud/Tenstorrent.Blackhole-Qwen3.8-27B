"""The C2 serving image's overlay: one manifest (docker/qwen-c2-overlay.txt), four readers.

The C2 image (docker/qwen-c2-serving.Dockerfile) is P8 plus files laid over P8's fast-path
tree. Those files used to be named three times - the CI build step's copy loop, the
Dockerfile's cp line and the Dockerfile's extra destinations - and a file named in one list
but not another either never arrived or failed the build (memory
serving-image-bundle-provenance). Now docker/qwen-c2-overlay.txt names them once and this
module is the only thing that reads it:

  stage   (CI build step, or by hand)  copies the manifest's sources, the Dockerfile, the model
                                       graft, the build script and these tools into a build
                                       context, from a checkout whose staged files are HEAD's
                                       (source-revision), with the layer model (layers.json)
                                       G1 holds the built image to;
  check   (build-c2-serving-image.sh)  refuses a context whose overlay/ tree is not exactly the
                                       manifest's sources, or that lacks a staged record;
  install (the Dockerfile)             copies each source to its destinations inside the image
                                       and records every destination's sha256 before and after;
                                       refuses a source whose bytes break a frozen-recipe pin;
  tests   (the Dockerfile)             prints the overlaid scripts/ci test modules to run there.

The parser refuses what an overlay cannot do: a pinned file (FORBIDDEN, by name, as source or
destination), a file whose effect P8 applied once at build time (BUILD_TIME_ONLY), and a
file the TT plugin carries its own copy of without that copy among its destinations
(REQUIRED_DESTINATIONS).

Stdlib only; runs on the rig host, in the image (python 3.10) and in the CPU suite.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import sys

MANIFEST = 'docker/qwen-c2-overlay.txt'
DOCKERFILE = 'docker/qwen-c2-serving.Dockerfile'
GRAFT = 'docker/qwen-c2-graft'
BUILD_SCRIPT = 'scripts/ci/build-c2-serving-image.sh'
# The v235 gate's QWEN_ environment (run 36087022223, m3native-gate.json qwen_configuration):
# G1 holds the exact profile's booted environment to it.
V235_ENVIRONMENT = 'docker/qwen-c2-v235-environment.json'
TOOLS = ('scripts/ci/c2_overlay.py', 'scripts/ci/c2_image_provenance.py', BUILD_SCRIPT)
# Repo files staged into the context under another name.
CONTEXT_FILES = ((DOCKERFILE, 'Dockerfile'), (MANIFEST, 'qwen-c2-overlay.txt'),
                 (V235_ENVIRONMENT, 'v235-environment.json'))
REVISION_FILE = 'source-revision'
LAYERS_FILE = 'layers.json'
PURELIB = '@purelib/'
PLUGIN = '/opt/qwen-fast-plugin/src/vllm_tt_plugin/'
# The directory frozen_combined_runtime.qualify is handed on every exact request.
PINNED_TREE = '/experiment-scripts/ci'

# Where the base image keeps each source root: a source with no listed destination replaces
# the base image's copy of itself.
ROOTS = {
    'scripts/ci/': '/experiment-scripts/ci/',
    'speculative-decoding/harness/': '/speculative-decoding/harness/',
}

# File names the overlay must never carry, as a source OR a destination, with the reason.
FORBIDDEN = {
    'frozen_combined_runtime.py':
        'its bytes are pinned by the native cache (serving_native_install.py:57, adapter_sha256); '
        'change its callers instead',
    'tp_common.py':
        'the serving path\'s simulator gates hash it: a changed copy dies at attach (memory '
        'qualification-pins-model-sources, run 35802496949); patch its callers instead',
}

# Scripts P8's Dockerfile RUNs once at build time (docker/qwen-fast-serving.Dockerfile at the P8
# base commit). Their effect is baked into P8, so overlaying one changes nothing at run time.
BUILD_TIME_ONLY = {
    'scripts/ci/serving_native_install.py': 'P8 RUNs it once to install the native runtime',
    'scripts/ci/serving_plugin_patch.py': 'P8 RUNs it once: it patches the TT plugin\'s config, platform, '
                                          'worker and entrypoints and copies the policy and registry in',
    'scripts/ci/serving_image_preflight.py': 'P8 RUNs it once to write startup-preflight.json',
}

# Sources the TT plugin imports its own copy of (serving_plugin_patch.stage copies them in at P8
# build). Overlaying one without that copy leaves the plugin on P8's version.
REQUIRED_DESTINATIONS = {
    'scripts/ci/serving_fast_policy.py': PLUGIN + 'qwen_fast_policy.py',
    'scripts/ci/serving_dflash_registry.py': PLUGIN + 'qwen_dflash_registry.py',
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
        if path.name in FORBIDDEN:
            raise ManifestError('%s: never overlaid - %s' % (where, FORBIDDEN[path.name]))
        if source in BUILD_TIME_ONLY:
            raise ManifestError('%s: %s, so overlaying it changes nothing at run time - it needs a P8 '
                                'rebuild or a re-apply RUN' % (where, BUILD_TIME_ONLY[source]))
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
                name = target_path.name
            if name in FORBIDDEN:
                raise ManifestError('%s: destination %s is never overwritten - %s' % (where, target, FORBIDDEN[name]))
            if target in destinations:
                raise ManifestError('%s: destination %s is also %s\'s' % (where, target, destinations[target]))
            destinations[target] = source
        required = REQUIRED_DESTINATIONS.get(source)
        if required and required not in targets:
            raise ManifestError('%s: the TT plugin imports its own copy at %s (serving_plugin_patch copies it '
                                'there at P8 build); list that destination too' % (where, required))
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


def frozen_pins(ci, logical=None):
    """What frozen_combined_runtime.qualify pins, read from a tree as an image holds it.

    Exact runs qualify on every request (verifier_engine -> target_t16_attention_gate ->
    frozen_combined_runtime.qualify): it hashes every source the frozen evidence names against
    the tree and raises inside execute_model at the first mismatch, which kills the engine
    (image v42, run 35495227738). The map is built exactly as qualify builds it (and as
    probe_frozen_pins.py did): draft-numerical.json's sources less '-probe.py', '../' and three
    exclusions, then target-replay.json's less '-probe.py'.

    ci is the directory qualify is handed; logical, when given, is the path the result's keys
    use for it (a staging tree standing in for the image's /experiment-scripts/ci). Returns
    dict(evidence, reports={name: sha256}, pins={path: pinned sha256}, live={path: sha256 or
    None}, problems=[...]). Self-contained, because c2_image_provenance runs its source inside
    images that do not hold this module."""
    import hashlib
    import json
    import os
    import posixpath

    logical = logical or ci
    evidence = os.path.join(ci, 'frozen-evidence')
    result = dict(evidence=os.path.isdir(evidence), reports={}, pins={}, live={}, problems=[])
    if not result['evidence']:
        return result
    sources = {}
    for name in ('draft-numerical.json', 'target-replay.json'):
        sources[name] = {}
        try:
            with open(os.path.join(evidence, name), 'rb') as handle:
                payload = handle.read()
            result['reports'][name] = hashlib.sha256(payload).hexdigest()
            named = json.loads(payload.decode('utf-8'))['sources']
            if not isinstance(named, dict):
                raise TypeError('sources is not a map')
            sources[name] = named
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['problems'].append('frozen-evidence/%s: %s' % (name, error))
    excluded = {'frozen_probe_evidence.py', 'dspark_attention_8k_gate.py', 'dspark_attention_value_diagnostics.py'}
    expected = {name: checksum for name, checksum in sources['draft-numerical.json'].items()
                if not name.endswith('-probe.py') and not name.startswith('../') and name not in excluded}
    for name, checksum in sources['target-replay.json'].items():
        if name.endswith('-probe.py'):
            continue
        if name in expected and expected[name] != checksum:
            result['problems'].append('the frozen evidence disagrees on shared source %s' % name)
        expected[name] = checksum
    for name, checksum in sorted(expected.items()):
        key = posixpath.normpath(posixpath.join(logical, name))
        real = os.path.normpath(os.path.join(ci, name))
        result['pins'][key] = checksum
        live = None
        if os.path.isfile(real):
            with open(real, 'rb') as handle:
                live = hashlib.sha256(handle.read()).hexdigest()
        result['live'][key] = live
    return result


def tree_digest(top):
    """sha256 over every entry under top - a file's sha256, a symlink's target, each with its
    path relative to top - or None when top is not a directory. Self-contained, like
    frozen_pins: G1 runs it inside the image on each K64i op directory and here on the
    context's copy."""
    import hashlib
    import os

    if not os.path.isdir(top):
        return None
    lines = []
    for base, directories, names in os.walk(top):
        for name in list(directories) + list(names):
            path = os.path.join(base, name)
            relative = os.path.relpath(path, top).replace(os.sep, '/')
            if os.path.islink(path):
                lines.append('link %s -> %s\n' % (relative, os.readlink(path)))
            elif os.path.isfile(path):
                with open(path, 'rb') as handle:
                    lines.append('file %s %s\n' % (relative, hashlib.sha256(handle.read()).hexdigest()))
    return hashlib.sha256(''.join(sorted(lines)).encode('utf-8')).hexdigest()


def git(repo, *arguments):
    result = subprocess.run(['git', '-C', str(repo), '--no-optional-locks'] + list(arguments),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise ManifestError('git %s: %s' % (' '.join(arguments), result.stderr.decode('utf-8', 'replace').strip()))
    return result.stdout.decode('utf-8', 'replace')


def staged_paths(entries):
    """Every repo path stage copies into a context."""
    return ([name for name, _ in CONTEXT_FILES] + list(TOOLS) + [GRAFT]
            + [entry.source for entry in entries])


def unclean(repo, paths):
    """Staged paths whose working-tree state is not HEAD's (modified, staged, untracked)."""
    return sorted({line[3:].strip() for line in git(repo, 'status', '--porcelain', '--untracked-files=all',
                                                     '--', *paths).splitlines() if line.strip()})


def stage(repo, out, require_clean=True, layers=True):
    """Build context for docker/qwen-c2-serving.Dockerfile from a checkout.

    out/ gets: Dockerfile, qwen-c2-overlay.txt, v235-environment.json, the tools and the build
    script, overlay/<source> for every manifest entry, overlay.sha256, the v235 model graft
    (graft/, graft.sha256, source.sha256), source-revision (the checkout's HEAD) and, with
    layers, layers.json (c2_image_layers.expected_tree: the version of every repo file the
    image should hold). build-c2-serving-image.sh adds what only the rig has.

    require_clean refuses a checkout whose staged files differ from HEAD: the context would
    otherwise ship bytes no commit holds while the closure tests (which read HEAD) pass."""
    repo, out = Path(repo), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ManifestError('%s is not empty' % out)
    entries = read_manifest(repo / MANIFEST)
    missing = [entry.source for entry in entries if not (repo / entry.source).is_file()]
    if missing:
        raise ManifestError('the manifest names files the checkout lacks: ' + ', '.join(missing))
    if require_clean:
        dirty = unclean(repo, staged_paths(entries))
        if dirty:
            raise ManifestError('these staged files differ from HEAD; commit them first: ' + ', '.join(dirty))
    revision = git(repo, 'rev-parse', 'HEAD').strip()
    tree = None
    if layers:
        sys.path.insert(0, str(repo / 'scripts' / 'ci'))
        try:
            import c2_image_layers
        finally:
            sys.path.pop(0)
        tree = c2_image_layers.expected_tree(c2_image_layers.Repo(repo))
        if tree['commits']['head'] != revision:
            raise ManifestError('the layer model read %s, not HEAD %s' % (tree['commits']['head'], revision))
    out.mkdir(parents=True, exist_ok=True)
    for name, staged in CONTEXT_FILES:
        shutil.copyfile(str(repo / name), str(out / staged))
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
    (out / REVISION_FILE).write_text(revision + '\n', encoding='utf-8')
    if tree is not None:
        with open(str(out / LAYERS_FILE), 'w', encoding='utf-8') as handle:
            json.dump(tree, handle, indent=0, sort_keys=True)
            handle.write('\n')
    return entries


def check_context(ctx):
    """Problems with a staged context, [] when its overlay/ tree is exactly the manifest's."""
    ctx = Path(ctx)
    try:
        entries = read_manifest(ctx / 'qwen-c2-overlay.txt')
    except (OSError, ManifestError) as error:
        return ['manifest: %s' % error]
    problems = []
    names = [staged for _, staged in CONTEXT_FILES] + [PurePosixPath(tool).name for tool in TOOLS]
    for name in names + [REVISION_FILE, LAYERS_FILE]:
        if not (ctx / name).is_file():
            problems.append('%s is missing from the context' % name)
    if (ctx / REVISION_FILE).is_file():
        revision = (ctx / REVISION_FILE).read_text(encoding='utf-8').strip()
        if len(revision) != 40 or any(char not in '0123456789abcdef' for char in revision):
            problems.append('%s is not a commit: %r' % (REVISION_FILE, revision))
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


def pin_breaks(entries, root, pins, purelib):
    """Overlay destinations whose new bytes differ from a frozen-recipe pin (frozen_pins)."""
    broken = []
    for entry in entries:
        digest = sha256(Path(root) / entry.source)
        for destination in entry.destinations:
            target = posixpath.normpath(resolve(destination, purelib))
            pinned = pins['pins'].get(target)
            if pinned is not None and pinned != digest:
                broken.append('%s -> %s is %s; the frozen recipe pins %s' % (entry.source, target, digest, pinned))
    return broken


def install(manifest, root, record=None, purelib=None, log=None, image_root=None):
    """Copy every source under root to its destinations; return and record what changed.

    Each destination's sha256 before (None when new) and after is recorded, so a build log
    shows exactly which of the base image's files the overlay changed. image_root, when
    given, is prefixed to every destination (a staging tree instead of /). Nothing is copied
    when a source would break a frozen-recipe pin (frozen_pins): exact would die at its
    first request."""
    import sysconfig

    purelib = purelib or sysconfig.get_paths()['purelib']
    log = log or (lambda text: sys.stdout.write(text + '\n'))
    root = Path(root)
    entries = read_manifest(manifest)

    def real(path):
        return os.path.join(str(image_root), path.lstrip('/')) if image_root else path

    pins = frozen_pins(real(PINNED_TREE), PINNED_TREE)
    broken = pin_breaks(entries, root, pins, purelib)
    if broken:
        raise ManifestError('the overlay breaks the frozen recipe\'s source pin, which kills exact at its first '
                            'request (frozen_combined_runtime.qualify; image v42): ' + '; '.join(broken))
    touched = sorted(set(pins['pins']) & {posixpath.normpath(resolve(d, purelib))
                                           for entry in entries for d in entry.destinations})
    log('[C2-OVERLAY] frozen pins: %s; %d pinned sources, %d overlaid with their pinned bytes%s' % (
        'evidence present' if pins['evidence'] else 'NO frozen-evidence in %s' % PINNED_TREE,
        len(pins['pins']), len(touched), (' (%s)' % ', '.join(touched)) if touched else ''))
    rows = []
    for entry in entries:
        source = root / entry.source
        digest = sha256(source)
        for destination in entry.destinations:
            target = real(resolve(destination, purelib))
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
            json.dump(dict(manifest=str(manifest), files=rows, frozen_pins=pins), handle, indent=1, sort_keys=True)
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
