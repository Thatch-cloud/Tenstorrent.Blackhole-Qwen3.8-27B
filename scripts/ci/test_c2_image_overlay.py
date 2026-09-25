"""The C2 serving image ships HEAD's version of every file it carries - or says why not.

The C2 image (docker/qwen-c2-serving.Dockerfile) holds the fast path's trees in three layers,
and a file lands at the version of the LAST layer that names it:

  1. the bundle archive, commit 77d6995a: /experiment-scripts/ci and /speculative-decoding;
  2. P8's copy lists (docker/qwen-fast-serving.Dockerfile + qwen-fast-serving-image.yml), at the
     commit the P8 base was built from - the C2 Dockerfile's ARG BASE, qwen-fast-serving:ci-<sha>.
     NOT at HEAD: C2 does not rebuild P8, so a P8-listed file changed since that commit is stale
     in C2 however the P8 lists read today;
  3. docker/qwen-c2-overlay.txt, at the commit the C2 image is built from (HEAD).

A file changed since its layer's commit and not re-laid by a later layer ships OLD. That is how
v87 ran ordered_cache.py's old line 54 (run 35790454545), and why S1/S2 changes to
serving_worker_hook, serving_lifecycle, serving_packed_step and greedy_session would silently
not ship: they sit in layers 1-2 only. test_serving_image_copy_closure guards P8's two lists;
this module guards what the C2 image ends up holding.

Everything is read from git; no image, no device. The CPU workflow checks out fetch-depth 0,
and test_the_layer_commits_are_reachable fails loudly on a shallow clone.
"""

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_image_provenance as provenance  # noqa: E402
import c2_overlay  # noqa: E402
import test_serving_image_copy_closure as p8  # noqa: E402

C2_DOCKERFILE = ROOT / c2_overlay.DOCKERFILE
C2_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml'
BUILD_SCRIPT = HERE / 'build-c2-serving-image.sh'
MANIFEST = ROOT / c2_overlay.MANIFEST
P8_DOCKERFILE = 'docker/qwen-fast-serving.Dockerfile'
P8_WORKFLOW = '.github/workflows/qwen-fast-serving-image.yml'
TREES = ('scripts/ci/', 'speculative-decoding/harness/')
SUFFIXES = ('.py', '.cpp', '.h', '.hpp', '.json', '.sh')

# Files the C2 image deliberately ships at an older version than HEAD, beyond p8.KNOWN_STALE
# (which covers layer 1 files neither P8 list overlays), each with a reason. Empty today.
C2_KNOWN_STALE = {}

# What the C2 layer overlaid before the manifest existed (the workflow's nine-file loop and the
# Dockerfile's cp line, image v6-teardown). Each must keep reaching the same place.
PRE_MANIFEST = {
    'scripts/ci/serving_fast_policy.py': ('/experiment-scripts/ci/serving_fast_policy.py',
                                          '/opt/qwen-fast-plugin/src/vllm_tt_plugin/qwen_fast_policy.py'),
    'scripts/ci/packed_verifier.py': ('/experiment-scripts/ci/packed_verifier.py',),
    'scripts/ci/serving_request_factory.py': ('/experiment-scripts/ci/serving_request_factory.py',),
    'scripts/ci/pooled_attention_replay.py': ('/experiment-scripts/ci/pooled_attention_replay.py',),
    'scripts/ci/serving_c2_contract.py': ('/experiment-scripts/ci/serving_c2_contract.py',),
    'scripts/ci/test_serving_c2_contract.py': ('/experiment-scripts/ci/test_serving_c2_contract.py',),
    'scripts/ci/test_serving_fast_policy.py': ('/experiment-scripts/ci/test_serving_fast_policy.py',),
    'scripts/ci/qwen_c2_profiles.json': ('/experiment-scripts/ci/qwen_c2_profiles.json', '/opt/qwen-c2/profiles.json'),
    'scripts/ci/qwen_c2_boot.py': ('@purelib/qwen_c2_boot.py',),
}


def git(*arguments):
    result = subprocess.run(['git'] + list(arguments), cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise AssertionError('git %s failed: %s' % (' '.join(arguments), result.stderr.decode('utf-8', 'replace')))
    return result.stdout.decode('utf-8', 'replace')


def git_paths(*arguments):
    return {line.strip() for line in git(*arguments).split(chr(10)) if line.strip()}


def show(commit, path):
    """The file at a commit, or None when it did not exist there."""
    result = subprocess.run(['git', 'show', '%s:%s' % (commit, path)], cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode('utf-8', 'replace') if result.returncode == 0 else None


def manifest_entries():
    return c2_overlay.read_manifest(MANIFEST)


def p8_commit():
    """The commit the C2 image's P8 base was built from: ARG BASE=qwen-fast-serving:ci-<sha>."""
    base = provenance.base_image(C2_DOCKERFILE.read_text(encoding='utf-8'))
    match = re.match(r'^qwen-fast-serving:ci-([0-9a-f]{40})$', base)
    if not match:
        raise AssertionError('the C2 base %s is not a qwen-fast-serving:ci-<commit> image' % base)
    return match.group(1)


def p8_overlay(commit):
    """scripts/ci files P8 laid over the bundle when built at commit: named in BOTH its lists."""
    tests = {PurePosixPath(path).name for path in
             git_paths('ls-tree', '--name-only', commit, '--', 'scripts/ci/')
             if re.match(r'^test_serving_.*[.]py$', PurePosixPath(path).name)}
    dockerfile = show(commit, P8_DOCKERFILE)
    workflow = show(commit, P8_WORKFLOW)
    if dockerfile is None or workflow is None:
        raise AssertionError('P8 copy lists missing at %s' % commit)
    dockerfile = dockerfile.replace(chr(13) + chr(10), chr(10))
    return {'scripts/ci/' + name for name in
            p8.dockerfile_modules(dockerfile, tests) & p8.context_modules(workflow, tests)}


def carried(path):
    return (path.startswith(TREES) and path.endswith(SUFFIXES)
            and '__pycache__' not in PurePosixPath(path).parts)


def image_layers():
    """{repo path: commit ('HEAD' for the C2 overlay)} for every repo file the C2 image carries."""
    layers = {path: p8.BUNDLE_COMMIT for path in
              git_paths('ls-tree', '-r', '--name-only', p8.BUNDLE_COMMIT, '--', *TREES) if carried(path)}
    commit = p8_commit()
    for path in p8_overlay(commit):
        layers[path] = commit
    for entry in manifest_entries():
        layers[entry.source] = 'HEAD'
    return layers


def stale_files(layers=None):
    """{path: commit} of files the image holds at a version other than HEAD's."""
    layers = image_layers() if layers is None else layers
    changed = {}
    for commit in set(layers.values()) - {'HEAD'}:
        changed[commit] = git_paths('diff', '--name-only', commit, 'HEAD', '--', *TREES)
    return {path: commit for path, commit in layers.items() if commit != 'HEAD' and path in changed[commit]}


def is_test(path):
    return PurePosixPath(path).name.startswith('test_')


def local_module(name):
    """The repo path a top-level import name resolves to in the image's trees, or None."""
    for tree in TREES:
        path = tree + name + '.py'
        if (ROOT / path).is_file():
            return path
    return None


def imports(source):
    """[(module, [names] or None for a plain import)] of every absolute import in the source."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module.split('.')[0], [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            found.extend((alias.name.split('.')[0], None) for alias in node.names)
    return found


def overlaid_python():
    return [entry.source for entry in manifest_entries() if entry.source.endswith('.py')]


def test_closure(layers):
    """Every test module the in-image run imports, transitively from the overlaid ones."""
    queue = [path for path in overlaid_python() if is_test(path)]
    seen = set(queue)
    while queue:
        path = queue.pop()
        for module, _ in imports((ROOT / path).read_text(encoding='utf-8')):
            dep = local_module(module)
            if dep and is_test(dep) and dep not in seen:
                seen.add(dep)
                queue.append(dep)
    return seen


class ImageContentTests(unittest.TestCase):
    """What the built image holds, derived from git."""

    @classmethod
    def setUpClass(cls):
        cls.layers = image_layers()
        cls.stale = stale_files(cls.layers)
        cls.allowed = set(p8.KNOWN_STALE) | set(C2_KNOWN_STALE)

    def test_the_layer_commits_are_reachable(self):
        for commit in (p8.BUNDLE_COMMIT, p8_commit()):
            with self.subTest(commit=commit):
                self.assertTrue(git('cat-file', '-t', commit).strip() == 'commit',
                                'cannot read %s - is the clone shallow?' % commit)
        self.assertIn('scripts/ci/serving_lifecycle.py', self.layers)

    def test_no_file_the_image_carries_is_silently_stale(self):
        """The C2 image's version of the ordered_cache trap: a file changed since the bundle
        (or since the P8 base's commit) that the C2 manifest does not re-lay ships OLD.
        Add it to docker/qwen-c2-overlay.txt, or to C2_KNOWN_STALE with a reason."""
        unexplained = sorted('%s (image has %s)' % (path, commit[:8]) for path, commit in self.stale.items()
                             if not is_test(path) and PurePosixPath(path).name not in self.allowed)
        self.assertEqual(unexplained, [],
                         'changed at HEAD but the C2 image ships an older version: add each to '
                         'docker/qwen-c2-overlay.txt, or to C2_KNOWN_STALE with a reason')

    def test_the_bundle_file_changed_since_77d6995a_is_caught(self):
        """The exact shape the plan asks for, stated directly: a scripts/ci or
        speculative-decoding file that differs between the bundle and HEAD, named neither
        by the manifest nor by P8's lists, is stale - and only KNOWN_STALE excuses it."""
        bundle_changed = git_paths('diff', '--name-only', p8.BUNDLE_COMMIT, 'HEAD', '--', *TREES)
        overlays = p8_overlay(p8_commit()) | {entry.source for entry in manifest_entries()}
        for path in sorted(bundle_changed):
            if path in self.layers and path not in overlays and not is_test(path):
                with self.subTest(path=path):
                    self.assertIn(path, self.stale)
                    self.assertIn(PurePosixPath(path).name, self.allowed)

    def test_c2_known_stale_entries_are_still_stale(self):
        stale_names = {PurePosixPath(path).name for path in self.stale}
        self.assertEqual(sorted(set(C2_KNOWN_STALE) - stale_names), [])

    def test_an_overlaid_module_imports_only_what_the_image_has(self):
        """A module laid over the image may import a module or a symbol the image holds only
        at an older version, or not at all (target_packed_pages.py, image v45; gdn_state_copy,
        run 35683127469). Every repo-local `from X import name` must resolve in the image."""
        report = []
        cache = {}
        for path in overlaid_python():
            for module, names in imports((ROOT / path).read_text(encoding='utf-8')):
                dep = local_module(module)
                if dep is None or names is None:
                    continue
                if dep not in self.layers:
                    report.append('%s imports %s, which the image does not carry' % (path, dep))
                    continue
                commit = self.layers[dep]
                if commit == 'HEAD' or dep not in self.stale:
                    continue
                if dep not in cache:
                    cache[dep] = p8.top_level_names(show(commit, dep))
                missing = [name for name in names if name != '*' and (cache[dep] is None or name not in cache[dep])]
                if missing:
                    report.append('%s needs %s from %s, which the image holds at %s' % (
                        path, missing, dep, commit[:8]))
        self.assertEqual(report, [], '; '.join(report))

    def test_an_overlaid_module_brings_its_sibling_kernel(self):
        """Path(__file__).with_suffix('.cpp') makes the sibling the kernel: an overlaid driver
        over the image's older kernel wedges a core instead of raising (runs 35684239068,
        35685401900)."""
        broken = []
        for path in overlaid_python():
            source = (ROOT / path).read_text(encoding='utf-8')
            kernel = path[:-3] + '.cpp'
            if any(marker in source for marker in p8.SIBLING_KERNEL) and (ROOT / kernel).is_file():
                if kernel not in self.layers or kernel in self.stale:
                    broken.append('%s is overlaid but the image holds %s at %s' % (
                        path, kernel, self.layers.get(kernel, 'no version')))
        self.assertEqual(broken, [], '; '.join(broken))

    def test_the_in_image_tests_import_current_test_modules(self):
        """The Dockerfile runs every overlaid test module inside the image; a test module one of
        them imports must be in the image at HEAD's version, or the build fails on a fixture the
        older copy lacks (the v121 PrefillWindowTests draft)."""
        report = []
        for path in sorted(test_closure(self.layers)):
            if path not in self.layers:
                report.append('%s is imported by an in-image test but the image does not carry it' % path)
            elif path in self.stale:
                report.append('%s is imported by an in-image test but the image holds it at %s' % (
                    path, self.stale[path][:8]))
        self.assertEqual(report, [], 'add each to docker/qwen-c2-overlay.txt: ' + '; '.join(report))

    def test_the_stage_1_and_2_modules_are_overlaid(self):
        """The plan's S1/S2 change these (plan 2.2 item 6); each must reach the image."""
        sources = {entry.source for entry in manifest_entries()}
        for path in ('scripts/ci/serving_worker_hook.py', 'scripts/ci/serving_lifecycle.py',
                     'scripts/ci/serving_packed_step.py', 'speculative-decoding/harness/greedy_session.py'):
            with self.subTest(path=path):
                self.assertIn(path, sources)
                self.assertEqual(self.layers[path], 'HEAD')


class ManifestTests(unittest.TestCase):
    def test_the_manifest_names_files_the_checkout_has(self):
        for entry in manifest_entries():
            with self.subTest(source=entry.source):
                self.assertTrue((ROOT / entry.source).is_file())

    def test_the_pre_manifest_overlay_still_lands_where_it_did(self):
        destinations = {entry.source: entry.destinations for entry in manifest_entries()}
        for source, wanted in sorted(PRE_MANIFEST.items()):
            with self.subTest(source=source):
                self.assertEqual(tuple(destinations.get(source, ())), wanted)

    def test_the_contract_reads_the_profiles_the_manifest_installs(self):
        import serving_c2_contract

        destinations = {entry.source: entry.destinations for entry in manifest_entries()}
        self.assertIn(serving_c2_contract.PROFILES, destinations['scripts/ci/qwen_c2_profiles.json'])
        self.assertIn('/experiment-scripts/ci', serving_c2_contract.FAST_PATHS)
        self.assertIn('/speculative-decoding/harness', serving_c2_contract.FAST_PATHS)

    def test_frozen_combined_runtime_is_never_overlaid(self):
        """Its bytes are pinned by the native cache (serving_native_install.py:57)."""
        self.assertNotIn('scripts/ci/frozen_combined_runtime.py', {e.source for e in manifest_entries()})
        with self.assertRaises(c2_overlay.ManifestError):
            c2_overlay.parse_manifest('scripts/ci/frozen_combined_runtime.py\n')
        pinned = (HERE / 'serving_native_install.py').read_text(encoding='utf-8')
        self.assertIn("digest(scripts / 'frozen_combined_runtime.py') != scratch['adapter_sha256']", pinned)

    def test_the_in_image_tests_are_the_overlaid_test_modules(self):
        self.assertEqual(c2_overlay.in_image_tests(manifest_entries()),
                         [PurePosixPath(e.source).stem for e in manifest_entries()
                          if PurePosixPath(e.source).name.startswith('test_')])
        self.assertIn('test_serving_c2_contract', c2_overlay.in_image_tests(manifest_entries()))

    def test_parse_rules(self):
        parse = c2_overlay.parse_manifest
        entries = parse('# c\nscripts/ci/a.py\nspeculative-decoding/harness/b.py  # t\n'
                        'scripts/ci/c.json /x/c.json @purelib/c.json\n')
        self.assertEqual([(e.source, e.destinations) for e in entries], [
            ('scripts/ci/a.py', ('/experiment-scripts/ci/a.py',)),
            ('speculative-decoding/harness/b.py', ('/speculative-decoding/harness/b.py',)),
            ('scripts/ci/c.json', ('/x/c.json', '@purelib/c.json'))])
        for bad in ('/scripts/ci/a.py', 'scripts/ci/../a.py', 'scripts/a.py', 'scripts/ci/',
                    'scripts/ci/a.py\nscripts/ci/a.py', 'scripts/ci/a.py x/y', 'scripts/ci/a.py @purelib/d/x.py',
                    'scripts/ci/a.py /d/a\nscripts/ci/b.py /d/a', 'scripts/ci/./a.py', '# nothing\n'):
            with self.subTest(manifest=bad):
                with self.assertRaises(c2_overlay.ManifestError):
                    parse(bad)
        self.assertEqual(c2_overlay.resolve('@purelib/x.py', '/usr/lib/p/'), '/usr/lib/p/x.py')
        self.assertEqual(c2_overlay.resolve('/a/b', '/usr'), '/a/b')


def code_lines(path):
    """The file without its comment lines."""
    return chr(10).join(line for line in path.read_text(encoding='utf-8').splitlines()
                        if not line.lstrip().startswith('#'))


def names_overlaid_file(text, source):
    name = PurePosixPath(source).name
    return re.search(r'(?<![A-Za-z0-9_.-])' + re.escape(name) + r'(?![A-Za-z0-9_])', text) is not None


class OneListTests(unittest.TestCase):
    """The manifest is the only list: nothing else may name the overlaid files."""

    def test_the_dockerfile_installs_from_the_manifest(self):
        text = code_lines(C2_DOCKERFILE)
        self.assertIn('COPY qwen-c2-overlay.txt c2_overlay.py /opt/qwen-c2/', text)
        self.assertIn('COPY overlay/ /opt/qwen-c2/overlay/', text)
        self.assertIn('c2_overlay.py install --manifest /opt/qwen-c2/qwen-c2-overlay.txt', text)
        self.assertIn('--record ' + provenance.INSTALL_RECORD, text)
        self.assertIn('c2_overlay.py tests --manifest', text)
        self.assertNotIn('/experiment-scripts/ci/;', text)
        self.assertNotIn('COPY ci/', text)
        for entry in manifest_entries():
            with self.subTest(source=entry.source):
                self.assertFalse(names_overlaid_file(text, entry.source))

    def test_the_ci_build_step_stages_from_the_manifest(self):
        text = code_lines(C2_WORKFLOW)
        stage = text.index('python3 -B scripts/ci/c2_overlay.py stage --repo . --out "$ctx"')
        closure = text.index('python3 -B -m unittest test_serving_image_copy_closure test_c2_image_overlay')
        self.assertLess(closure, stage, 'the closure tests gate the build: images come from tags, '
                                        'which the CPU suite never runs on')
        self.assertIn('fetch-depth: 0', text)
        self.assertNotIn('for file in', text)
        for entry in manifest_entries():
            with self.subTest(source=entry.source):
                self.assertFalse(names_overlaid_file(text, entry.source))

    def test_the_one_list_check_sees_a_second_list(self):
        self.assertTrue(names_overlaid_file('cp serving_lifecycle.py /x', 'scripts/ci/serving_lifecycle.py'))
        self.assertFalse(names_overlaid_file('test_serving_lifecycle.py', 'scripts/ci/serving_lifecycle.py'))

    def test_the_rig_script_checks_before_building_and_tags_after_provenance(self):
        text = BUILD_SCRIPT.read_text(encoding='utf-8')
        check = text.index('c2_overlay.py" check --context "$ctx"')
        build = text.index('docker build')
        verify = text.index('c2_image_provenance.py" "${provenance[@]}"')
        tag = text.index('docker tag "$id" "$image"')
        self.assertLess(check, build)
        self.assertLess(build, verify)
        self.assertLess(verify, tag)
        self.assertIn('--iidfile', text[build:verify])
        self.assertNotIn(' -t ', text[build:verify])
        self.assertTrue(text.startswith('#!/usr/bin/env bash'))
        self.assertIn('set -euo pipefail', text)

    def test_the_provenance_binaries_are_the_dockerfile_s(self):
        self.assertEqual(provenance.graft_pairs(C2_DOCKERFILE.read_text(encoding='utf-8')),
                         provenance.GRAFT_BINARIES)


class OverlayToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='c2-overlay-'))

    def tearDown(self):
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def test_stage_then_check_the_real_repo(self):
        out = self.tmp / 'ctx'
        entries = c2_overlay.stage(ROOT, out)
        self.assertEqual(c2_overlay.check_context(out), [])
        for entry in entries:
            self.assertEqual((out / 'overlay' / entry.source).read_bytes(), (ROOT / entry.source).read_bytes())
        for name in ('Dockerfile', 'qwen-c2-overlay.txt', 'c2_overlay.py', 'c2_image_provenance.py',
                     'graft.sha256', 'source.sha256', 'overlay.sha256'):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue((out / 'graft' / 'mlp.py').is_file())
        with self.assertRaises(c2_overlay.ManifestError):
            c2_overlay.stage(ROOT, out)

    def test_check_refuses_a_context_that_is_not_the_manifest(self):
        out = self.tmp / 'ctx'
        entries = c2_overlay.stage(ROOT, out)
        edited, removed = entries[0].source, entries[-1].source
        (out / 'overlay' / 'scripts/ci/extra.py').write_text('x = 1\n')
        victim = out / 'overlay' / edited
        victim.write_bytes(victim.read_bytes() + b'# edited\n')
        (out / 'overlay' / removed).unlink()
        problems = c2_overlay.check_context(out)
        self.assertIn('%s is in the manifest but not in overlay/' % removed, problems)
        self.assertIn('overlay/scripts/ci/extra.py is not in the manifest', problems)
        self.assertIn('overlay/%s differs from overlay.sha256' % edited, problems)
        self.assertEqual(len(problems), 3, problems)
        self.assertEqual(c2_overlay.main(['check', '--context', str(out)]), 1)

    def test_install_records_what_changed(self):
        root = self.tmp / 'overlay'
        image = self.tmp / 'image'
        for directory in (root / 'scripts/ci', root / 'speculative-decoding/harness',
                          image / 'experiment-scripts/ci', image / 'speculative-decoding/harness', image / 'site'):
            directory.mkdir(parents=True)
        (root / 'scripts/ci/a.py').write_text('a = 2\n')
        (root / 'scripts/ci/b.py').write_text('b = 1\n')
        (root / 'speculative-decoding/harness/g.py').write_text('g = 1\n')
        (image / 'experiment-scripts/ci/a.py').write_text('a = 1\n')
        (image / 'experiment-scripts/ci/b.py').write_text('b = 1\n')
        manifest = self.tmp / 'm.txt'
        manifest.write_text('scripts/ci/a.py\nscripts/ci/b.py /experiment-scripts/ci/b.py @purelib/b.py\n'
                            'speculative-decoding/harness/g.py\n')
        record = self.tmp / 'record.json'
        lines = []
        rows = c2_overlay.install(manifest, root, record, purelib='/site', log=lines.append, image_root=image)
        self.assertEqual([(row['destination'], row['state']) for row in rows], [
            ('/experiment-scripts/ci/a.py', 'changed'), ('/experiment-scripts/ci/b.py', 'same'),
            ('@purelib/b.py', 'new'), ('/speculative-decoding/harness/g.py', 'new')])
        self.assertEqual((image / 'experiment-scripts/ci/a.py').read_text(), 'a = 2\n')
        self.assertEqual((image / 'site/b.py').read_text(), 'b = 1\n')
        self.assertEqual((image / 'speculative-decoding/harness/g.py').read_text(), 'g = 1\n')
        self.assertEqual(json.loads(record.read_text())['files'], rows)
        self.assertEqual(len(lines), 4)
        self.assertIn('(was ', lines[0])
        shutil.rmtree(str(image / 'speculative-decoding'))
        with self.assertRaises(c2_overlay.ManifestError):
            c2_overlay.install(manifest, root, None, purelib='/site', log=lines.append, image_root=image)
        (root / 'scripts/ci/a.py').unlink()
        with self.assertRaises(OSError):
            c2_overlay.install(manifest, root, None, purelib='/site', log=lines.append, image_root=image)


def fake_image(context, entries, tampered=None, dropped=None, record=True):
    """Probe results for an image built faithfully from context (with optional faults)."""
    graft = {binary: c2_overlay.sha256(Path(context) / 'opgraft-K64i' / binary) for binary in ('_ttnn.so', '_ttnncpp.so')}
    strings = provenance.qwen_strings((Path(context) / 'opgraft-K64i' / '_ttnncpp.so').read_bytes())
    files = {}
    for binary, path in provenance.GRAFT_BINARIES:
        files[path] = dict(exists=True, path=path, realpath=path, sha256=graft[binary])
        if binary == '_ttnncpp.so':
            files[path]['qwen'] = strings
        files[provenance.GRAFT_IN_IMAGE + '/' + binary] = dict(files[path])
    rows = []
    for entry in entries:
        digest = c2_overlay.sha256(Path(context) / 'overlay' / entry.source)
        for destination in entry.destinations:
            sha = 'f' * 64 if destination == tampered else digest
            files[destination] = dict(exists=True, path=destination, realpath=destination, sha256=sha)
            rows.append(dict(source=entry.source, destination=destination, path=destination, state='same',
                             before=digest, after=digest))
    files[provenance.MANIFEST_IN_IMAGE] = dict(exists=True, path='', realpath='', sha256=c2_overlay.sha256(
        Path(context) / 'qwen-c2-overlay.txt'))
    base_strings = strings + list(dropped or ())
    base = {provenance.GRAFT_BINARIES[-1][1]: dict(exists=True, path='', realpath='', sha256='0' * 64,
                                                   qwen=sorted(base_strings))}
    return dict(files=files, record=dict(files=rows) if record else None, trees={}), dict(files=base)


class FakeDocker(object):
    def __init__(self, context, built, base, env, argv_override=None):
        self.context, self.built, self.base, self.environment = Path(context), built, base, env
        self.argv_override = argv_override or {}
        self.contract = provenance.load_contract(self.context / 'overlay/scripts/ci/serving_c2_contract.py')
        self.profiles = self.context / 'overlay/scripts/ci/qwen_c2_profiles.json'
        self.images = []

    def probe(self, image, request):
        self.images.append(image)
        return self.built if image == 'sha256:built' else self.base

    def env(self, image):
        return dict(self.environment)

    def boot(self, image, models, profile):
        name = profile or json.loads(self.profiles.read_text())['default']
        loaded = self.contract.load_profile(str(self.profiles), name)
        argv = self.argv_override.get(name) or provenance.expected_argv(
            self.contract, self.profiles, name, loaded['snapshots'][0])
        return 'INFO noise\n[QWEN-C2] profile %s: vLLM argv %s\n[QWEN-C2] mesh x\nusage: ...\n' % (
            name, json.dumps(argv))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='c2-provenance-'))
        self.context = self.tmp / 'ctx'
        self.entries = c2_overlay.stage(ROOT, self.context)
        graft = self.context / 'opgraft-K64i'
        graft.mkdir()
        (graft / '_ttnncpp.so').write_bytes(b'\x7fELF..QWEN_SDPA_TREE_SCRATCH_ROUNDS\x00QWEN_FAST_X\x00QWEN_A')
        (graft / '_ttnn.so').write_bytes(b'\x7fELF ttnn')
        self.env = {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1', 'QWEN_DROPPED_FLAG': '1', 'PATH': '/usr/bin'}
        self.log = []

    def tearDown(self):
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def verify(self, **faults):
        argv_override = faults.pop('argv_override', None)
        built, base = fake_image(self.context, self.entries, **faults)
        docker = FakeDocker(self.context, built, base, self.env, argv_override)
        problems, report = provenance.verify('sha256:built', self.context, '/models', docker=docker,
                                             log=self.log.append)
        return problems, docker

    def test_a_faithful_image_passes(self):
        problems, docker = self.verify()
        self.assertEqual(problems, [], problems)
        self.assertEqual(docker.images[1], provenance.base_image((self.context / 'Dockerfile').read_text()))
        self.assertTrue(any(line.startswith('[G1] (c) default: [QWEN-C2] profile general: vLLM argv')
                            for line in self.log))
        self.assertIn('[G1] PASS: 0 problem(s)', self.log)
        profiles = json.loads((self.context / 'overlay/scripts/ci/qwen_c2_profiles.json').read_text())
        for name in profiles['profiles']:
            self.assertIn("[G1] (c) %s: argv matches the context's contract and profile" % name, self.log)

    def test_a_tampered_overlay_file_fails(self):
        problems, _ = self.verify(tampered='/opt/qwen-c2/profiles.json')
        self.assertEqual(len(problems), 1)
        self.assertIn('(b) scripts/ci/qwen_c2_profiles.json -> /opt/qwen-c2/profiles.json', problems[0])

    def test_a_missing_install_record_fails(self):
        problems, _ = self.verify(record=False)
        self.assertEqual([p[:30] for p in problems], ['(b) the image has no overlay i'])

    def test_a_dropped_patch_whose_flag_the_image_sets_fails(self):
        problems, _ = self.verify(dropped=['QWEN_DROPPED_FLAG', 'QWEN_UNSET_ELSEWHERE'])
        self.assertEqual(len(problems), 2)
        for problem in problems:
            self.assertIn("['QWEN_DROPPED_FLAG']", problem)
            self.assertIn('graft-so-drops-image-patches', problem)
        self.assertTrue(any("only in the base binary: ['QWEN_DROPPED_FLAG', 'QWEN_UNSET_ELSEWHERE']" in line
                            for line in self.log))

    def test_an_accepted_base_only_flag_is_reported_not_failed(self):
        from unittest import mock

        with mock.patch.dict(provenance.ACCEPTED_BASE_ONLY, {'QWEN_DROPPED_FLAG': 'the gate ran without it'}):
            problems, _ = self.verify(dropped=['QWEN_DROPPED_FLAG'])
        self.assertEqual(problems, [])
        self.assertTrue(any('lacks QWEN_DROPPED_FLAG, accepted: the gate ran without it' in line for line in self.log))
        self.assertEqual(provenance.ACCEPTED_BASE_ONLY, {})

    def test_a_profile_env_flag_counts_as_set(self):
        self.env.pop('QWEN_DROPPED_FLAG')
        problems, _ = self.verify(dropped=['QWEN_FAST_OUTPUT_BUDGET'])
        self.assertEqual(len(problems), 2, problems)

    def test_a_launched_argv_the_contract_does_not_give_fails(self):
        contract = provenance.load_contract(self.context / 'overlay/scripts/ci/serving_c2_contract.py')
        profiles = self.context / 'overlay/scripts/ci/qwen_c2_profiles.json'
        snapshot = contract.load_profile(str(profiles), 'exact')['snapshots'][0]
        argv = provenance.expected_argv(contract, profiles, 'exact', snapshot)
        argv[argv.index('--max-model-len') + 1] = '65536'
        problems, _ = self.verify(argv_override={'exact': argv})
        self.assertEqual(len(problems), 1)
        self.assertIn('(c) exact: the image launched', problems[0])

    def test_a_boot_that_logs_no_argv_fails(self):
        docker_boot = FakeDocker.boot
        try:
            FakeDocker.boot = lambda self, image, models, profile: 'boot failed:\nTraceback ...\nValueError: x\n'
            problems, _ = self.verify()
        finally:
            FakeDocker.boot = docker_boot
        profiles = json.loads((self.context / 'overlay/scripts/ci/qwen_c2_profiles.json').read_text())
        self.assertEqual(len(problems), 1 + len(profiles['profiles']))
        self.assertIn('ValueError: x', problems[0])

    def test_the_binary_pin_must_name_the_installed_binary(self):
        self.env['QWEN_FAST_RUNTIME_BINARY_SHA256'] = 'a' * 64
        problems, _ = self.verify()
        self.assertEqual(len(problems), 1)
        self.assertIn('QWEN_FAST_RUNTIME_BINARY_SHA256', problems[0])

    def test_the_argv_check_agrees_with_a_real_boot(self):
        """Boot the real contract as the .pth hook would under `python3 -m` with the build's
        platform argv, then hold its logged line to expected_argv."""
        profiles = self.context / 'overlay/scripts/ci/qwen_c2_profiles.json'
        code = (
            'import sys\n'
            'sys.path.insert(0, %r)\n'
            'import serving_c2_contract as contract\n'
            'from unittest import mock\n'
            'sys.argv = ["-m"] + %r\n'
            'with mock.patch.object(contract, "resolve_snapshot", lambda profile: profile["snapshots"][-1]):\n'
            '    contract.boot(environ={"QWEN_C2_SERVING": "1", "QWEN_C2_PROFILES": %r},\n'
            '                  orig_argv=["python3", "-m", contract.API_SERVER] + %r)\n'
        ) % (str(self.context / 'overlay/scripts/ci'), list(provenance.BOOT_ARGS), str(profiles),
             list(provenance.BOOT_ARGS))
        contract = provenance.load_contract(self.context / 'overlay/scripts/ci/serving_c2_contract.py')
        for name in json.loads(profiles.read_text())['profiles']:
            with self.subTest(profile=name):
                env = dict(os.environ, QWEN_C2_PROFILE=name)
                env.pop('QWEN_C2_SERVING', None)
                result = subprocess.run([sys.executable, '-B', '-c', code], env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                problems, lines = provenance.check_argv(contract, profiles, name, name, result.stderr)
                self.assertEqual(problems, [])
                self.assertIn("(c) %s: argv matches the context's contract and profile" % name, lines)

    def test_the_probe_runs(self):
        """PROBE is what runs inside the image; run it here on real files."""
        (self.tmp / 'tree').mkdir()
        binary = self.tmp / 'lib.so'
        binary.write_bytes(b'\x00QWEN_B_FLAG\x00' + b'x' * 100 + b'QWEN_A1\x00')
        (self.tmp / 'tree' / 'm.py').write_text('m = 1\n')
        (self.tmp / 'tree' / 'm.txt').write_text('ignored\n')
        record = self.tmp / 'record.json'
        record.write_text('{"files": []}')
        request = dict(files=[[binary.as_posix(), True], [(self.tmp / 'absent').as_posix(), False]],
                       trees=[(self.tmp / 'tree').as_posix()], suffixes=['.py'], record=record.as_posix())
        result = subprocess.run([sys.executable, '-B', '-c', provenance.PROBE, json.dumps(request)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        probed = provenance.parse_probe('noise\n' + result.stdout)
        found = probed['files'][binary.as_posix()]
        self.assertEqual(found['sha256'], hashlib.sha256(binary.read_bytes()).hexdigest())
        self.assertEqual(found['qwen'], ['QWEN_A1', 'QWEN_B_FLAG'])
        self.assertFalse(probed['files'][(self.tmp / 'absent').as_posix()]['exists'])
        self.assertEqual([Path(path).name for path in probed['trees']], ['m.py'])
        self.assertEqual(probed['record'], {'files': []})

    def test_helpers(self):
        self.assertEqual(provenance.qwen_strings(b'..QWEN_B\x00QWEN_A_1 QWEN_B'), ['QWEN_A_1', 'QWEN_B'])
        self.assertEqual(provenance.base_image('FROM x\nARG BASE=img:tag\n'), 'img:tag')
        self.assertEqual(provenance.argv_line('[QWEN-C2] profile exact: vLLM argv ["--a", "1"]\n', 'exact'),
                         ('[QWEN-C2] profile exact: vLLM argv ["--a", "1"]', ['--a', '1']))
        self.assertEqual(provenance.argv_line('[QWEN-C2] profile exact: vLLM argv []', 'general'), (None, None))
        with self.assertRaises(ValueError):
            provenance.parse_probe('nothing')
        tree = self.tmp / 'co' / 'scripts' / 'ci'
        tree.mkdir(parents=True)
        (tree / 'same.py').write_text('s\n')
        (tree / 'diff.py').write_text('d\n')
        trees = {'/experiment-scripts/ci/same.py': c2_overlay.sha256(tree / 'same.py'),
                 '/experiment-scripts/ci/diff.py': '0' * 64, '/experiment-scripts/ci/only_image.py': '1' * 64}
        self.assertEqual(provenance.drift_lines(trees, self.tmp / 'co'), [
            '(d) image vs checkout: 1 identical, 1 differ (informational)',
            '(d) differs from the checkout: /experiment-scripts/ci/diff.py'])


if __name__ == '__main__':
    unittest.main()
