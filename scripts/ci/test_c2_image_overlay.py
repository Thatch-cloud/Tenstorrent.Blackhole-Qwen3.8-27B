"""The C2 serving image ships HEAD's version of every file it carries - or says why not.

c2_image_layers models the C2 image's fast-path trees as three layers - the bundle (modelled as
commit 77d6995a), P8's copy lists at the P8 base's commit, and docker/qwen-c2-overlay.txt at HEAD -
and a file lands at the version of the LAST layer that names it. A file changed since its layer's
commit and not re-laid by a later layer ships OLD. That is how v87 ran ordered_cache.py's old
line 54 (run 35790454545), and why S1/S2 changes to serving_worker_hook, serving_lifecycle,
serving_packed_step and greedy_session would silently not ship: they sit in layers 1-2 only.
test_serving_image_copy_closure guards P8's two lists; this module guards what the C2 image ends
up holding, and the tools that build it (c2_overlay, c2_image_provenance).

The layer model is a model: G1 (c2_image_provenance (d)) holds every built image to it, so a file
where it is wrong fails the build instead of passing here.

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
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_image_layers as layers_model  # noqa: E402
import c2_image_provenance as provenance  # noqa: E402
import c2_overlay  # noqa: E402
import c2_serving_job  # noqa: E402
import qwen_prefix_stage  # noqa: E402
import test_serving_image_copy_closure as p8  # noqa: E402

C2_DOCKERFILE = ROOT / c2_overlay.DOCKERFILE
C2_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml'
BUILD_SCRIPT = ROOT / c2_overlay.BUILD_SCRIPT
MANIFEST = ROOT / c2_overlay.MANIFEST
V235_ENVIRONMENT = ROOT / c2_overlay.V235_ENVIRONMENT
TREES = layers_model.TREES
REPO = layers_model.Repo(ROOT)

_UNREVIEWED = ('UNREVIEWED whether serving reaches the changed code; blob-pinned, so any further edit fails '
               'here and must be overlaid or reviewed')

# Files the C2 image deliberately ships at an older version than HEAD: {repo path: (HEAD blob when
# reviewed, reason)}. The blob pin is the point: an entry excuses THAT version of HEAD's file only,
# so a further edit to it - which would silently not reach the image - fails
# test_c2_known_stale_entries_pin_the_reviewed_blob until it is overlaid or re-reviewed. Keyed by
# path, never by basename, and never inherited from P8's KNOWN_STALE (whose UNREVIEWED entries
# are basename-keyed and unpinned).
C2_KNOWN_STALE = {
    'scripts/ci/feature_collective.py': (
        '70e1936e608eb450539f8c88d04a59d8de740b0b',
        'serving imports it (dflash_device, draft_attention_branch, draft_mlp_branch). HEAD adds only the '
        'observe= keyword of gather_add_projection, which P8\'s draft branches pass only under '
        'QWEN_FAST_PROPOSAL_AUDIT=1 - unset in every C2 profile - so the bundle copy serves as it served v235'),
    'scripts/ci/frozen_combined_adapters.py': (
        '567685ab1d5ff8ab1cffca2913622116b93fadd4',
        'recipe staging tool, imported only by frozen_recipe_context; HEAD adds 65536 staging (4c297e73), '
        '32768 byte-identical. ' + _UNREVIEWED),
    'scripts/ci/frozen_combined_gate.py': (
        '4375462f7f5f23670ebd37af8437e5a7318defa6',
        'imported by the image\'s frozen_combined_runtime, which is the bundle\'s copy the native cache pins: the '
        'gate must stay the copy that runtime was cut against. HEAD adds CONTEXT_REPORTS (65536 unqualified)'),
    'scripts/ci/frozen_combined_runtime.py': (
        '7b26f984e634d8b8c57a8baa30b2aeb61d6893d8',
        'never overlaid (c2_overlay FORBIDDEN): serving_native_install checks its bytes against the native '
        'cache\'s adapter_sha256 at P8 build. HEAD\'s per-context staging is not what exact runs'),
    'scripts/ci/frozen_mlp_wait_zones.py': (
        '21425fe2e7141dd225b5d5eb43f16909bfde2334',
        'wait-zone evidence tooling; HEAD re-pins the drain regeneration (d08c1140). ' + _UNREVIEWED),
    'scripts/ci/frozen_recipe_context.py': (
        'b21c97141336cc140597d13e82b6f142a62cde3f',
        'recipe staging for the 65536 lanes (43a1da0d..80023624), imported by *_stage tools and staging '
        'adapters. ' + _UNREVIEWED),
    'scripts/ci/frozen_stage_evidence.py': (
        '6ce87801f65040208effc95bca31fdcb9e80d0d3',
        'evidence-lane command line (64k tag arms, b1981bdc); no module imports it'),
    'scripts/ci/frozen_target_replay.py': (
        '8dd33932b2452ea85b38a0ebb27c95a65ccdf51c',
        '65536 replay numerics (5edded94..0d182ee5); imported by frozen_combined_gate and '
        'frozen_recipe_context. ' + _UNREVIEWED),
    'scripts/ci/lever_n_model_patch.py': (
        '1332ff19b8e1820813aa019093aaeea4d0aefaa2',
        'host-side only: imported by lever_n_m3native_patch in the graft job on the runner, never inside the '
        'container'),
    'scripts/ci/longctx_cycle_bench.py': (
        '714d89db78fa52b4793b3e5a84105cea97402db6',
        'mounted per arm at /bench/longctx_cycle_bench.py (lever_n_m3native_run_arm.sh, and c2_serving_gate.'
        'BENCH_SCRIPTS since s1/gates), not imported from the baked tree: its only importer is the harness, '
        'lever_n_m3native_gate, mounted beside it. Re-pinned at the S1 integration for the stream_once of s1/gates '
        '(a cancel through the StreamWatch)'),
    'scripts/ci/mlp_clock_samples.py': (
        'e3e5d6602c4296e7927ac5689a7139f357d96fc6',
        'clock evidence tooling; HEAD re-pins the drain regeneration (d08c1140). ' + _UNREVIEWED),
    'scripts/ci/mlp_progressive_input.py': (
        '9de70d0c5a7844d866b40eb6b36f961e6b1edfce',
        'progressive-input arm (mlp_block_stream_runtime imports it only with progressive evidence); HEAD '
        'repeats the exit drain (4d890d6a). ' + _UNREVIEWED),
    'scripts/ci/mlp_progressive_input_gate.py': (
        '9a05ef05b04c8fa599db62aa6c9dc440c55cd6e1',
        'progressive-input gate (mlp_block_stream_runtime/_request, progressive route only); HEAD moves '
        'READER_SHA256 for the drain. ' + _UNREVIEWED),
}

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
    return REPO.git(*arguments)


def show(commit, path):
    return REPO.show(commit, path)


def manifest_entries():
    return c2_overlay.read_manifest(MANIFEST)


def p8_commit():
    return layers_model.p8_commit(REPO)


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
        cls.layers = layers_model.image_layers(REPO)
        cls.stale = layers_model.stale_files(cls.layers, REPO)

    def test_the_layer_commits_are_reachable(self):
        for commit in (p8.BUNDLE_COMMIT, p8_commit(), layers_model.READ_COMBINED):
            with self.subTest(commit=commit):
                self.assertTrue(git('cat-file', '-t', commit).strip() == 'commit',
                                'cannot read %s - is the clone shallow?' % commit)
        self.assertIn('scripts/ci/serving_lifecycle.py', self.layers)

    def test_no_file_the_image_carries_is_silently_stale(self):
        """The C2 image's version of the ordered_cache trap: a file changed since the bundle
        (or since the P8 base's commit) that the C2 manifest does not re-lay ships OLD.
        Add it to docker/qwen-c2-overlay.txt, or to C2_KNOWN_STALE with its HEAD blob and a reason."""
        unexplained = sorted('%s (image has %s)' % (path, commit[:8]) for path, commit in self.stale.items()
                             if not is_test(path) and path not in C2_KNOWN_STALE)
        self.assertEqual(unexplained, [],
                         'changed at HEAD but the C2 image ships an older version: add each to '
                         'docker/qwen-c2-overlay.txt, or to C2_KNOWN_STALE with its HEAD blob and a reason')

    def test_the_bundle_file_changed_since_77d6995a_is_caught(self):
        """The exact shape the plan asks for, stated directly: a scripts/ci or
        speculative-decoding file that differs between the bundle and HEAD, named neither
        by the manifest nor by P8's lists, is stale - and only C2_KNOWN_STALE excuses it."""
        bundle_changed = REPO.paths('diff', '--name-only', p8.BUNDLE_COMMIT, 'HEAD', '--', *TREES)
        overlays = layers_model.p8_overlay(p8_commit(), REPO) | {
            entry.source for entry in manifest_entries() if layers_model.relays_tree_copy(entry)}
        for path in sorted(bundle_changed):
            if path in self.layers and path not in overlays and not is_test(path):
                with self.subTest(path=path):
                    self.assertIn(path, self.stale)
                    self.assertIn(path, C2_KNOWN_STALE)

    def test_c2_known_stale_entries_are_still_stale(self):
        """An entry since overlaid, or reverted to the image's version, must go: otherwise
        C2_KNOWN_STALE rots into a blanket exemption."""
        self.assertEqual(sorted(set(C2_KNOWN_STALE) - set(self.stale)), [])

    def test_c2_known_stale_entries_pin_the_reviewed_blob(self):
        """An entry excuses the HEAD version it was reviewed at, nothing later: a further edit
        to a stale file would not reach the image either, so it must be overlaid or re-reviewed."""
        moved = []
        for path, (blob, _) in sorted(C2_KNOWN_STALE.items()):
            now = git('rev-parse', 'HEAD:' + path).strip()
            if now != blob:
                moved.append('%s is %s at HEAD, reviewed at %s' % (path, now, blob))
        self.assertEqual(moved, [], 'changed since reviewed as stale, and the C2 image still ships its old version: '
                                    'overlay it (docker/qwen-c2-overlay.txt) or re-review it and re-pin the blob: '
                                    + '; '.join(moved))

    def test_c2_known_stale_is_its_own_review(self):
        for path, (blob, reason) in sorted(C2_KNOWN_STALE.items()):
            with self.subTest(path=path):
                self.assertTrue(path.startswith(TREES), 'keyed by repo path, not basename')
                self.assertRegex(blob, r'^[0-9a-f]{40}$')
                self.assertNotEqual(reason, p8.UNREVIEWED)
                self.assertGreater(len(reason), 40)

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

    @unittest.skipIf(sys.version_info < (3, 10), 'parses sources written for python 3.10, the image\'s')
    def test_an_older_module_imports_only_what_the_overlay_keeps(self):
        """The other direction: a module the image keeps at its bundle or P8 version may import a
        symbol from a module the overlay replaces with HEAD's - which may have renamed or dropped it."""
        head = {path for path, commit in self.layers.items() if commit == 'HEAD' and path.endswith('.py')}
        stems = {PurePosixPath(path).stem for path in head}
        older = [path for path, commit in self.layers.items() if commit != 'HEAD' and path.endswith('.py')]
        texts = layers_model.layer_sources(self.layers, older, REPO)
        report, edges, cache = [], 0, {}
        for path, text in sorted(texts.items()):
            if not any(stem in text for stem in stems):
                continue
            try:
                found = imports(text)
            except SyntaxError as error:
                report.append('%s (image version %s) does not parse: %s' % (path, self.layers[path][:8], error))
                continue
            for module, names in found:
                dep = local_module(module)
                if dep not in head or names is None:
                    continue
                edges += 1
                if dep not in cache:
                    cache[dep] = p8.top_level_names((ROOT / dep).read_text(encoding='utf-8'))
                missing = [name for name in names if name != '*' and (cache[dep] is None or name not in cache[dep])]
                if missing:
                    report.append('%s (image version %s) needs %s from %s, which the overlay replaces with HEAD\'s' % (
                        path, self.layers[path][:8], missing, dep))
        self.assertGreater(edges, 0, 'no older module imports an overlaid one - the scan has drifted')
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

    def test_an_entry_off_its_tree_path_does_not_relay_the_tree_copy(self):
        """A source listed only elsewhere (qwen_c2_boot -> @purelib) leaves the tree's copy at
        its older layer, and the model must say so."""
        self.assertFalse(layers_model.relays_tree_copy(c2_overlay.Entry('scripts/ci/x.py', ['@purelib/x.py'], 1)))
        self.assertTrue(layers_model.relays_tree_copy(c2_overlay.Entry(
            'scripts/ci/x.py', ['/experiment-scripts/ci/x.py', '/opt/x.py'], 1)))
        boot = [entry for entry in manifest_entries() if entry.source == 'scripts/ci/qwen_c2_boot.py'][0]
        self.assertNotEqual(self.layers.get(boot.source), 'HEAD')


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
        for text in ('scripts/ci/frozen_combined_runtime.py\n',
                     'scripts/ci/a.py /experiment-scripts/ci/frozen_combined_runtime.py\n',
                     'scripts/ci/a.py /opt/tt-metal/models/demos/blackhole/qwen36/tt/tp_common.py\n',
                     'scripts/ci/a.py @purelib/tp_common.py\n'):
            with self.subTest(manifest=text):
                with self.assertRaises(c2_overlay.ManifestError):
                    c2_overlay.parse_manifest(text)
        pinned = (HERE / 'serving_native_install.py').read_text(encoding='utf-8')
        self.assertIn("digest(scripts / 'frozen_combined_runtime.py') != scratch['adapter_sha256']", pinned)

    def test_build_time_only_is_what_p8_runs(self):
        """P8's Dockerfile RUNs these once; an overlay of one would change nothing."""
        self.assertEqual(set(c2_overlay.BUILD_TIME_ONLY), layers_model.p8_build_time_scripts(p8_commit(), REPO))
        for source in c2_overlay.BUILD_TIME_ONLY:
            with self.subTest(source=source):
                with self.assertRaises(c2_overlay.ManifestError) as caught:
                    c2_overlay.parse_manifest(source + '\n')
                self.assertIn('P8 rebuild or a re-apply RUN', str(caught.exception))

    def test_required_destinations_are_where_p8_s_plugin_patch_copies(self):
        """serving_plugin_patch.stage (run at P8 build) copies the policy and the registry into the
        plugin package, which P8 installs editable from /opt/qwen-fast-plugin."""
        patch = show(p8_commit(), 'scripts/ci/serving_plugin_patch.py')
        dockerfile = show(p8_commit(), layers_model.P8_DOCKERFILE)
        self.assertIn("package = Path(root) / 'src' / 'vllm_tt_plugin'", patch)
        self.assertIn("pip install --no-deps -e /opt/qwen-fast-plugin", dockerfile)
        self.assertIn('serving_plugin_patch.py /opt/qwen-fast-plugin', dockerfile)
        for source, destination in sorted(c2_overlay.REQUIRED_DESTINATIONS.items()):
            with self.subTest(source=source):
                name = PurePosixPath(destination).name
                variable = {'qwen_fast_policy.py': 'policy', 'qwen_dflash_registry.py': 'registry'}[name]
                self.assertIn("%s = package / '%s'" % (variable, name), patch)
                self.assertIn("shutil.copyfile(Path(__file__).with_name('%s'), %s)" % (
                    PurePosixPath(source).name, variable), patch)
                self.assertEqual(destination, c2_overlay.PLUGIN + name)
                self.assertTrue(c2_overlay.PLUGIN.startswith('/opt/qwen-fast-plugin/src/vllm_tt_plugin/'))
                with self.assertRaises(c2_overlay.ManifestError):
                    c2_overlay.parse_manifest(source + '\n')
                entries = c2_overlay.parse_manifest('%s %s %s\n' % (
                    source, c2_overlay.default_destination(source), destination))
                self.assertEqual(entries[0].destinations[-1], destination)

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

    def test_frozen_pins_mirror_qualify(self):
        """frozen_pins builds qualify's map; the exclusions must be the ones qualify uses, in the
        copy the image runs (the bundle's, which the native cache pins) and in HEAD's."""
        source = ast.parse(Path(c2_overlay.__file__).read_text(encoding='utf-8'))
        mine = [node for node in ast.walk(source) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'excluded' for t in node.targets)]
        self.assertEqual(len(mine), 1)
        wanted = ast.literal_eval(mine[0].value)
        for text in (show(p8.BUNDLE_COMMIT, 'scripts/ci/frozen_combined_runtime.py'),
                     (HERE / 'frozen_combined_runtime.py').read_text(encoding='utf-8')):
            theirs = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'excluded' for t in node.targets)]
            self.assertEqual([ast.literal_eval(node.value) for node in theirs], [wanted])
            self.assertIn("evidence = directory / 'frozen-evidence'", text)
            self.assertIn("reports['target-replay.json']['sources'].items()", text)
            self.assertIn("not name.endswith('-probe.py') and not name.startswith('../')", text)


def code_lines(path):
    """The file without its comment lines."""
    return '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                     if not line.lstrip().startswith('#'))


def names_overlaid_file(text, source):
    name = PurePosixPath(source).name
    return re.search(r'(?<![A-Za-z0-9_.-])' + re.escape(name) + r'(?![A-Za-z0-9_])', text) is not None


def workflow_step(text, name):
    """One step of a workflow, from its `- name:` line to the next step (comment lines dropped)."""
    start = text.index('\n      - name: %s\n' % name)
    end = text.find('\n      - name: ', start + 1)
    step = text[start:end if end != -1 else len(text)]
    return '\n'.join(line for line in step.split('\n') if not line.lstrip().startswith('#'))


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

    def test_the_dockerfile_records_its_provenance_last(self):
        """The revision label and the build stamp, after every cached layer: an ARG busts the
        cache of each RUN after it."""
        text = code_lines(C2_DOCKERFILE)
        tail = text[text.index('ARG SOURCE_REVISION'):]
        self.assertGreater(text.index('ARG SOURCE_REVISION'), text.index('QWEN_C2_SERVING=1'))
        self.assertIn('LABEL %s=${SOURCE_REVISION}' % provenance.REVISION_LABEL, tail)
        self.assertIn('COPY %s %s /opt/qwen-c2/' % (c2_overlay.REVISION_FILE, provenance.STAMP_FILE), tail)
        self.assertEqual(provenance.REVISION_IN_IMAGE, '/opt/qwen-c2/' + c2_overlay.REVISION_FILE)
        self.assertEqual(provenance.STAMP_IN_IMAGE, '/opt/qwen-c2/' + provenance.STAMP_FILE)
        self.assertEqual(text.count('ARG SOURCE_REVISION'), 1)

    def test_the_ci_build_step_stages_from_the_manifest(self):
        workflow = C2_WORKFLOW.read_text(encoding='utf-8')
        step = workflow_step(workflow, 'Build')
        stage = step.index('python3 -B scripts/ci/c2_overlay.py stage --repo . --out "$ctx"')
        closure = step.index('-B -m unittest test_serving_image_copy_closure test_c2_image_overlay')
        self.assertLess(closure, stage, 'the closure tests gate the build: images come from tags, '
                                        'which the CPU suite never runs on')
        self.assertIn('fetch-depth: 0', workflow)
        self.assertNotIn('for file in', step)
        for entry in manifest_entries():
            with self.subTest(source=entry.source):
                self.assertFalse(names_overlaid_file(step, entry.source))

    def test_the_closure_tests_run_in_the_p8_base(self):
        """Not on the rig host's python: sources written for 3.10 must parse as the image parses them."""
        step = workflow_step(C2_WORKFLOW.read_text(encoding='utf-8'), 'Build')
        self.assertIn("base=$(sed -n 's/^ARG BASE=//p' docker/qwen-c2-serving.Dockerfile)", step)
        run = step[step.index('docker run --rm --network none'):step.index('test_c2_image_overlay')]
        for token in ('--user "$(id -u):$(id -g)"', '-v "$PWD:/src:ro"', '-w /src/scripts/ci',
                      '--entrypoint python3 "$base"'):
            self.assertIn(token, run)
        dockerfile = C2_DOCKERFILE.read_text(encoding='utf-8')
        sed = [line[len('ARG BASE='):] for line in dockerfile.splitlines() if line.startswith('ARG BASE=')]
        self.assertEqual(sed, [provenance.base_image(dockerfile)])

    def test_the_drift_action_is_read_only(self):
        workflow = C2_WORKFLOW.read_text(encoding='utf-8')
        step = workflow_step(workflow, 'G1 base drift')
        # The job file's parse (c2_serving_job, since s1/gates) knows the action, in the workflow's order.
        self.assertIn('python3 scripts/ci/c2_serving_job.py .github/c2-serving-job.env',
                      workflow[:workflow.index('- name: Status')])
        self.assertLess(c2_serving_job.ACTIONS.index('reset'), c2_serving_job.ACTIONS.index('drift'))
        self.assertLess(c2_serving_job.ACTIONS.index('drift'), c2_serving_job.ACTIONS.index('build'))
        self.assertIn('--base-drift --context "$ctx"', step)
        self.assertIn('|| status=$?', step)
        for word in ('docker build', 'docker tag', 'docker push', 'docker rmi', 'build-c2-serving-image.sh',
                     '/dev/tenstorrent'):
            self.assertNotIn(word, step)
        self.assertLess(workflow.index('- name: G1 base drift'), workflow.index('- name: Build'))

    def test_the_one_list_check_sees_a_second_list(self):
        self.assertTrue(names_overlaid_file('cp serving_lifecycle.py /x', 'scripts/ci/serving_lifecycle.py'))
        self.assertFalse(names_overlaid_file('test_serving_lifecycle.py', 'scripts/ci/serving_lifecycle.py'))
        text = '\n      - name: Build\n        run: echo a\n      - name: Smoke\n        run: for file in x; do\n'
        self.assertNotIn('for file in', workflow_step(text, 'Build'))

    def test_the_rig_script_checks_before_building_and_tags_after_provenance(self):
        text = BUILD_SCRIPT.read_text(encoding='utf-8')
        guard = text.index('"$ctx"/*) echo')
        check = text.index('c2_overlay.py" check --context "$ctx"')
        own = text.index('is not the context\'s build-c2-serving-image.sh')
        stamp = text.index('> "$ctx/build-stamp"')
        trap = text.index('trap cleanup EXIT')
        build = text.index('docker build')
        verify = text.index('c2_image_provenance.py" "${provenance[@]}"')
        tag = text.index('docker tag "$built" "$image"')
        for earlier, later in ((guard, check), (check, own), (own, stamp), (stamp, build), (trap, build),
                               (build, verify), (verify, tag)):
            self.assertLess(earlier, later)
        self.assertIn('--iidfile', text[build:verify])
        self.assertIn('--build-arg "SOURCE_REVISION=$source_revision"', text[build:verify])
        self.assertNotIn(' -t ', text[build:verify])
        self.assertIn('docker rmi "$built"', text[trap - 1000:trap])
        self.assertIn('built=\n', text[tag:])
        self.assertTrue(text.startswith('#!/usr/bin/env bash'))
        self.assertIn('set -euo pipefail', text)
        self.assertIn(c2_overlay.BUILD_SCRIPT, c2_overlay.TOOLS)

    def test_the_provenance_graft_tables_are_the_dockerfile_s(self):
        text = C2_DOCKERFILE.read_text(encoding='utf-8')
        self.assertEqual(provenance.graft_pairs(text), provenance.GRAFT_BINARIES)
        self.assertEqual(provenance.graft_dirs(text), provenance.GRAFT_OP_DIRS)


class EnvironmentTests(unittest.TestCase):
    """Exact's environment is the v235 gate's, read from the gate's own record."""

    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(V235_ENVIRONMENT.read_text(encoding='utf-8'))

    def booted(self, profile_name):
        """The environment a process in the image sees under a profile: the Dockerfile's ENV,
        then the contract's own apply_environment."""
        import serving_c2_contract

        env = provenance.dockerfile_env(C2_DOCKERFILE.read_text(encoding='utf-8'))
        profile = serving_c2_contract.load_profile(str(HERE / 'qwen_c2_profiles.json'), profile_name)
        return serving_c2_contract.apply_environment(profile, env)

    def test_the_reference_is_the_gate_s_record(self):
        self.assertEqual(self.reference['run'], 36087022223)
        gate = self.reference['qwen_configuration']
        self.assertEqual(len(gate), 68)
        self.assertTrue(all(name.startswith('QWEN_') for name in gate))
        self.assertEqual(gate['QWEN_FAST_MAX_POSITION'], '131328')
        self.assertEqual(gate['QWEN_DSPARK_REQUEST_CONTEXT'], '131072')
        self.assertNotIn('QWEN_FAST_OUTPUT_BUDGET', gate)
        self.assertEqual(set(self.reference['unset_equivalents']), {'QWEN_FAST_OUTPUT_BUDGET'})

    def test_the_output_budget_equivalence_is_the_code_s_default(self):
        import serving_fast_policy

        value = self.reference['unset_equivalents']['QWEN_FAST_OUTPUT_BUDGET']['value']
        self.assertEqual(serving_fast_policy._output_budget(None), int(value))
        self.assertEqual(serving_fast_policy._output_budget(value), int(value))

    def test_the_exact_profile_boots_into_the_v235_environment(self):
        differences, equivalents = provenance.environment_differences(self.reference, self.booted('exact'))
        self.assertEqual(differences, [])
        self.assertEqual(equivalents, ["QWEN_FAST_OUTPUT_BUDGET=256 is the gate's unset default"])

    def test_a_difference_from_the_gate_is_reported(self):
        env = self.booted('exact')
        env['QWEN_FAST_SHARD_CHECK'] = '1'
        env['QWEN_NEW_FLAG'] = '1'
        del env['QWEN_SDPA_BF8']
        problems, lines = provenance.check_environment(self.reference, {'exact': env, 'general': self.booted('general')})
        self.assertEqual(sorted(problems), sorted([
            '(c) exact: QWEN_FAST_SHARD_CHECK=1, the v235 gate ran 0',
            '(c) exact: QWEN_SDPA_BF8=(unset), the v235 gate ran 1',
            '(c) exact: QWEN_NEW_FLAG=1, which the v235 gate never set']))
        self.assertTrue(any(line.startswith('(c) general: QWEN_ environment sha256') for line in lines))
        self.assertEqual(provenance.check_environment(self.reference, {})[0][0][:23], '(c) no exact profile en')


class OverlayToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='c2-overlay-'))

    def tearDown(self):
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def quick_context(self):
        """A staged context without the (slow) layer model, and a placeholder layers.json."""
        out = self.tmp / 'ctx'
        entries = c2_overlay.stage(ROOT, out, require_clean=False, layers=False)
        (out / c2_overlay.LAYERS_FILE).write_text('{}\n')
        return out, entries

    def test_stage_then_check_the_real_repo(self):
        out = self.tmp / 'ctx'
        entries = c2_overlay.stage(ROOT, out, require_clean=False)
        self.assertEqual(c2_overlay.check_context(out), [])
        for entry in entries:
            self.assertEqual((out / 'overlay' / entry.source).read_bytes(), (ROOT / entry.source).read_bytes())
        for name in ('Dockerfile', 'qwen-c2-overlay.txt', 'v235-environment.json', 'c2_overlay.py',
                     'c2_image_provenance.py', 'build-c2-serving-image.sh', 'graft.sha256', 'source.sha256',
                     'overlay.sha256', c2_overlay.REVISION_FILE, c2_overlay.LAYERS_FILE):
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue((out / 'graft' / 'mlp.py').is_file())
        head = git('rev-parse', 'HEAD').strip()
        self.assertEqual((out / c2_overlay.REVISION_FILE).read_text().strip(), head)
        tree = json.loads((out / c2_overlay.LAYERS_FILE).read_text())
        self.assertEqual(tree['commits']['head'], head)
        self.assertGreater(len(tree['files']), 1000)
        lifecycle = tree['files']['/experiment-scripts/ci/serving_lifecycle.py']
        self.assertEqual((lifecycle['layer'], lifecycle['sha256']), (head, lifecycle['head']))
        self.assertEqual(lifecycle['sha256'], c2_overlay.sha256(ROOT / 'scripts/ci/serving_lifecycle.py'))
        stale = tree['files']['/experiment-scripts/ci/frozen_combined_runtime.py']
        self.assertEqual(stale['layer'], tree['commits']['bundle'])
        self.assertNotEqual(stale['sha256'], stale['head'])
        with self.assertRaises(c2_overlay.ManifestError):
            c2_overlay.stage(ROOT, out)

    def test_stage_refuses_a_checkout_that_differs_from_head(self):
        with mock.patch.object(c2_overlay, 'unclean', return_value=['scripts/ci/serving_lifecycle.py']):
            with self.assertRaises(c2_overlay.ManifestError) as caught:
                c2_overlay.stage(ROOT, self.tmp / 'ctx', layers=False)
        self.assertIn('scripts/ci/serving_lifecycle.py', str(caught.exception))
        self.assertFalse((self.tmp / 'ctx').exists())
        staged = c2_overlay.staged_paths(manifest_entries())
        for path in (c2_overlay.DOCKERFILE, c2_overlay.MANIFEST, c2_overlay.GRAFT, c2_overlay.BUILD_SCRIPT,
                     c2_overlay.V235_ENVIRONMENT, 'scripts/ci/serving_lifecycle.py'):
            self.assertIn(path, staged)

    def test_unclean_sees_modified_and_untracked_files(self):
        repo = self.tmp / 'repo'
        repo.mkdir()
        run = lambda *a: subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t'] + list(a),
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        run('init', '-q')
        (repo / 'a.txt').write_text('a\n')
        (repo / 'd').mkdir()
        (repo / 'd' / 'b.txt').write_text('b\n')
        run('add', '.')
        run('commit', '-q', '-m', 'x')
        self.assertEqual(c2_overlay.unclean(repo, ['a.txt', 'd']), [])
        (repo / 'a.txt').write_text('changed\n')
        (repo / 'd' / 'new.txt').write_text('n\n')
        (repo / 'other.txt').write_text('o\n')
        self.assertEqual(c2_overlay.unclean(repo, ['a.txt', 'd']), ['a.txt', 'd/new.txt'])

    def test_check_refuses_a_context_that_is_not_the_manifest(self):
        out, entries = self.quick_context()
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

    def test_check_refuses_a_context_without_its_staged_records(self):
        out, _ = self.quick_context()
        self.assertEqual(c2_overlay.check_context(out), [])
        (out / c2_overlay.REVISION_FILE).write_text('main\n')
        (out / c2_overlay.LAYERS_FILE).unlink()
        (out / 'build-c2-serving-image.sh').unlink()
        self.assertEqual(sorted(c2_overlay.check_context(out)), sorted([
            "source-revision is not a commit: 'main'", 'layers.json is missing from the context',
            'build-c2-serving-image.sh is missing from the context']))

    def image_tree(self, pins=None, target_pins=None):
        """A staging tree standing in for the image, with frozen evidence pinning `pins`."""
        root = self.tmp / 'overlay'
        image = self.tmp / 'image'
        for directory in (root / 'scripts/ci', root / 'speculative-decoding/harness', image / 'experiment-scripts/ci',
                          image / 'speculative-decoding/harness', image / 'site'):
            directory.mkdir(parents=True, exist_ok=True)
        if pins is not None:
            evidence = image / 'experiment-scripts/ci/frozen-evidence'
            evidence.mkdir()
            (evidence / 'draft-numerical.json').write_text(json.dumps(dict(sources=pins)))
            (evidence / 'target-replay.json').write_text(json.dumps(dict(sources=target_pins or {})))
        return root, image

    def test_install_records_what_changed(self):
        root, image = self.image_tree()
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
        saved = json.loads(record.read_text())
        self.assertEqual(saved['files'], rows)
        self.assertFalse(saved['frozen_pins']['evidence'])
        self.assertEqual(len(lines), 5)
        self.assertIn('NO frozen-evidence', lines[0])
        self.assertIn('(was ', lines[1])
        shutil.rmtree(str(image / 'speculative-decoding'))
        with self.assertRaises(c2_overlay.ManifestError):
            c2_overlay.install(manifest, root, None, purelib='/site', log=lines.append, image_root=image)
        (root / 'scripts/ci/a.py').unlink()
        with self.assertRaises(OSError):
            c2_overlay.install(manifest, root, None, purelib='/site', log=lines.append, image_root=image)

    def test_install_refuses_a_source_that_breaks_a_frozen_pin(self):
        """Image v42: a pinned file overlaid with other bytes kills exact at its first request."""
        pinned = hashlib.sha256(b'a = 1\n').hexdigest()
        harness = hashlib.sha256(b'g = 1\n').hexdigest()
        root, image = self.image_tree(pins={'a.py': pinned, '../../x/unpinned.py': '0' * 64},
                                      target_pins={'../../speculative-decoding/harness/g.py': harness})
        (image / 'experiment-scripts/ci/a.py').write_bytes(b'a = 1\n')
        (root / 'scripts/ci/a.py').write_bytes(b'a = 2\n')
        (root / 'speculative-decoding/harness/g.py').write_bytes(b'g = 1\n')
        manifest = self.tmp / 'm.txt'
        manifest.write_text('scripts/ci/a.py\nspeculative-decoding/harness/g.py\n')
        with self.assertRaises(c2_overlay.ManifestError) as caught:
            c2_overlay.install(manifest, root, None, purelib='/site', log=lambda _: None, image_root=image)
        self.assertIn('scripts/ci/a.py -> /experiment-scripts/ci/a.py', str(caught.exception))
        self.assertEqual((image / 'experiment-scripts/ci/a.py').read_bytes(), b'a = 1\n')
        self.assertFalse((image / 'speculative-decoding/harness/g.py').exists())
        (root / 'speculative-decoding/harness/g.py').write_bytes(b'g = 2\n')
        with self.assertRaises(c2_overlay.ManifestError) as caught:
            c2_overlay.install(self.write_manifest('speculative-decoding/harness/g.py\n'), root, None,
                               purelib='/site', log=lambda _: None, image_root=image)
        self.assertIn('/speculative-decoding/harness/g.py', str(caught.exception))
        (root / 'scripts/ci/a.py').write_bytes(b'a = 1\n')
        (root / 'speculative-decoding/harness/g.py').write_bytes(b'g = 1\n')
        lines = []
        rows = c2_overlay.install(manifest, root, None, purelib='/site', log=lines.append, image_root=image)
        self.assertEqual([row['state'] for row in rows], ['same', 'new'])
        self.assertIn('2 pinned sources, 2 overlaid with their pinned bytes', lines[0])

    def write_manifest(self, text):
        manifest = self.tmp / ('m%d.txt' % len(list(self.tmp.glob('m*.txt'))))
        manifest.write_text(text)
        return manifest

    def test_frozen_pins_builds_qualify_s_map(self):
        root, image = self.image_tree(
            pins={'a.py': '1' * 64, 'b-probe.py': '2' * 64, '../up.py': '3' * 64, 'frozen_probe_evidence.py': '4' * 64,
                  'shared.py': '5' * 64},
            target_pins={'shared.py': '6' * 64, 'c-probe.py': '7' * 64, '../../speculative-decoding/harness/g.py': '8' * 64})
        (image / 'experiment-scripts/ci/a.py').write_bytes(b'a\n')
        pins = c2_overlay.frozen_pins(str(image / 'experiment-scripts/ci'), '/experiment-scripts/ci')
        self.assertTrue(pins['evidence'])
        self.assertEqual(pins['pins'], {'/experiment-scripts/ci/a.py': '1' * 64,
                                        '/experiment-scripts/ci/shared.py': '6' * 64,
                                        '/speculative-decoding/harness/g.py': '8' * 64})
        self.assertEqual(pins['live']['/experiment-scripts/ci/a.py'], hashlib.sha256(b'a\n').hexdigest())
        self.assertIsNone(pins['live']['/speculative-decoding/harness/g.py'])
        self.assertEqual(pins['problems'], ['the frozen evidence disagrees on shared source shared.py'])
        self.assertEqual(sorted(pins['reports']), ['draft-numerical.json', 'target-replay.json'])
        (image / 'experiment-scripts/ci/frozen-evidence/target-replay.json').unlink()
        broken = c2_overlay.frozen_pins(str(image / 'experiment-scripts/ci'))
        self.assertEqual(len(broken['problems']), 1)
        self.assertIn('target-replay.json', broken['problems'][0])
        self.assertFalse(c2_overlay.frozen_pins(str(self.tmp / 'nowhere'))['evidence'])

    def test_tree_digest(self):
        top = self.tmp / 'op'
        (top / 'device' / 'kernels').mkdir(parents=True)
        (top / 'a.cpp').write_text('a\n')
        (top / 'device' / 'kernels' / 'k.cpp').write_text('k\n')
        first = c2_overlay.tree_digest(str(top))
        self.assertRegex(first, r'^[0-9a-f]{64}$')
        copy = self.tmp / 'copy'
        shutil.copytree(str(top), str(copy))
        self.assertEqual(c2_overlay.tree_digest(str(copy)), first)
        (copy / 'device' / 'kernels' / 'k.cpp').write_text('K\n')
        self.assertNotEqual(c2_overlay.tree_digest(str(copy)), first)
        (copy / 'device' / 'kernels' / 'k.cpp').write_text('k\n')
        (copy / 'extra.cpp').write_text('')
        self.assertNotEqual(c2_overlay.tree_digest(str(copy)), first)
        self.assertIsNone(c2_overlay.tree_digest(str(self.tmp / 'absent')))


def context_dirs(context):
    return {op: c2_overlay.tree_digest(str(Path(context) / 'opgraft-K64i' / op)) for op, _ in provenance.GRAFT_OP_DIRS}


def faithful_prefix_record(context):
    """The prefix stage's record and the targets' shas for an image built faithfully from context: each
    target held its pin, and the stage table's modules (the context's overlaid sources) patched it."""
    context = Path(context)
    stages = [dict(target=target, module=module, function=function, path='/experiment-scripts/ci/%s.py' % module,
                   sha256=c2_overlay.sha256(context / 'overlay' / 'scripts/ci' / (module + '.py')))
              for target, module, function in qwen_prefix_stage.STAGES]
    rows, shas = {}, {}
    for name, path, pin, source in qwen_prefix_stage.TARGETS:
        patched_by = ['%s.%s' % (stage['module'], stage['function']) for stage in stages if stage['target'] == name]
        after = hashlib.sha256((pin + ''.join(patched_by)).encode()).hexdigest() if patched_by else pin
        rows[name] = dict(path=path, pin=pin, pin_source=source, before=pin, after=after, patched_by=patched_by)
        shas[path] = after
    return dict(schema=qwen_prefix_stage.SCHEMA, targets=rows, stages=stages, complete=bool(stages),
                resolved=dict(plugin=qwen_prefix_stage.PLUGIN_ROOT, models=qwen_prefix_stage.MODEL_TREE + '/models'),
                python='3.10.12'), shas


def fake_image(context, entries, tampered=None, dropped=None, record=True, ttnn_strings=None, op_dir=None,
               trees=None, pins=None, base_pins=None, prefix=True, prefix_shas=None):
    """Probe results for an image built faithfully from context (with optional faults)."""
    context = Path(context)
    graft = {binary: c2_overlay.sha256(context / 'opgraft-K64i' / binary) for binary in ('_ttnn.so', '_ttnncpp.so')}
    strings = {binary: provenance.qwen_strings((context / 'opgraft-K64i' / binary).read_bytes())
               for binary in graft}
    files = {}
    for binary, path in provenance.GRAFT_BINARIES:
        files[path] = dict(exists=True, path=path, realpath=path, sha256=graft[binary], qwen=list(strings[binary]))
        files[provenance.GRAFT_IN_IMAGE + '/' + binary] = dict(files[path])
    if ttnn_strings is not None:
        files['/opt/tt-metal/ttnn/ttnn/_ttnn.so']['qwen'] = ttnn_strings
    rows = []
    for entry in entries:
        digest = c2_overlay.sha256(context / 'overlay' / entry.source)
        for destination in entry.destinations:
            sha = 'f' * 64 if destination == tampered else digest
            files[destination] = dict(exists=True, path=destination, realpath=destination, sha256=sha)
            rows.append(dict(source=entry.source, destination=destination, path=destination, state='same',
                             before=digest, after=digest))
    for path, name in ((provenance.MANIFEST_IN_IMAGE, 'qwen-c2-overlay.txt'),
                       (provenance.REVISION_IN_IMAGE, c2_overlay.REVISION_FILE),
                       (provenance.STAMP_IN_IMAGE, provenance.STAMP_FILE)):
        files[path] = dict(exists=True, path=path, realpath=path, sha256=c2_overlay.sha256(context / name))
    dirs = {path: dict(realpath=path, sha256=context_dirs(context)[op]) for op, path in provenance.GRAFT_OP_DIRS}
    if op_dir is not None:
        dirs[op_dir]['sha256'] = '0' * 64
    layers = json.loads((context / c2_overlay.LAYERS_FILE).read_text())
    image_trees = {path: want['sha256'] for path, want in layers['files'].items()}
    image_trees['/experiment-scripts/ci/frozen-evidence/draft-numerical.json'] = '9' * 64
    image_trees.update(trees or {})
    image_trees = {path: sha for path, sha in image_trees.items() if sha is not None}
    live = image_trees['/experiment-scripts/ci/serving_lifecycle.py']
    faithful_pins = dict(evidence=True, reports={'draft-numerical.json': 'a' * 64, 'target-replay.json': 'b' * 64},
                         pins={'/experiment-scripts/ci/serving_lifecycle.py': live}, live={
                             '/experiment-scripts/ci/serving_lifecycle.py': live}, problems=[])
    base_strings = {binary: strings[binary] + list(dropped or ()) for binary in strings}
    base = {path: dict(exists=True, path=path, realpath=path, sha256='0' * 64, qwen=sorted(base_strings[binary]))
            for binary, path in provenance.GRAFT_BINARIES}
    prefix_record, target_shas = faithful_prefix_record(context)
    target_shas.update(prefix_shas or {})
    for path, sha in target_shas.items():
        files[path] = dict(exists=sha is not None, path=path, realpath=path, sha256=sha)
    for name, path, pin, _ in qwen_prefix_stage.TARGETS:
        base[path] = dict(exists=True, path=path, realpath=path, sha256=pin)
    built = dict(files=files, record=dict(files=rows) if record else None, trees=image_trees, dirs=dirs,
                 pins=faithful_pins if pins is None else pins, bundle={})
    built['prefix'] = prefix_record if prefix is True else (prefix or None)
    return built, dict(files=base, pins=faithful_pins if base_pins is None else base_pins, trees=image_trees, bundle={})


class FakeDocker(object):
    def __init__(self, context, built, base, env, argv_override=None, labels=None, environments=None):
        self.context, self.built, self.base, self.environment_ = Path(context), built, base, env
        self.argv_override = argv_override or {}
        self.labels = labels
        self.environments = environments or {}
        self.contract = provenance.load_contract(self.context / 'overlay/scripts/ci/serving_c2_contract.py')
        self.profiles = self.context / 'overlay/scripts/ci/qwen_c2_profiles.json'
        self.images = []

    def probe(self, image, request):
        self.images.append(image)
        return self.built if image == 'sha256:built' else self.base

    def config(self, image):
        labels = self.labels if self.labels is not None else {
            provenance.REVISION_LABEL: (self.context / c2_overlay.REVISION_FILE).read_text().strip()}
        return dict(env=dict(self.environment_), labels=dict(labels))

    def environment(self, image, profile):
        if profile in self.environments:
            return dict(self.environments[profile])
        loaded = self.contract.load_profile(str(self.profiles), profile)
        env = provenance.dockerfile_env((self.context / 'Dockerfile').read_text())
        return self.contract.apply_environment(loaded, env)

    def boot(self, image, models, profile):
        name = profile or json.loads(self.profiles.read_text())['default']
        loaded = self.contract.load_profile(str(self.profiles), name)
        argv = self.argv_override.get(name) or provenance.expected_argv(
            self.contract, self.profiles, name, loaded['snapshots'][0])
        return 'INFO noise\n[QWEN-C2] profile %s: vLLM argv %s\n[QWEN-C2] mesh x\nusage: ...\n' % (
            name, json.dumps(argv))


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_tmp = Path(tempfile.mkdtemp(prefix='c2-provenance-'))
        cls.staged = cls.class_tmp / 'ctx'
        cls.entries = c2_overlay.stage(ROOT, cls.staged, require_clean=False)
        graft = cls.staged / 'opgraft-K64i'
        graft.mkdir()
        (graft / '_ttnncpp.so').write_bytes(b'\x7fELF..QWEN_SDPA_TREE_SCRATCH_ROUNDS\x00QWEN_FAST_X\x00QWEN_A')
        (graft / '_ttnn.so').write_bytes(b'\x7fELF ttnn QWEN_TTNN_ONLY\x00')
        for op, _ in provenance.GRAFT_OP_DIRS:
            (graft / op / 'device').mkdir(parents=True)
            (graft / op / 'device' / (op + '.cpp')).write_text('// %s\n' % op)
        script = cls.staged / 'build-c2-serving-image.sh'
        (cls.staged / provenance.STAMP_FILE).write_text(c2_overlay.sha256(script) + '\n')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.class_tmp), ignore_errors=True)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='c2-provenance-test-'))
        self.context = self.tmp / 'ctx'
        shutil.copytree(str(self.staged), str(self.context))
        self.env = {'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1', 'QWEN_DROPPED_FLAG': '1', 'PATH': '/usr/bin'}
        self.log = []

    def tearDown(self):
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def verify(self, **faults):
        argv_override = faults.pop('argv_override', None)
        labels = faults.pop('labels', None)
        environments = faults.pop('environments', None)
        built, base = fake_image(self.context, self.entries, **faults)
        docker = FakeDocker(self.context, built, base, self.env, argv_override, labels, environments)
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
        for op, path in provenance.GRAFT_OP_DIRS:
            self.assertTrue(any(line.startswith('[G1] (a) %s is the K64i %s tree' % (path, op)) for line in self.log))
        self.assertTrue(any('(b) built by the context\'s build-c2-serving-image.sh' in line for line in self.log))
        self.assertTrue(any('(c) exact: QWEN_ environment sha256' in line and 'equal up to the unset defaults' in line
                            for line in self.log))
        self.assertTrue(any(re.search(r'\(d\) layers .*: \d+ files as modelled, \d+ at the bundle record.s bytes, \d+ never in the bundle, 0 at HEAD', line) for line in self.log))

    def test_prefix_reuse_is_on_exactly_where_the_profile_says(self):
        """(f): the launched argv and booted environment of every profile agree with its prefix switch."""
        problems, _ = self.verify()
        self.assertEqual(problems, [])
        self.assertIn('[G1] (f) general-prefix: prefix reuse ON in the launched argv (--enable-prefix-caching '
                      '--enable-chunked-prefill --no-async-scheduling) and QWEN_PREFIX_REUSE=1', self.log)
        self.assertIn('[G1] (f) general: prefix reuse off in the launched argv (--no-enable-prefix-caching) and '
                      'no QWEN_PREFIX_REUSE', self.log)
        self.assertTrue(any(line.startswith('[G1] (f) prefix-reuse stage: %d stage(s)' % len(qwen_prefix_stage.STAGES))
                            for line in self.log))

    def test_a_prefix_profile_launched_without_prefix_caching_fails(self):
        contract = provenance.load_contract(self.context / 'overlay/scripts/ci/serving_c2_contract.py')
        profiles = self.context / 'overlay/scripts/ci/qwen_c2_profiles.json'
        snapshot = contract.load_profile(str(profiles), 'general-prefix')['snapshots'][0]
        argv = provenance.expected_argv(contract, profiles, 'general-prefix', snapshot)
        argv[argv.index('--enable-prefix-caching')] = '--no-enable-prefix-caching'
        problems, _ = self.verify(argv_override={'general-prefix': argv})
        self.assertEqual(len(problems), 2, problems)
        self.assertIn('(c) general-prefix: the image launched', problems[0])
        self.assertIn("(f) general-prefix: prefix reuse is on in the profile, but the launched argv lacks "
                      "['--enable-prefix-caching'] and has ['--no-enable-prefix-caching']", problems[1])

    def test_a_switch_leaking_into_another_profile_fails(self):
        contract = provenance.load_contract(self.context / 'overlay/scripts/ci/serving_c2_contract.py')
        loaded = contract.load_profile(str(self.context / 'overlay/scripts/ci/qwen_c2_profiles.json'), 'general')
        env = contract.apply_environment(loaded, provenance.dockerfile_env((self.context / 'Dockerfile').read_text()))
        env['QWEN_PREFIX_REUSE'] = '1'
        problems, _ = self.verify(environments={'general': env})
        self.assertEqual(problems, ['(f) general: the booted environment has QWEN_PREFIX_REUSE=1 under a profile '
                                    'without prefix reuse'])

    def test_a_missing_or_foreign_prefix_stage_record_fails(self):
        problems, _ = self.verify(prefix=None)
        self.assertEqual(problems, ['(f) the image has no /opt/qwen-c2/prefix-stage.json: the prefix-reuse stage '
                                    'never ran'])
        model = qwen_prefix_stage.MODEL_ROOT + '/model.py'
        problems, _ = self.verify(prefix_shas={model: 'e' * 64})
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('(f) model/model.py: %s is %s in the image' % (model, 'e' * 64), problems[0])

    def test_base_drift_reads_the_prefix_anchors(self):
        built, base = fake_image(self.context, self.entries)
        worker = qwen_prefix_stage.PLUGIN_ROOT + '/worker.py'
        base['files'][worker] = dict(base['files'][worker], sha256='1' * 64)
        log = []
        problems, _ = provenance.base_drift(self.context, None, FakeDocker(self.context, built, base, {}), log.append)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('(f) plugin/worker.py: the base holds %s at %s, not the pinned' % (worker, '1' * 64), problems[0])
        self.assertTrue(any('(f) model/qwen36_vllm.py: the base holds the pinned' in line for line in log))

    def test_a_tampered_overlay_file_fails(self):
        problems, _ = self.verify(tampered='/opt/qwen-c2/profiles.json')
        self.assertEqual(len(problems), 1)
        self.assertIn('(b) scripts/ci/qwen_c2_profiles.json -> /opt/qwen-c2/profiles.json', problems[0])

    def test_a_missing_install_record_fails(self):
        problems, _ = self.verify(record=False)
        self.assertEqual([p[:30] for p in problems], ['(b) the image has no overlay i'])

    def test_a_dropped_patch_whose_flag_the_image_sets_fails(self):
        problems, _ = self.verify(dropped=['QWEN_DROPPED_FLAG', 'QWEN_UNSET_ELSEWHERE'])
        self.assertEqual(len(problems), 1)
        self.assertIn("['QWEN_DROPPED_FLAG']", problems[0])
        self.assertIn('graft-so-drops-image-patches', problems[0])
        self.assertTrue(any("only in the base binaries: ['QWEN_DROPPED_FLAG', 'QWEN_UNSET_ELSEWHERE']" in line
                            for line in self.log))

    def test_an_accepted_base_only_flag_is_reported_not_failed(self):
        with mock.patch.dict(provenance.ACCEPTED_BASE_ONLY, {'QWEN_DROPPED_FLAG': 'the gate ran without it'}):
            problems, _ = self.verify(dropped=['QWEN_DROPPED_FLAG'])
        self.assertEqual(problems, [])
        self.assertTrue(any('lack QWEN_DROPPED_FLAG, accepted: the gate ran without it' in line for line in self.log))
        self.assertEqual(provenance.ACCEPTED_BASE_ONLY, {})

    def test_a_profile_env_flag_counts_as_set(self):
        self.env.pop('QWEN_DROPPED_FLAG')
        problems, _ = self.verify(dropped=['QWEN_FAST_OUTPUT_BUDGET'])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('QWEN_FAST_OUTPUT_BUDGET', problems[0])

    def test_ttnn_strings_that_are_not_the_graft_s_fail(self):
        problems, _ = self.verify(ttnn_strings=['QWEN_SOMETHING_ELSE'])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("/opt/tt-metal/ttnn/ttnn/_ttnn.so QWEN_ strings differ from the graft's", problems[0])

    def test_a_wrong_op_directory_fails(self):
        path = provenance.GRAFT_OP_DIRS[2][1]
        problems, _ = self.verify(op_dir=path)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('(a) %s' % path, problems[0])
        self.assertIn('the JIT would compile other kernels', problems[0])

    def test_a_context_without_an_op_directory_fails(self):
        shutil.rmtree(str(self.context / 'opgraft-K64i' / 'sdpa'))
        built, base = fake_image(self.staged, self.entries)
        problems, _ = provenance.verify('sha256:built', self.context, '/models', log=self.log.append,
                                        docker=FakeDocker(self.context, built, base, self.env))
        self.assertEqual(problems, ['(a) the context has no opgraft-K64i/sdpa to compare '
                                    '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa with'])

    def test_a_wrong_revision_label_fails(self):
        problems, _ = self.verify(labels={provenance.REVISION_LABEL: 'be9e184e672756c8eee040f03e48dbff58e68fda'})
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("revision label is be9e184e", problems[0])

    def test_a_build_stamp_from_another_script_fails(self):
        (self.context / provenance.STAMP_FILE).write_text('0' * 64 + '\n')
        built, base = fake_image(self.context, self.entries)
        built['files'][provenance.STAMP_IN_IMAGE]['sha256'] = c2_overlay.sha256(self.staged / provenance.STAMP_FILE)
        problems, _ = provenance.verify('sha256:built', self.context, '/models', log=self.log.append,
                                        docker=FakeDocker(self.context, built, base, self.env))
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("the image's /opt/qwen-c2/build-stamp is not the context's build-stamp", problems[0])
        self.assertIn('the build ran a build script with sha256 ' + '0' * 64, problems[1])

    def test_exact_booting_into_another_environment_fails(self):
        contract = provenance.load_contract(self.context / 'overlay/scripts/ci/serving_c2_contract.py')
        loaded = contract.load_profile(str(self.context / 'overlay/scripts/ci/qwen_c2_profiles.json'), 'exact')
        env = contract.apply_environment(loaded, provenance.dockerfile_env((self.context / 'Dockerfile').read_text()))
        env['QWEN_FAST_QUAD_DRAFT'] = '0'
        problems, _ = self.verify(environments={'exact': env})
        self.assertEqual(problems, ['(c) exact: QWEN_FAST_QUAD_DRAFT=0, the v235 gate ran 1'])

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
        with mock.patch.object(FakeDocker, 'boot', lambda self, image, models, profile:
                               'boot failed:\nTraceback ...\nValueError: x\n'):
            problems, _ = self.verify()
        profiles = json.loads((self.context / 'overlay/scripts/ci/qwen_c2_profiles.json').read_text())
        self.assertEqual(len(problems), 1 + len(profiles['profiles']))
        self.assertIn('ValueError: x', problems[0])

    def test_the_binary_pin_must_name_the_installed_binary(self):
        self.env['QWEN_FAST_RUNTIME_BINARY_SHA256'] = 'a' * 64
        problems, _ = self.verify()
        self.assertEqual(len(problems), 1)
        self.assertIn('QWEN_FAST_RUNTIME_BINARY_SHA256', problems[0])

    def test_layer_drift_fails_unless_it_is_head_s_or_accepted(self):
        layers = json.loads((self.context / c2_overlay.LAYERS_FILE).read_text())
        stale = '/experiment-scripts/ci/frozen_combined_gate.py'
        modelled = '/experiment-scripts/ci/attention_replay.py'
        absent = '/experiment-scripts/ci/ordered_cache.py'
        test_module = '/experiment-scripts/ci/test_serving_lifecycle_delegate.py'
        problems, _ = self.verify(trees={stale: layers['files'][stale]['head'], modelled: 'e' * 64, absent: None,
                                         test_module: 'd' * 64})
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(any('1 test modules off the model (not gated)' in line for line in self.log))
        self.assertTrue(problems[0].startswith('(d) %s is eeeeeeeeeeeeeeee: the layer model says '
                                               'scripts/ci/attention_replay.py' % modelled), problems[0])
        self.assertTrue(problems[1].startswith('(d) %s is absent' % absent), problems[1])
        self.assertTrue(any("(d) %s holds HEAD's version" % stale in line for line in self.log))
        self.log[:] = []
        with mock.patch.dict(provenance.ACCEPTED_LAYER_DRIFT, {modelled: ('e' * 64, 'reviewed: x'),
                                                               absent: (None, 'reviewed: y')}):
            problems, _ = self.verify(trees={modelled: 'e' * 64, absent: None})
        self.assertEqual(problems, [])
        self.assertTrue(any('drift accepted: reviewed: x' in line for line in self.log))
        self.assertEqual(provenance.ACCEPTED_LAYER_DRIFT, {})

    def test_layer_drift_names_the_read_combined_version(self):
        layers = json.loads((self.context / c2_overlay.LAYERS_FILE).read_text())
        path = next(path for path, want in sorted(layers['files'].items())
                    if want['read_combined'] not in (None, want['sha256'], want['head']))
        problems, _ = self.verify(trees={path: layers['files'][path]['read_combined']})
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('[= read-combined 8c102b20]', problems[0])

    def test_layer_drift_defers_to_the_bundle_record(self):
        # Base drift v20 (run 36210232102): the git model expected 572 files the bundle archive never
        # held and 35 at git's bytes the archive's staging had rewritten. P8's bundle record decides.
        def entry(name, sha):
            return {'sha256': sha, 'head': sha, 'layer': '77d6995a', 'source': 'scripts/ci/' + name}

        ci = '/experiment-scripts/ci/'
        layers = {'commits': {}, 'files': {ci + 'staged.py': entry('staged.py', '1' * 64),
                                           ci + 'never.py': entry('never.py', '2' * 64),
                                           ci + 'stray.py': entry('stray.py', '3' * 64),
                                           ci + 'lost.py': entry('lost.py', '4' * 64)}}
        bundle = {'experiment-scripts/ci/staged.py': '9' * 64, 'experiment-scripts/ci/stray.py': '3' * 64,
                  'experiment-scripts/ci/lost.py': '4' * 64}
        trees = {ci + 'staged.py': '9' * 64, ci + 'stray.py': '8' * 64}
        problems, _, counts = provenance.check_layers(layers, trees, bundle=bundle, accepted={})
        self.assertEqual((counts['bundle'], counts['unbundled']), (1, 1))
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(problems[0].startswith('(d) %slost.py is absent' % ci), problems[0])
        self.assertTrue(problems[1].startswith('(d) %sstray.py is 8888888888888888' % ci), problems[1])
        for missing in (None, {}):
            problems, _, counts = provenance.check_layers(layers, trees, bundle=missing, accepted={})
            self.assertEqual(len(problems), 4, problems)

    def test_a_broken_frozen_pin_fails(self):
        pinned = '/experiment-scripts/ci/attention_replay.py'
        pins = dict(evidence=True, reports={'draft-numerical.json': 'a' * 64}, problems=[],
                    pins={pinned: '1' * 64}, live={pinned: '2' * 64})
        base_pins = dict(pins, live={pinned: '1' * 64})
        problems, _ = self.verify(pins=pins, base_pins=base_pins)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('(e) %s is 2222222222222222 but the frozen recipe pins 1111111111111111' % pinned, problems[0])
        self.assertIn('the P8 base holds the pinned bytes, so C2 broke it', problems[0])

    def test_an_accepted_pin_mismatch_is_reported_not_failed(self):
        pinned = '/experiment-scripts/ci/attention_replay.py'
        pins = dict(evidence=True, reports={}, problems=[], pins={pinned: '1' * 64}, live={pinned: '2' * 64})
        with mock.patch.dict(provenance.ACCEPTED_PIN_MISMATCH, {pinned: ('2' * 64, 'reviewed: z')}):
            problems, _ = self.verify(pins=pins)
        self.assertEqual(problems, [])
        self.assertTrue(any('breaks its pin, accepted: reviewed: z' in line for line in self.log))
        self.assertEqual(provenance.ACCEPTED_PIN_MISMATCH, {})

    def test_an_image_without_frozen_evidence_fails(self):
        problems, _ = self.verify(pins=dict(evidence=False, reports={}, pins={}, live={}, problems=[]))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('frozen-evidence', problems[0])

    def test_base_drift_reports_what_the_first_build_would_fail_on(self):
        layers = json.loads((self.context / c2_overlay.LAYERS_FILE).read_text())
        built, base = fake_image(self.context, self.entries, dropped=['QWEN_DROPPED_FLAG'])
        base['trees'] = dict(base['trees'])
        overlaid = '/experiment-scripts/ci/serving_lifecycle.py'
        base['trees'][overlaid] = '1' * 64
        base['trees']['/experiment-scripts/ci/attention_replay.py'] = '2' * 64
        destination = '/experiment-scripts/ci/packed_verifier.py'
        # The base breaks the lifecycle pin, but the overlay writes the pinned bytes there: no problem.
        lifecycle = c2_overlay.sha256(self.context / 'overlay/scripts/ci/serving_lifecycle.py')
        base['pins'] = dict(base['pins'], pins={destination: '3' * 64, overlaid: lifecycle},
                            live={destination: '3' * 64, overlaid: '5' * 64})
        docker = FakeDocker(self.context, built, base, {})
        log = []
        with mock.patch.object(provenance, 'dockerfile_env', return_value={'QWEN_DROPPED_FLAG': '1'}):
            problems, report = provenance.base_drift(self.context, self.context / 'opgraft-K64i', docker, log.append)
        self.assertEqual(docker.images, [provenance.base_image((self.context / 'Dockerfile').read_text())])
        self.assertEqual(len(problems), 3, problems)
        self.assertIn("the graft binaries lack ['QWEN_DROPPED_FLAG']", problems[0])
        self.assertTrue(problems[1].startswith('(d) /experiment-scripts/ci/attention_replay.py is 2222'))
        self.assertIn('over pinned %s: install refuses it' % destination, problems[2])
        self.assertEqual(report['layers']['replaced'], len([p for p in layers['files'] if p in {
            d for e in self.entries for d in e.destinations}]))

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
        """PROBE is what runs inside the image, frozen_pins and tree_digest included; run it here."""
        (self.tmp / 'tree').mkdir()
        binary = self.tmp / 'lib.so'
        binary.write_bytes(b'\x00QWEN_B_FLAG\x00' + b'x' * 100 + b'QWEN_A1\x00')
        (self.tmp / 'tree' / 'm.py').write_text('m = 1\n')
        (self.tmp / 'tree' / 'm.txt').write_text('ignored\n')
        (self.tmp / 'ci' / 'frozen-evidence').mkdir(parents=True)
        (self.tmp / 'ci' / 'm.py').write_bytes(b'm = 1\n')
        (self.tmp / 'ci' / 'frozen-evidence' / 'draft-numerical.json').write_text(json.dumps(
            dict(sources={'m.py': '1' * 64})))
        (self.tmp / 'ci' / 'frozen-evidence' / 'target-replay.json').write_text(json.dumps(dict(sources={})))
        (self.tmp / 'bundle.json').write_text(json.dumps(dict(sources={'experiment-scripts/ci/m.py': '2' * 64})))
        record = self.tmp / 'record.json'
        record.write_text('{"files": []}')
        request = dict(files=[[binary.as_posix(), True], [(self.tmp / 'absent').as_posix(), False]],
                       trees=[(self.tmp / 'tree').as_posix()], suffixes=['.py'], record=record.as_posix(),
                       dirs=[(self.tmp / 'tree').as_posix()], pins=(self.tmp / 'ci').as_posix(),
                       bundle=(self.tmp / 'bundle.json').as_posix())
        result = subprocess.run([sys.executable, '-B', '-c', provenance.PROBE, json.dumps(request)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        probed = provenance.parse_probe('noise\n' + result.stdout)
        found = probed['files'][binary.as_posix()]
        self.assertEqual(found['sha256'], hashlib.sha256(binary.read_bytes()).hexdigest())
        self.assertEqual(found['qwen'], ['QWEN_A1', 'QWEN_B_FLAG'])
        self.assertFalse(probed['files'][(self.tmp / 'absent').as_posix()]['exists'])
        self.assertEqual([Path(path).name for path in probed['trees']], ['m.py'])
        self.assertEqual(probed['dirs'][(self.tmp / 'tree').as_posix()]['sha256'],
                         c2_overlay.tree_digest(str(self.tmp / 'tree')))
        self.assertEqual(list(probed['pins']['pins'].values()), ['1' * 64])
        self.assertEqual(list(probed['pins']['live'].values()), [hashlib.sha256(b'm = 1\n').hexdigest()])
        self.assertEqual(probed['bundle'], {'experiment-scripts/ci/m.py': '2' * 64})
        self.assertEqual(probed['record'], {'files': []})

    def test_the_environment_probe_runs(self):
        env = dict(os.environ, QWEN_X='1', TT_Y='2', MESH_DEVICE='P300', OTHER='3')
        result = subprocess.run([sys.executable, '-B', '-c', provenance.ENV_PROBE], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = provenance.parse_marked(result.stdout, 'C2ENV')
        self.assertEqual((seen['QWEN_X'], seen['TT_Y'], seen['MESH_DEVICE']), ('1', '2', 'P300'))
        self.assertNotIn('OTHER', seen)

    def test_the_command_line_refuses_a_mixed_mode(self):
        for argv in (['--context', 'x'], ['--context', 'x', '--image', 'i'],
                     ['--context', 'x', '--base-drift', '--image', 'i']):
            with self.subTest(argv=argv):
                with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                    provenance.main(argv)

    def test_helpers(self):
        self.assertEqual(provenance.qwen_strings(b'..QWEN_B\x00QWEN_A_1 QWEN_B'), ['QWEN_A_1', 'QWEN_B'])
        self.assertEqual(provenance.base_image('FROM x\nARG BASE=img:tag\n'), 'img:tag')
        self.assertEqual(provenance.argv_line('[QWEN-C2] profile exact: vLLM argv ["--a", "1"]\n', 'exact'),
                         ('[QWEN-C2] profile exact: vLLM argv ["--a", "1"]', ['--a', '1']))
        self.assertEqual(provenance.argv_line('[QWEN-C2] profile exact: vLLM argv []', 'general'), (None, None))
        with self.assertRaises(ValueError):
            provenance.parse_probe('nothing')
        self.assertEqual(provenance.dockerfile_env('FROM x\nENV A=1 B=2 ' + chr(92) + '\n    C=${X}\nRUN y\nENV D=4\n'),
                         {'A': '1', 'B': '2', 'C': '${X}', 'D': '4'})
        tree = self.tmp / 'co' / 'scripts' / 'ci'
        tree.mkdir(parents=True)
        (tree / 'same.py').write_text('s\n')
        (tree / 'diff.py').write_text('d\n')
        trees = {'/experiment-scripts/ci/same.py': c2_overlay.sha256(tree / 'same.py'),
                 '/experiment-scripts/ci/diff.py': '0' * 64, '/experiment-scripts/ci/only_image.py': '1' * 64}
        self.assertEqual(provenance.drift_lines(trees, self.tmp / 'co'), [
            '(i) image vs checkout: 1 identical, 1 differ (informational; (d) is the gate)',
            '(i) differs from the checkout: /experiment-scripts/ci/diff.py'])


if __name__ == '__main__':
    unittest.main()
