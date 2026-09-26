"""A file COPYed into the image must not need a symbol the bundle cannot supply.

The image's /experiment-scripts/ci is two layers. The bundle archive is the base and is
frozen at commit 77d6995a; repo modules are then layered over the top - but that takes
TWO lists agreeing. qwen-fast-serving-image.yml assembles a build context by copying
named files into it, and the Dockerfile COPYs from that context, so a module is in the
image only when both name it. A module in the context loop alone never arrives and
nothing says so; a module in the Dockerfile alone fails the build.

probe_image_drift calls the repo layer the "thin adapter layer" and warns that
overriding one of the core modules is only safe when the repo copy is byte-identical or
a deliberate change. What neither it nor anything else checked is the seam BETWEEN the
layers: a repo module copied over the bundle may import a symbol that only exists in the
repo, from a module that was NOT copied and therefore arrives as the bundle's older copy.

That is what run 35683127469 died of, at engine init, before readiness:

    ImportError: cannot import name 'batch_enabled' from 'gdn_state_copy'
                 (/experiment-scripts/ci/gdn_state_copy.py)

gdn_device_loop_state.py is COPYed and imports batch_enabled and copy_compact_batch from
gdn_state_copy.py, which is not. Commit 7ecf980a added both symbols and the import
together and added nothing to the Dockerfile, so every fast-serving image built since
has been unable to run the state-copy batching arm. It went unnoticed for two days
because the m3native lane serves a different image that happens to carry the symbols.

The check is sound and needs no image: `git show 77d6995a:scripts/ci/<module>` is exactly
what the bundle holds, so a symbol absent there is a symbol the bundle cannot supply.
"""

import ast
from pathlib import Path
import re
import subprocess
import unittest

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DOCKERFILE = ROOT / 'docker' / 'qwen-fast-serving.Dockerfile'
WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml'

# The tree the bundle archive was built from. docs/batch-spec-tasks-2026-09-19.md
# records it as inventory sha 826ea8f0 at commit 77d6995a, and memory
# serving-image-bundle-provenance says the same. If the bundle is ever re-pinned this
# constant moves with it, and test_the_bundle_commit_is_still_the_pinned_one fails first.
BUNDLE_COMMIT = '77d6995a'
BUNDLE_INVENTORY_SHA = '826ea8f04ae12dcc7f3000f30cc4bd714320e3da8402378990ab94043ab6e0f2'


def dockerfile_text():
    return DOCKERFILE.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


def working_tree_test_serving():
    return {p.name for p in HERE.glob('test_serving_*.py')}


def dockerfile_modules(text, test_serving=None):
    """Every scripts/ci module a Dockerfile COPY line names. test_serving expands the
    test_serving_*.py glob (default: the working tree's matches); pass another tree's
    names to read the lists as some other commit had them."""
    names = set()
    for line in text.splitlines():
        if not line.startswith('COPY '):
            continue
        for token in line.split():
            match = re.match(r'^scripts/ci/([A-Za-z0-9_]+[.](?:py|cpp))$', token)
            if match:
                names.add(match.group(1))
            elif token == 'scripts/ci/test_serving_*.py':
                names.update(working_tree_test_serving() if test_serving is None else test_serving)
    if not names:
        raise AssertionError('no scripts/ci COPY lines parsed - the regex has drifted')
    return names


def context_modules(text=None, test_serving=None):
    """Every scripts/ci module the image workflow copies into the build context (default:
    the working tree's workflow; see dockerfile_modules for test_serving)."""
    if text is None:
        text = WORKFLOW.read_text(encoding='utf-8')
    text = text.replace(chr(13) + chr(10), chr(10))
    names = set()
    for match in re.finditer(r'for name in ([^;]+); do', text):
        for token in match.group(1).split():
            if re.match(r'^[A-Za-z0-9_]+[.](?:py|cpp)$', token):
                names.add(token)
    for match in re.finditer(r'cp serving-build-orchestrator/scripts/ci/(\S+)', text):
        token = match.group(1)
        if token == 'test_serving_*.py':
            names.update(working_tree_test_serving() if test_serving is None else test_serving)
        elif re.match(r'^[A-Za-z0-9_]+[.](?:py|cpp)$', token):
            names.add(token)
    if not names:
        raise AssertionError('no context copies parsed - the workflow shape has drifted')
    return names


def copied_modules(text):
    """Modules that actually reach the image: named in BOTH lists."""
    return dockerfile_modules(text) & context_modules()


def source_at_bundle(name):
    """The module as the bundle holds it, or None when it did not exist yet."""
    result = subprocess.run(['git', 'show', '%s:scripts/ci/%s' % (BUNDLE_COMMIT, name)],
                            capture_output=True, cwd=str(HERE))
    if result.returncode != 0:
        return None
    return result.stdout.decode('utf-8', 'replace')


def top_level_names(source):
    """Names a module exposes: functions, classes, assignments and re-exports."""
    if source is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split('.')[0] for a in node.names)
    return names


def unsatisfiable_imports():
    """{module: {(symbol, importer)}} the bundle cannot supply."""
    text = dockerfile_text()
    copied = copied_modules(text)
    local = {p.name for p in HERE.glob('*.py')}
    cache, missing = {}, {}
    for name in sorted(copied):
        path = HERE / name
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module and node.level == 0):
                continue
            dep = node.module.split('.')[0] + '.py'
            if dep in copied or dep not in local:
                continue
            if dep not in cache:
                cache[dep] = top_level_names(source_at_bundle(dep))
            available = cache[dep]
            for alias in node.names:
                if alias.name == '*':
                    continue
                if available is None or alias.name not in available:
                    missing.setdefault(dep, set()).add((alias.name, name))
    return missing


def uncopied_test_imports():
    """{importer: {test module}} for copied modules importing a test module neither list copies.

    The build runs unittest over the test_serving_* modules INSIDE the image, so every test
    module they import must be in the image at HEAD. One that is not arrives as the bundle's
    copy, and unsatisfiable_imports cannot see the damage: the imported name (a TestCase
    class) exists at the bundle, while the helper methods later called on it may not.
    """
    copied = copied_modules(dockerfile_text())
    local_tests = {p.name for p in HERE.glob('test_*.py')}
    found = {}
    for name in sorted(copied):
        if not name.endswith('.py'):
            continue
        try:
            tree = ast.parse((HERE / name).read_text(encoding='utf-8'))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                dep = module.split('.')[0] + '.py'
                if dep in local_tests and dep not in copied:
                    found.setdefault(name, set()).add(dep)
    return found


SIBLING_KERNEL = ('with_suffix(' + chr(34) + '.cpp' + chr(34) + ')',
                  'with_suffix(' + chr(39) + '.cpp' + chr(39) + ')')


def sibling_kernel_modules():
    """Modules that build a kernel from the .cpp sitting next to them.

    ttnn.KernelDescriptor(kernel_source=Path(__file__).with_suffix('.cpp')) means the
    sibling file IS the kernel, so the pair is atomic: copy one without the other and
    the python drives a kernel expecting different arguments.
    """
    found = []
    for path in sorted(HERE.glob('*.py')):
        try:
            source = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        if any(marker in source for marker in SIBLING_KERNEL):
            found.append(path.name)
    if not found:
        raise AssertionError('no sibling-kernel modules found - the marker has drifted')
    return found



# Modules the image deliberately ships at their BUNDLE version although HEAD has moved on.
# A module in the bundle tree, changed since BUNDLE_COMMIT and overlaid by neither copy list
# is invisible to every other check in this file - both lists 'agree' about a file neither
# names. That is how run 35790454545 (v87) ran the old ordered_cache.py on an image built
# from the commit that changed it. Every such module must be listed here with a reason.
UNREVIEWED = 'UNREVIEWED: changed at HEAD since bundle 77d6995a and overlaid by neither list, so the image ships the bundle version. Confirm whether serving imports it before relying on the HEAD behaviour.'

KNOWN_STALE = {
    'lever_n_model_patch.py': 'host-side only: imported by lever_n_m3native_patch in the graft '
                              'job on the runner, never inside the container',
    'longctx_cycle_bench.py': 'mounted per arm at /bench/longctx_cycle_bench.py '
                              '(lever_n_m3native_run_arm.sh), not imported from the baked tree',
    'feature_collective.py': UNREVIEWED,
    'frozen_combined_adapters.py': UNREVIEWED,
    'frozen_combined_gate.py': UNREVIEWED,
    'frozen_combined_runtime.py': UNREVIEWED,
    'frozen_mlp_wait_zones.py': UNREVIEWED,
    'frozen_recipe_context.py': UNREVIEWED,
    'frozen_stage_evidence.py': UNREVIEWED,
    'frozen_target_replay.py': UNREVIEWED,
    'mlp_clock_samples.py': UNREVIEWED,
    'mlp_progressive_input.py': UNREVIEWED,
    'mlp_progressive_input_gate.py': UNREVIEWED,
}


def stale_in_image():
    """Bundle modules changed since BUNDLE_COMMIT that neither copy list overlays."""
    def names(*command):
        out = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                             cwd=str(ROOT)).stdout
        return {line.rsplit('/', 1)[-1] for line in out.split(chr(10)) if line.strip()}

    bundle = names('git', 'ls-tree', '-r', '--name-only', BUNDLE_COMMIT, '--', 'scripts/ci')
    changed = names('git', 'diff', '--name-only', BUNDLE_COMMIT, 'HEAD', '--', 'scripts/ci')
    overlaid = set(dockerfile_modules(dockerfile_text())) | set(context_modules())
    return {name for name in bundle & changed
            if name.endswith(('.py', '.cpp')) and not name.startswith('test_')
            and name not in overlaid}

class CopyClosureTests(unittest.TestCase):
    def test_a_copied_module_brings_its_sibling_kernel(self):
        """The blind spot that cost runs 35684239068 and 35685401900.

        Import closure cannot see a .cpp loaded by path, so a python module copied over
        the bundle can end up driving the bundle's older kernel with a newer argument
        layout. That does not raise - it wedges a core and the next generic_op hangs.
        """
        docker = dockerfile_modules(dockerfile_text())
        context = context_modules()
        broken = []
        for name in sibling_kernel_modules():
            kernel = name[:-3] + '.cpp'
            if not (HERE / kernel).is_file():
                continue
            py_copied = name in docker or name in context
            kernel_copied = kernel in docker and kernel in context
            if py_copied and not kernel_copied:
                broken.append('%s is copied but %s is not' % (name, kernel))
        self.assertEqual(broken, [], '; '.join(broken))

    def test_gdn_state_copy_travels_as_a_pair(self):
        """Named explicitly because it is the one that broke, and because 7ecf980a
        changed both halves in the same commit - 84 lines of python, 21 of kernel."""
        docker = dockerfile_modules(dockerfile_text())
        context = context_modules()
        for name in ('gdn_state_copy.py', 'gdn_state_copy.cpp'):
            with self.subTest(file=name):
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_no_copied_module_needs_a_symbol_the_bundle_lacks(self):
        missing = unsatisfiable_imports()
        report = []
        for dep in sorted(missing):
            for symbol, importer in sorted(missing[dep]):
                report.append('%s.%s needed by %s' % (dep[:-3], symbol, importer))
        self.assertEqual(
            report, [],
            'these arrive from the frozen bundle, which cannot supply the symbol - add '
            'the module to the Dockerfile COPY list: ' + '; '.join(report))

    def test_a_copied_test_module_imports_only_copied_test_modules(self):
        """The in-image unittest step imports every test_serving_* module, so a test
        module one of them imports must be overlaid too. The first draft of the v121 fix
        (moved GDN prefill slot) had test_serving_request_factory import
        PrefillWindowTests from test_dflash_prefill_window and call plugin_sequence on it.
        test_dflash_prefill_window is in neither list, so the image got the bundle's
        copy, which has the class but not the method: AttributeError in docker build,
        twice (test_serving_lifecycle imports RequestFactoryTests). The symbol check
        above passed, because the class name exists at the bundle."""
        report = ['%s imports %s' % (importer, dep)
                  for importer, deps in sorted(uncopied_test_imports().items())
                  for dep in sorted(deps)]
        self.assertEqual(
            report, [],
            'a copied test module imports a test module the image ships at its bundle '
            'version: move the shared fixture into a copied module, or add the imported '
            'module to BOTH copy lists: ' + '; '.join(report))

    def test_the_two_modules_run_35683127469_needed_are_copied(self):
        """gdn_state_copy is the one that killed the run. test_packed_verifier was
        latent: test_serving_packed_step imports three fixtures from it and the
        test_serving_*.py glob copies that importer, but it is not in the build-time
        unittest list, so the image built clean while carrying the fault."""
        copied = copied_modules(dockerfile_text())
        for name in ('gdn_state_copy.py', 'test_packed_verifier.py'):
            with self.subTest(module=name):
                self.assertIn(name, copied)

    def test_a_copied_module_travels_with_its_importer(self):
        """The specific pairing, stated as itself: gdn_device_loop_state is COPYed and
        cannot work without the repo's gdn_state_copy."""
        copied = copied_modules(dockerfile_text())
        self.assertIn('gdn_device_loop_state.py', copied)
        self.assertIn('gdn_state_copy.py', copied)
        source = (HERE / 'gdn_device_loop_state.py').read_text(encoding='utf-8')
        self.assertIn('from gdn_state_copy import', source)

    def test_the_two_lists_agree(self):
        """A module in one list only. Context-only never reaches the image and nothing
        reports it; Dockerfile-only fails the build on a missing context file - which is
        how image v80 (run 35683705303) was going to fail, because I added
        gdn_state_copy.py to the Dockerfile and not to the context loop."""
        docker = dockerfile_modules(dockerfile_text())
        context = context_modules()
        self.assertEqual(
            sorted(docker - context), [],
            'in the Dockerfile but not copied into the build context, so docker build '
            'cannot find them')
        self.assertEqual(
            sorted(context - docker), [],
            'copied into the build context but never COPYed into the image, so they are '
            'silently absent at runtime')

    def test_both_new_modules_are_in_both_lists(self):
        for name in ('gdn_state_copy.py', 'test_packed_verifier.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(dockerfile_text()))
                self.assertIn(name, context_modules())

    def test_the_bundle_commit_is_still_the_pinned_one(self):
        """The check is only sound while BUNDLE_COMMIT is the tree the bundle was built
        from. The image workflow asserts the inventory sha, so comparing against it
        catches a re-pin that left this constant behind."""
        workflow = WORKFLOW.read_text(encoding='utf-8')
        self.assertIn(BUNDLE_INVENTORY_SHA, workflow,
                      'the image workflow no longer pins the inventory this commit '
                      'belongs to; re-check BUNDLE_COMMIT against the new bundle')

    def test_the_bundle_commit_is_reachable(self):
        """A shallow clone would make every symbol look absent and the check vacuous in
        the wrong direction - loudly, not silently."""
        self.assertIsNotNone(source_at_bundle('serving_lifecycle.py'),
                             'cannot read scripts/ci at %s - is the clone shallow?'
                             % BUNDLE_COMMIT)


    def test_no_changed_module_is_silently_stale_in_the_image(self):
        """Run 35790454545 ran ordered_cache.py's OLD line 54 on an image built from the
        commit that changed it: the file was in neither list, and every other test here
        passed. A module changed since the bundle and overlaid by neither list must be named
        in KNOWN_STALE with a reason - or added to BOTH lists."""
        unlisted = sorted(stale_in_image() - set(KNOWN_STALE))
        self.assertEqual(unlisted, [],
                         'changed since the bundle but overlaid by neither copy list, so the image '
                         'ships the OLD version: add each to BOTH the Dockerfile COPY list and the '
                         'workflow context loop, or to KNOWN_STALE with a reason')

    def test_known_stale_entries_are_still_stale(self):
        """An entry that has since been overlaid, or reverted to the bundle version, must be
        removed - otherwise KNOWN_STALE rots into a blanket exemption."""
        self.assertEqual(sorted(set(KNOWN_STALE) - stale_in_image()), [])

    def test_the_verify_t2_table_is_in_both_lists(self):
        """verify_trace_t2.RUNTIME_FILES is the one table of what the T2 cuts load at run time -
        the card-M harness mounts exactly it - and the packed windows driver loads its kernel as
        its sibling .cpp, so every entry must reach the image through BOTH lists."""
        import verify_trace_t2

        docker = dockerfile_modules(dockerfile_text())
        context = context_modules()
        self.assertIn('gdn_conv_windows_packed.py', sibling_kernel_modules())
        for name in verify_trace_t2.RUNTIME_FILES:
            with self.subTest(file=name):
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_ordered_cache_is_overlaid(self):
        """The file whose absence cost run 35790454545."""
        self.assertIn('ordered_cache.py', dockerfile_modules(dockerfile_text()))
        self.assertIn('ordered_cache.py', context_modules())

if __name__ == '__main__':
    unittest.main()
