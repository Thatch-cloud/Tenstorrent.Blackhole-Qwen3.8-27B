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


def dockerfile_modules(text):
    """Every scripts/ci module a Dockerfile COPY line names."""
    names = set()
    for line in text.splitlines():
        if not line.startswith('COPY '):
            continue
        for token in line.split():
            match = re.match(r'^scripts/ci/([A-Za-z0-9_]+[.]py)$', token)
            if match:
                names.add(match.group(1))
            elif token == 'scripts/ci/test_serving_*.py':
                names.update(p.name for p in HERE.glob('test_serving_*.py'))
    if not names:
        raise AssertionError('no scripts/ci COPY lines parsed - the regex has drifted')
    return names


def context_modules():
    """Every scripts/ci module the image workflow copies into the build context."""
    text = WORKFLOW.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))
    names = set()
    for match in re.finditer(r'for name in ([^;]+); do', text):
        for token in match.group(1).split():
            if re.match(r'^[A-Za-z0-9_]+[.]py$', token):
                names.add(token)
    for match in re.finditer(r'cp serving-build-orchestrator/scripts/ci/(\S+)', text):
        token = match.group(1)
        if token == 'test_serving_*.py':
            names.update(p.name for p in HERE.glob('test_serving_*.py'))
        elif re.match(r'^[A-Za-z0-9_]+[.]py$', token):
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


class CopyClosureTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
