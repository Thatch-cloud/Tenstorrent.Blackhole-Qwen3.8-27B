"""The C2 serving image must carry every module its overlaid modules import, at a version that has
the symbols they import.

The C2 image (docker/qwen-c2-serving.Dockerfile) is P8 (qwen-fast-serving:ci-<commit>) plus an
overlay of repo files at the commit the C2 image is built from. So /experiment-scripts/ci holds,
for each scripts/ci module, the version of the LAST layer that names it:
  1. the bundle archive, commit 77d6995a (memory serving-image-bundle-provenance);
  2. P8's two copy lists (docker/qwen-fast-serving.Dockerfile and qwen-fast-serving-image.yml), as
     they stood at P8's commit - the Dockerfile's ARG BASE, ci-be9e184e today;
  3. the C2 overlay, at HEAD: the qwen-c2-serving.yml context loop and the Dockerfile's cp list
     (both must name a file), or docker/qwen-c2-overlay.txt where that one list exists.

C2-any (plan stage S1) changed serving_request_factory.py (overlaid) together with
serving_runtime.py and serving_lifecycle.py (not overlaid) and added serving_request_quarantine.py
(in no layer). An image built that way offers a selectable c2 profile whose request factory
refuses Phase 0 because the old serving_runtime never ran the attach check, and whose old
lifecycle has neither D2 nor D4: it dies on the platform's first warmup. Nothing failed before it
reached the rig, because the P8 closure test (test_serving_image_copy_closure) guards P8's lists,
not C2's. This is the same check for the C2 overlay: an overlaid module may import a repo module
only if the image holds that module, and each symbol only if the version the image holds defines
it. Everything is read from git; no image, no device.
"""

import ast
from pathlib import Path
import re
import subprocess
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
C2_DOCKERFILE = ROOT / 'docker' / 'qwen-c2-serving.Dockerfile'
C2_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml'
# The one-list manifest (plan item 6, track s1/image). Where it exists it IS the overlay.
C2_MANIFEST = ROOT / 'docker' / 'qwen-c2-overlay.txt'
P8_DOCKERFILE = 'docker/qwen-fast-serving.Dockerfile'
P8_WORKFLOW = '.github/workflows/qwen-fast-serving-image.yml'
BUNDLE_COMMIT = '77d6995a'
TREE = '/experiment-scripts/ci/'

# What C2-any (QWEN_FAST_ANY_REQUEST, the c2 and c2-gate profiles) needs from the image: the
# request factory's Phase 0 and budget, the runtime's attach-time source check, the lifecycle's
# D2 and D4, the scheduler-side quarantine consumer, the one-fresh-prefill-per-step cap on the
# scheduler (run 36211578069), the flag, the contract and profiles, and the parsers' R21 fix M
# that the contract arms in the API server.
C2_ANY_MODULES = ('serving_request_factory.py', 'serving_runtime.py', 'serving_lifecycle.py',
                  'serving_request_quarantine.py', 'serving_prefill_admission.py', 'serving_fast_policy.py',
                  'serving_c2_contract.py', 'qwen_c2_profiles.json', 'c2_parser_rechunk.py')

FILE_NAME = re.compile(r'^[A-Za-z0-9_]+[.](?:py|json|cpp)$')


def read(path):
    return path.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


def git_show(commit, path):
    """`path` as `commit` holds it, or None when it does not exist there."""
    result = subprocess.run(['git', 'show', '%s:%s' % (commit, path)], capture_output=True, cwd=str(ROOT))
    if result.returncode != 0:
        return None
    return result.stdout.decode('utf-8', 'replace').replace(chr(13) + chr(10), chr(10))


def reachable(commit):
    result = subprocess.run(['git', 'cat-file', '-e', commit + '^{commit}'], capture_output=True, cwd=str(ROOT))
    return result.returncode == 0


def p8_commit(dockerfile_text=None):
    """P8's commit, from the C2 Dockerfile's ARG BASE (qwen-fast-serving:ci-<commit>)."""
    text = read(C2_DOCKERFILE) if dockerfile_text is None else dockerfile_text
    found = re.findall(r'ARG BASE=qwen-fast-serving:ci-([0-9a-f]{7,40})', text)
    if len(found) != 1:
        raise AssertionError('one ARG BASE=qwen-fast-serving:ci-<commit> expected in the C2 Dockerfile, got %r' % found)
    return found[0]


def p8_dockerfile_modules(text):
    """scripts/ci modules a P8 Dockerfile COPY line names (test_serving_image_copy_closure's rule)."""
    names = set()
    for line in text.splitlines():
        if not line.startswith('COPY '):
            continue
        for token in line.split():
            match = re.match(r'^scripts/ci/([A-Za-z0-9_]+[.](?:py|cpp))$', token)
            if match:
                names.add(match.group(1))
            elif token == 'scripts/ci/test_serving_*.py':
                names.update(p.name for p in HERE.glob('test_serving_*.py'))
    return names


def p8_context_modules(text):
    """scripts/ci modules P8's image workflow copies into its build context."""
    names = set()
    for match in re.finditer(r'for name in ([^;]+); do', text):
        names.update(token for token in match.group(1).split() if FILE_NAME.match(token))
    for match in re.finditer(r'cp serving-build-orchestrator/scripts/ci/(\S+)', text):
        token = match.group(1)
        if token == 'test_serving_*.py':
            names.update(p.name for p in HERE.glob('test_serving_*.py'))
        elif FILE_NAME.match(token):
            names.add(token)
    return names


def p8_overlay(commit):
    """What P8's two lists both named at `commit`: the modules P8 laid over the bundle."""
    docker, workflow = git_show(commit, P8_DOCKERFILE), git_show(commit, P8_WORKFLOW)
    if docker is None or workflow is None:
        raise AssertionError('P8 copy lists missing at %s' % commit)
    overlay = p8_dockerfile_modules(docker) & p8_context_modules(workflow)
    if not overlay:
        raise AssertionError('no P8 overlay parsed at %s - the list shapes have drifted' % commit)
    return overlay


def c2_workflow_modules(text):
    """The qwen-c2-serving.yml build step's context loop: for file in ...; do cp "scripts/ci/$file" ..."""
    match = re.search(r'for file in ([^;]+); do\s+cp "scripts/ci/[$]file"', text)
    if match is None:
        raise AssertionError('no C2 context loop parsed in qwen-c2-serving.yml - the workflow shape has drifted')
    return {token for token in match.group(1).split() if FILE_NAME.match(token)}


def c2_dockerfile_modules(text):
    """The C2 Dockerfile's `cp <files> /experiment-scripts/ci/;` out of the build context."""
    match = re.search(r'cp ([^;]+?) /experiment-scripts/ci/;', text)
    if match is None:
        raise AssertionError('no cp ... /experiment-scripts/ci/ parsed in the C2 Dockerfile - its shape has drifted')
    return {token for token in match.group(1).split() if FILE_NAME.match(token)}


def manifest_modules(text):
    """scripts/ci files docker/qwen-c2-overlay.txt lays into /experiment-scripts/ci: a SOURCE with no
    DESTINATION replaces the tree's copy, one with DESTINATIONs lands only where they say."""
    names = set()
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        source, *destinations = line.split()
        if not source.startswith('scripts/ci/'):
            continue
        name = source[len('scripts/ci/'):]
        if not destinations or TREE + name in destinations:
            names.add(name)
    return names


def c2_overlay():
    """The scripts/ci files the C2 image lays over /experiment-scripts/ci at HEAD."""
    if C2_MANIFEST.is_file():
        return manifest_modules(read(C2_MANIFEST))
    dockerfile = read(C2_DOCKERFILE)
    context, docker = c2_workflow_modules(read(C2_WORKFLOW)), c2_dockerfile_modules(dockerfile)
    # The Dockerfile's tree list is what lands; each must be staged (else the build fails), and a
    # staged file the Dockerfile never names (qwen_c2_boot.py goes to site-packages) never arrives.
    unstaged = sorted(docker - context)
    unused = sorted(name for name in context if name not in dockerfile)
    if unstaged or unused:
        raise AssertionError('the C2 context loop and the Dockerfile disagree: copied but never staged %s, staged '
                             'but never copied %s' % (unstaged, unused))
    return docker


def top_level_names(source):
    """Names a module defines or re-exports at top level (test_serving_image_copy_closure's rule)."""
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


class ImageView(object):
    """The version of each scripts/ci module /experiment-scripts/ci holds in the C2 image."""

    def __init__(self, overlay, p8, p8_modules):
        self.overlay, self.p8, self.p8_modules, self.cache = overlay, p8, p8_modules, {}

    def layer(self, name):
        if name in self.overlay:
            return 'C2 overlay (HEAD)'
        if name in self.p8_modules:
            return 'P8 %s' % self.p8
        return 'bundle %s' % BUNDLE_COMMIT

    def source(self, name):
        if name not in self.cache:
            if name in self.overlay:
                path = HERE / name
                self.cache[name] = read(path) if path.is_file() else None
            elif name in self.p8_modules:
                self.cache[name] = git_show(self.p8, 'scripts/ci/' + name)
            else:
                self.cache[name] = git_show(BUNDLE_COMMIT, 'scripts/ci/' + name)
        return self.cache[name]


def unsatisfied_imports(view, modules):
    """[(importer, module, symbol or None, layer)] an overlaid module needs and the image lacks: a repo
    module the image does not hold at all (symbol None), or a symbol its version does not define."""
    local = {p.name for p in HERE.glob('*.py')}
    missing = []
    for importer in sorted(modules):
        if not importer.endswith('.py'):
            continue
        tree = ast.parse(read(HERE / importer))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                wanted = [(node.module.split('.')[0] + '.py', [alias.name for alias in node.names])]
            elif isinstance(node, ast.Import):
                wanted = [(alias.name.split('.')[0] + '.py', []) for alias in node.names]
            else:
                continue
            for dependency, symbols in wanted:
                if dependency not in local or dependency == importer:
                    continue
                available = top_level_names(view.source(dependency))
                if available is None:
                    missing.append((importer, dependency, None, view.layer(dependency)))
                    continue
                for symbol in symbols:
                    if symbol != '*' and symbol not in available:
                        missing.append((importer, dependency, symbol, view.layer(dependency)))
    return sorted(set(missing), key=repr)


class C2OverlayClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p8 = p8_commit()
        for commit in (cls.p8, BUNDLE_COMMIT):
            if not reachable(commit):
                raise AssertionError('commit %s is unreachable (a shallow clone?): the closure cannot be checked, '
                                     'and must not pass vacuously' % commit)
        cls.overlay = c2_overlay()
        cls.view = ImageView(cls.overlay, cls.p8, p8_overlay(cls.p8))

    def test_the_c2_any_modules_are_overlaid(self):
        """Every half of C2-any reaches the image, or a selectable c2 profile dies on its first warmup."""
        self.assertEqual(sorted(set(C2_ANY_MODULES) - self.overlay), [])

    def test_an_overlaid_module_imports_only_what_the_image_holds(self):
        """The symbol-level closure (the seam that cost run 35683127469 on P8), for C2's overlay."""
        missing = unsatisfied_imports(self.view, self.overlay)
        self.assertEqual(missing, [], '\n'.join(
            '%s imports %s from %s, which the image holds at %s' % (
                importer, 'the module' if symbol is None else repr(symbol), dependency, layer)
            for importer, dependency, symbol, layer in missing))

    def test_the_check_sees_a_module_the_image_lacks_and_a_symbol_it_predates(self):
        """Positive controls, against the real layers: the quarantine module exists in no older layer,
        and attach_source_check is newer than every older copy of serving_request_factory."""
        older = ImageView(set(), self.p8, self.view.p8_modules)
        self.assertIsNone(older.source('serving_request_quarantine.py'))
        self.assertNotIn('attach_source_check', top_level_names(older.source('serving_request_factory.py')))
        self.assertIn('attach_source_check', top_level_names(self.view.source('serving_request_factory.py')))
        # serving_runtime at HEAD, laid over an image that does not overlay the factory
        found = unsatisfied_imports(ImageView({'serving_runtime.py'}, self.p8, self.view.p8_modules),
                                    {'serving_runtime.py'})
        self.assertIn(('serving_runtime.py', 'serving_request_factory.py', 'attach_source_check',
                       'P8 %s' % self.p8 if 'serving_request_factory.py' in self.view.p8_modules
                       else 'bundle %s' % BUNDLE_COMMIT), found)

    def test_the_manifest_rule_reads_destinations(self):
        text = chr(10).join((
            '# comment',
            'scripts/ci/a.py',
            'scripts/ci/b.json  /experiment-scripts/ci/b.json  /opt/qwen-c2/profiles.json',
            'scripts/ci/c.py  @purelib/c.py',
            'speculative-decoding/harness/d.py',
        ))
        self.assertEqual(manifest_modules(text), {'a.py', 'b.json'})


if __name__ == '__main__':
    unittest.main()
