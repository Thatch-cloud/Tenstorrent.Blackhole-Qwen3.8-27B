"""The C2 serving image's layer model, from git: which version of each repo file the image holds.

The C2 image (docker/qwen-c2-serving.Dockerfile) holds the fast path's trees in three layers,
and a file lands at the version of the LAST layer that names it:

  1. the bundle archive, modelled as commit 77d6995a (test_serving_image_copy_closure
     BUNDLE_COMMIT): /experiment-scripts/ci and /speculative-decoding;
  2. P8's copy lists (docker/qwen-fast-serving.Dockerfile + qwen-fast-serving-image.yml), at the
     commit the P8 base was built from - the C2 Dockerfile's ARG BASE, qwen-fast-serving:ci-<sha>;
  3. docker/qwen-c2-overlay.txt, at the commit the C2 image is built from (HEAD) - for a file
     whose destinations include its default place in the tree.

Layer 1 is a MODEL. The bundle was cut by serving_bundle.py from a runner workspace
(read-combined, whose recipe REVISION is 8c102b20) with staged adaptations, plus serving_*.py and
dflash_combined_request.py from 77d6995a - so it need not equal `git show 77d6995a` file for
file (docs/t16-recipe-rung-65k.md section 3). test_c2_image_overlay reasons from this model;
G1 (c2_image_provenance check (d)) holds every built image to the expected_tree this module
writes into the build context, so a file where the model is wrong fails the build instead of
hiding a stale file.

Stdlib only, python >= 3.6: stage runs it on the rig host from the checkout.
"""

import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_overlay  # noqa: E402
import test_serving_image_copy_closure as p8  # noqa: E402

TREES = ('scripts/ci/', 'speculative-decoding/harness/')
SUFFIXES = ('.py', '.cpp', '.h', '.hpp', '.json', '.sh')
P8_DOCKERFILE = 'docker/qwen-fast-serving.Dockerfile'
P8_WORKFLOW = '.github/workflows/qwen-fast-serving-image.yml'
# Diagnostic only: the read-combined workspace's recipe revision (docs/batch-spec-tasks-2026-09-19.md
# "REVISION unchanged at 8c102b20"). expected_tree records each file's blob there too, so a G1
# layer failure says at once whether the image holds that version instead.
READ_COMBINED = '8c102b20'


class Repo(object):
    """git, read-only, in one checkout."""

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else HERE.parent.parent

    def git(self, *arguments):
        result = subprocess.run(['git', '-C', str(self.root), '--no-optional-locks'] + list(arguments),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise AssertionError('git %s failed: %s' % (' '.join(arguments),
                                                        result.stderr.decode('utf-8', 'replace')))
        return result.stdout.decode('utf-8', 'replace')

    def paths(self, *arguments):
        return {line.strip() for line in self.git(*arguments).split('\n') if line.strip()}

    def show(self, commit, path):
        """The file at a commit as text, or None when it did not exist there."""
        result = subprocess.run(['git', '-C', str(self.root), 'show', '%s:%s' % (commit, path)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8', 'replace') if result.returncode == 0 else None

    def commit(self, name):
        return self.git('rev-parse', '--verify', name + '^{commit}').strip()

    def blobs(self, commit, trees=TREES):
        """{path: blob id} of the regular files under trees at commit (no links, no submodules)."""
        found = {}
        for line in self.git('ls-tree', '-r', '--full-tree', commit, '--', *trees).split('\n'):
            meta, _, path = line.partition('\t')
            fields = meta.split()
            if len(fields) == 3 and fields[1] == 'blob' and fields[0] in ('100644', '100755'):
                found[path] = fields[2]
        return found

    def read(self, objects):
        """{object id: bytes} through one `git cat-file --batch`."""
        objects = sorted(set(objects))
        if not objects:
            return {}
        result = subprocess.run(['git', '-C', str(self.root), 'cat-file', '--batch'],
                                input=''.join(name + '\n' for name in objects).encode('ascii'),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise AssertionError('git cat-file --batch: ' + result.stderr.decode('utf-8', 'replace'))
        data, offset, found = result.stdout, 0, {}
        for name in objects:
            end = data.index(b'\n', offset)
            header = data[offset:end].decode('ascii').split()
            if len(header) != 3 or header[1] == 'missing':
                raise AssertionError('git cat-file: %s is missing' % name)
            size = int(header[2])
            found[name] = data[end + 1:end + 1 + size]
            offset = end + 1 + size + 1
        return found


def p8_commit(repo=None):
    """The commit the C2 image's P8 base was built from: ARG BASE=qwen-fast-serving:ci-<sha>."""
    repo = repo or Repo()
    text = (repo.root / c2_overlay.DOCKERFILE).read_text(encoding='utf-8')
    match = re.search(r'^ARG BASE=qwen-fast-serving:ci-([0-9a-f]{40})\s*$', text, re.MULTILINE)
    if not match:
        raise AssertionError('the C2 base in %s is not a qwen-fast-serving:ci-<commit> image' % c2_overlay.DOCKERFILE)
    return match.group(1)


def p8_overlay(commit, repo=None):
    """scripts/ci files P8 laid over the bundle when built at commit: named in BOTH its lists."""
    repo = repo or Repo()
    tests = {PurePosixPath(path).name for path in
             repo.paths('ls-tree', '--name-only', commit, '--', 'scripts/ci/')
             if re.match(r'^test_serving_.*[.]py$', PurePosixPath(path).name)}
    dockerfile = repo.show(commit, P8_DOCKERFILE)
    workflow = repo.show(commit, P8_WORKFLOW)
    if dockerfile is None or workflow is None:
        raise AssertionError('P8 copy lists missing at %s' % commit)
    dockerfile = dockerfile.replace('\r\n', '\n')
    return {'scripts/ci/' + name for name in
            p8.dockerfile_modules(dockerfile, tests) & p8.context_modules(workflow, tests)}


def p8_build_time_scripts(commit, repo=None):
    """scripts/ci files P8's Dockerfile RUNs at build time (their effect is baked into P8)."""
    repo = repo or Repo()
    text = (repo.show(commit, P8_DOCKERFILE) or '').replace('\r\n', '\n').replace('\\\n', ' ')
    return {'scripts/ci/' + name for line in text.split('\n') if line.startswith('RUN ')
            for name in re.findall(r'python3 (?:-B )?/experiment-scripts/ci/([A-Za-z0-9_]+[.]py)', line)}


def carried(path):
    return (path.startswith(TREES) and path.endswith(SUFFIXES)
            and '__pycache__' not in PurePosixPath(path).parts)


def manifest_entries(repo=None):
    repo = repo or Repo()
    return c2_overlay.read_manifest(repo.root / c2_overlay.MANIFEST)


def relays_tree_copy(entry):
    """True when a manifest entry replaces the image's tree copy of its source."""
    return c2_overlay.default_destination(entry.source) in entry.destinations


def image_layers(repo=None):
    """{repo path: commit ('HEAD' for the C2 overlay)} for every repo file the C2 image carries."""
    repo = repo or Repo()
    layers = {path: p8.BUNDLE_COMMIT for path in
              repo.paths('ls-tree', '-r', '--name-only', p8.BUNDLE_COMMIT, '--', *TREES) if carried(path)}
    commit = p8_commit(repo)
    for path in p8_overlay(commit, repo):
        layers[path] = commit
    for entry in manifest_entries(repo):
        if relays_tree_copy(entry):
            layers[entry.source] = 'HEAD'
    return layers


def stale_files(layers=None, repo=None):
    """{path: commit} of files the image holds at a version other than HEAD's."""
    repo = repo or Repo()
    layers = image_layers(repo) if layers is None else layers
    changed = {}
    for commit in set(layers.values()) - {'HEAD'}:
        changed[commit] = repo.paths('diff', '--name-only', commit, 'HEAD', '--', *TREES)
    return {path: commit for path, commit in layers.items() if commit != 'HEAD' and path in changed[commit]}


def layer_sources(layers, paths, repo=None):
    """{path: text} of each path as the image holds it under the model (its layer's version)."""
    repo = repo or Repo()
    wanted = {}
    for commit in {layers[path] for path in paths}:
        blobs = repo.blobs(commit)
        for path in paths:
            if layers[path] == commit and path in blobs:
                wanted[path] = blobs[path]
    data = repo.read(wanted.values())
    return {path: data[blob].decode('utf-8', 'replace') for path, blob in wanted.items()}


def expected_tree(repo=None):
    """What G1 holds a built image's trees to (written to the context as layers.json).

    {'commits': {bundle, p8, head, read_combined: full sha}, 'files': {image path: {source,
    layer (full sha), sha256 (the modelled version's), head (HEAD's sha256 or None),
    read_combined (the READ_COMBINED blob's sha256 or None)}}}."""
    repo = repo or Repo()
    layers = image_layers(repo)
    commits = dict(bundle=repo.commit(p8.BUNDLE_COMMIT), p8=repo.commit(p8_commit(repo)),
                   head=repo.commit('HEAD'), read_combined=repo.commit(READ_COMBINED))
    names = {p8.BUNDLE_COMMIT: commits['bundle'], p8_commit(repo): commits['p8'], 'HEAD': commits['head']}
    blobs = {key: repo.blobs(value) for key, value in commits.items()}
    data = repo.read(tree[path] for tree in blobs.values() for path in layers if path in tree)
    digests = {blob: hashlib.sha256(payload).hexdigest() for blob, payload in data.items()}

    def at(key, path):
        blob = blobs[key].get(path)
        return digests[blob] if blob is not None else None

    files = {}
    for path, layer in sorted(layers.items()):
        commit = names[layer]
        key = next(name for name, value in commits.items() if value == commit)
        files[c2_overlay.default_destination(path)] = dict(
            source=path, layer=commit, sha256=at(key, path), head=at('head', path),
            read_combined=at('read_combined', path))
    return dict(commits=commits, files=files)
