"""Reconstruct pinned component source snapshots without executing simulator code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import subprocess
import tempfile

from frozen_combined_gate import REPORTS, qualify


BASE = '8c102b20df22329106955b4006bf4d650bb94e40'
REVISIONS = {'draft': 'f080c1c', 'target': '09802df'}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def git_source(repository, revision, name):
    return subprocess.check_output(['git', '-C', str(repository), 'show', f'{revision}:{name}'])


def reconstruct(repository, artifacts, destination, kind):
    manifest = json.loads((artifacts / 'frozen-geometry.json').read_text())
    if manifest['revision'] != BASE or manifest['geometry']['context'] != 32768:
        raise ValueError('Pinned 32K historical snapshot required')
    scripts = destination / 'scripts/ci'
    scripts.mkdir(parents=True)
    for name, checksum in manifest['before'].items():
        if Path(name).name != name:
            raise ValueError('Flat deployment source names required')
        payload = git_source(repository, BASE, 'scripts/ci/' + name)
        if digest(payload) != checksum:
            raise ValueError('Historical deployment source differs: ' + name)
        (scripts / name).write_bytes(payload)
    patch_file = (artifacts / 'geometry.patch').resolve()
    patch_text = patch_file.read_text()
    paths = [line[6:] for line in patch_text.splitlines() if line.startswith('+++ b/')]
    if not paths or any(path not in {'scripts/ci/' + name for name in manifest['before']} for path in paths):
        raise ValueError('Patch changes outside historical deployment inputs')
    for mode in (['--check'], []):
        environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(destination.parent.resolve()))
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'apply', *mode, str(patch_file)], cwd=destination,
            env=environment, check=True, timeout=10)
    for name, checksum in manifest['after'].items():
        if Path(name).name != name:
            raise ValueError('Flat deployment source names required')
        path = scripts / name
        if name not in manifest['before']:
            path.write_bytes(git_source(repository, REVISIONS[kind], 'scripts/ci/' + name))
        if digest(path.read_bytes()) != checksum:
            raise ValueError('Reconstructed deployment differs: ' + name)
    report_name = 'dspark-native-8k-attention.json' if kind == 'draft' else 'target-t16-attention-8k.json'
    report = json.loads((artifacts / report_name).read_text())
    for name, checksum in report['sources'].items():
        relative = posixpath.normpath('scripts/ci/' + name)
        path = destination / relative
        path.resolve().relative_to(destination.resolve())
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(git_source(repository, BASE, relative))
        if digest(path.read_bytes()) != checksum:
            raise ValueError('Reconstructed probe source differs: ' + name)


def stage(repository, numerical, diagnostics, target, output):
    if output.exists():
        raise ValueError('Fresh evidence destination required')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        root = Path(temporary) / 'evidence'
        root.mkdir()
        for name, origin in (
                ('draft-numerical.json', numerical / 'dspark-native-8k-attention.json'),
                ('draft-diagnostics.json', diagnostics / 'dspark-native-8k-attention.json'),
                ('target-replay.json', target / 'target-t16-attention-8k.json')):
            payload = origin.read_bytes()
            if digest(payload) != REPORTS[name]:
                raise ValueError('Pinned report required: ' + name)
            (root / name).write_bytes(payload)
        reconstruct(repository, numerical, root / 'draft', 'draft')
        reconstruct(repository, target, root / 'target', 'target')
        result = qualify(root, draft_sources=root / 'draft/scripts/ci',
            target_sources=root / 'target/scripts/ci', context=32768)
        (root / 'qualification.json').write_text(json.dumps(result, indent=2) + '\n')
        root.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'numerical', 'diagnostics', 'target', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(stage(**vars(options)), indent=2))


if __name__ == '__main__':
    main()
