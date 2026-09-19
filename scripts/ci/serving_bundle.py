"""Package inventoried staged sources, preserving frozen runtime adaptations."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from serving_image_inventory import BINARY_SHA256, CACHE_KEY


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expose_singleton_pages(source):
    anchor = '        singleton_pages = upload(pages, ttnn.int32)\n'
    exposed = '        self.singleton_pages = singleton_pages\n'
    if source.count(anchor) != 1 or exposed in source:
        raise ValueError('Expected pristine staged singleton-page allocation')
    result = source.replace(anchor, anchor + exposed, 1)
    compile(result, 'model_batch.py', 'exec')
    return result


def package(staged, updates, report, output):
    staged, updates, output = Path(staged), Path(updates), Path(output)
    if (report.get('cached_binary_matches') is not True or report.get('cache_key') != CACHE_KEY
            or report.get('cached_binary', {}).get('sha256') != BINARY_SHA256
            or report.get('runtime_revision') != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'):
        raise ValueError('Verified retained runtime inventory required')
    fingerprints = report.get('staged_files', {})
    required = {'model_batch.py', 'verifier_engine.py', 'dflash_combined_request.py',
        'draft_kv_slide.cpp', 'draft_kv_slide_gate.py', 'frozen_combined_runtime.py',
        'target_t16_attention_gate.py', 'dflash_t16_native_scope.py'}
    if set(fingerprints) != required:
        raise ValueError('Complete critical staged-source inventory required')
    for name, record in fingerprints.items():
        path = staged / 'scripts/ci' / name
        if record.get('present') is not True or digest(path) != record.get('sha256'):
            raise ValueError('Staged source changed since inventory: ' + name)
    singleton = expose_singleton_pages((staged / 'scripts/ci/model_batch.py').read_text())
    if output.exists():
        raise ValueError('Fresh serving bundle destination required')
    roots = {'scripts': 'experiment-scripts', 'optimisation': 'experiment-optimisation',
        'speculative-decoding/harness': 'speculative-decoding/harness'}
    selected = []
    suffixes = ('.py', '.cpp', '.h', '.hpp', '.json', '.sh', '.patch', '.txt',
        '.yaml', '.yml', '.textproto', '.sha256', '.exit-status')
    for relative, destination in roots.items():
        root = staged / relative
        if not root.is_dir():
            raise ValueError('Missing staged source root: ' + relative)
        for source in sorted(root.rglob('*')):
            if source.is_symlink():
                raise ValueError('Symlinks are not admitted in serving source bundles')
            if source.is_file() and source.name.endswith(suffixes) and '__pycache__' not in source.parts:
                selected.append((source, output / destination / source.relative_to(root)))
    for source, destination in selected:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    scripts = output / 'experiment-scripts/ci'
    for source in [*sorted((updates / 'scripts/ci').glob('serving_*.py')),
            updates / 'scripts/ci/dflash_combined_request.py']:
        shutil.copy2(source, scripts / source.name)
    (scripts / 'model_batch.py').write_text(singleton)
    manifest = dict(cache_key=CACHE_KEY, binary_sha256=BINARY_SHA256,
        staged_inputs=fingerprints, serving_qualified=False, performance_qualified=False,
        sources={path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob('*')) if path.is_file()})
    (output / 'serving-bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('staged', 'updates', 'inventory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    arguments = parser.parse_args()
    package(arguments.staged, arguments.updates, json.loads(arguments.inventory.read_text()), arguments.output)
