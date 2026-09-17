"""Read-only loader sources and cache filesystem evidence; no tensor/device imports."""

import argparse
import hashlib
import json
from pathlib import Path
import os


def inspect_tree(root, *, limit=100, byte_limit=2097152):
    root = Path(root)
    report = dict(root=str(root), files=[], truncated=False)
    total = 0
    for path in sorted(root.rglob('*.py')):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        size = path.stat().st_size
        if len(report['files']) >= limit or total + size > byte_limit:
            report['truncated'] = True
            break
        payload = path.read_bytes()
        total += len(payload)
        report['files'].append(dict(path=str(path.relative_to(root)),
            sha256=hashlib.sha256(payload).hexdigest(), source=payload.decode('utf-8')))
    report['bytes'] = total
    return report


def filesystem(path):
    try:
        stats = os.statvfs(path)
        return dict(path=str(path), available_bytes=stats.f_bavail * stats.f_frsize,
            total_bytes=stats.f_blocks * stats.f_frsize, free_inodes=stats.f_favail,
            entries=sorted(entry.name for entry in Path(path).iterdir())[:64])
    except OSError as error:
        return dict(path=str(path), unavailable=type(error).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, action='append', default=[])
    parser.add_argument('--extra-source', type=Path, action='append', default=[])
    options = parser.parse_args()
    source = inspect_tree(options.source_root)
    if not source['files']:
        raise ValueError('Pinned model source directory is empty or missing')
    extra = []
    if len(options.extra_source) > 4:
        raise ValueError('At most four explicit extra source files')
    for path in options.extra_source:
        if path.stat().st_size > 262144:
            raise ValueError('Extra source exceeds diagnostic byte bound')
        payload = path.read_bytes()
        extra.append(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
            source=payload.decode('utf-8')))
    print(json.dumps(dict(scope=__doc__, source=source,
        extra_sources=extra, filesystems=[filesystem(path) for path in options.cache]), indent=2))


if __name__ == '__main__':
    main()
