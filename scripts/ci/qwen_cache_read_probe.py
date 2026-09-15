"""Bounded cached-file reads; initial/repeated access, not cold-disk or model TG."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time


def read_prefix(path, limit):
    started, cpu = perf_counter(), process_time()
    checksum, count = hashlib.sha256(), 0
    with Path(path).open('rb') as source:
        before = Path(path).stat()
        while count < limit:
            block = source.read(min(1048576, limit - count))
            if not block:
                break
            checksum.update(block)
            count += len(block)
        after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Cache file changed during read')
    return dict(bytes_read=count, file_bytes=before.st_size, prefix_only=count < before.st_size,
        elapsed_ms=(perf_counter() - started) * 1000, cpu_ms=(process_time() - cpu) * 1000,
        sha256=checksum.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--file', action='append', required=True)
    options = parser.parse_args()
    root = options.root.resolve(strict=True)
    if len(options.file) > 4:
        raise ValueError('At most four explicitly selected cache files')
    for name in options.file:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Regular cache file inside the selected root required')
        previous = None
        for repetition in range(2):
            print(json.dumps(dict(event='cache_read_started', file=name, repetition=repetition)), flush=True)
            result = read_prefix(path, 67108864)
            if previous is not None and (result['sha256'], result['bytes_read'], result['file_bytes']) != previous:
                raise ValueError('Repeated cache file prefix changed')
            previous = (result['sha256'], result['bytes_read'], result['file_bytes'])
            print(json.dumps(dict(event='cache_read_complete', file=name, repetition=repetition,
                **result, scope=__doc__, performance_qualified=False)), flush=True)


if __name__ == '__main__':
    main()
