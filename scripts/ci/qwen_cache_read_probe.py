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
    parser.add_argument('--deserialize', action='store_true')
    options = parser.parse_args()
    root = options.root.resolve(strict=True)
    if len(options.file) > 4:
        raise ValueError('At most four explicitly selected cache files')
    if options.deserialize:
        print(json.dumps(dict(event='native_import_started')), flush=True)
        import ttnn
        print(json.dumps(dict(event='native_import_complete', module=ttnn.__file__)), flush=True)
    for name in options.file:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Regular cache file inside the selected root required')
        if options.deserialize and path.stat().st_size > 134217728:
            raise ValueError('Native probe file exceeds 128 MiB bound')
        previous = None
        for repetition in range(2):
            print(json.dumps(dict(event='cache_read_started', file=name, repetition=repetition)), flush=True)
            result = read_prefix(path, 67108864)
            if previous is not None and (result['sha256'], result['bytes_read'], result['file_bytes']) != previous:
                raise ValueError('Repeated cache file prefix changed')
            previous = (result['sha256'], result['bytes_read'], result['file_bytes'])
            print(json.dumps(dict(event='cache_read_complete', file=name, repetition=repetition,
                **result, scope=__doc__, performance_qualified=False)), flush=True)
            if options.deserialize:
                print(json.dumps(dict(event='host_deserialize_started', file=name, repetition=repetition)), flush=True)
                started, cpu = perf_counter(), process_time()
                tensor = ttnn._ttnn.tensor.load_tensor_flatbuffer(str(path), device=None)
                elapsed_ms, cpu_ms = (perf_counter() - started) * 1000, (process_time() - cpu) * 1000
                if tensor.storage_type() != ttnn.StorageType.HOST:
                    raise AssertionError('Host-only deserialization required')
                print(json.dumps(dict(event='host_deserialize_complete', file=name, repetition=repetition,
                    elapsed_ms=elapsed_ms, cpu_ms=cpu_ms, shape=list(tensor.shape),
                    scope='Full file deserialization after buffered prefix read; no device upload or TG',
                    performance_qualified=False)), flush=True)
                del tensor


if __name__ == '__main__':
    main()
