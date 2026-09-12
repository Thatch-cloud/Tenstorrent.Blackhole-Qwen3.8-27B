"""Compare repeated whole-file and streaming reads without loading tensor runtimes."""

import argparse
import hashlib
import json
from pathlib import Path
import platform


def inspect_file(path, expected):
    data = path.read_bytes()
    whole = hashlib.sha256(data).hexdigest()
    second = path.read_bytes()
    second_hash = hashlib.sha256(second).hexdigest()
    digest = hashlib.sha256()
    differences, second_differences = [], []
    offset = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            previous = data[offset:offset + len(chunk)]
            if previous != chunk and len(differences) < 8:
                for index, (first, streamed_byte) in enumerate(zip(previous, chunk)):
                    if first != streamed_byte:
                        differences.append(dict(offset=offset + index, whole_byte=first, stream_byte=streamed_byte))
                        if len(differences) == 8:
                            break
            other = second[offset:offset + len(chunk)]
            if other != chunk and len(second_differences) < 8:
                for index, (first, last) in enumerate(zip(other, chunk)):
                    if first != last:
                        second_differences.append(dict(offset=offset + index, whole_byte=first, stream_byte=last))
                        if len(second_differences) == 8:
                            break
            offset += len(chunk)
    rehashed = hashlib.sha256(data).hexdigest()
    streamed = digest.hexdigest()
    return dict(file=str(path), bytes=len(data), streamed_bytes=offset, expected=expected,
        whole=whole, second_whole=second_hash, streamed=streamed, rehashed=rehashed,
        differences=differences, second_differences=second_differences,
        passed=whole == second_hash == streamed == rehashed == expected and len(data) == len(second) == offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, choices=range(1, 11), default=3)
    options = parser.parse_args()
    from draft_mlp_fixture import TENSORS, TENSOR_SHA256

    report = dict(scope=__doc__, platform=platform.platform(), passed=False, checks=[])
    for repetition in range(options.repetitions):
        for name, (_, filename) in TENSORS.items():
            result = inspect_file(options.fixture / filename, TENSOR_SHA256[name])
            result['repetition'] = repetition
            report['checks'].append(result)
            options.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(result), flush=True)
    report['passed'] = all(check['passed'] for check in report['checks'])
    options.output.write_text(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
