"""Audited native fix: explicit CCL link counts must not evaluate discovery."""

import argparse
import hashlib
import json
from pathlib import Path
import re


SOURCES = {
    'all_gather_async/all_gather_async.cpp': ('1d0e498c0a28b577e1391859f33e11d209b48cf430e7d7b42a6433aab29631bb', 6),
    'reduce_scatter_minimal_async/reduce_scatter_minimal_async.cpp': ('1f8ae325e777cfa986ca4d2e9bacea1f4aff2949d81510d3ebf253c7ca406064', 1),
}
PATTERN = re.compile(r'(num_links|num_preferred_links)\.value_or\((ttnn::operations::ccl::common::get_num_links\([^()]*\))\)')


def rewrite(source, expected):
    result, count = PATTERN.subn(lambda match:
        f'({match[1]}.has_value() ? {match[1]}.value() : {match[2]})', source)
    if count != expected:
        raise ValueError(f'Expected {expected} eager discovery sites, found {count}')
    return result


def apply(root):
    directory = Path(root) / 'ttnn/cpp/ttnn/operations/experimental/ccl'
    pending, report = [], {}
    for name, (digest, count) in SOURCES.items():
        path = directory / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Native CCL source differs from pinned runtime: {name}')
        replacement = rewrite(data.decode(), count).encode()
        pending.append((path, replacement))
        report[name] = dict(before=digest, after=hashlib.sha256(replacement).hexdigest(), sites=count)
    for path, replacement in pending:
        path.write_bytes(replacement)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    options.output.write_text(json.dumps(apply(options.root), indent=2))
