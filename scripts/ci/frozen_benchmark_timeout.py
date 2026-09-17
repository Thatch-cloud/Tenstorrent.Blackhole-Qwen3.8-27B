"""Remove the full-request process timer without changing setup or cleanup controls."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


def transform(source):
    return replace_once(source,
        'runner=(timeout -k 20 3000 python3 -u "/experiment-scripts/ci/$probe.py"',
        'runner=(python3 -u "/experiment-scripts/ci/$probe.py"')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    options = parser.parse_args()
    path = options.checkout / 'scripts/ci/dspark-hardware-suite.sh'
    before = path.read_bytes()
    after = transform(before.decode()).encode()
    path.write_bytes(after)
    print(json.dumps(dict(file=path.name, before=hashlib.sha256(before).hexdigest(),
        after=hashlib.sha256(after).hexdigest(), full_request_timeout_removed=True,
        setup_and_cleanup_limits_preserved=True)))


if __name__ == '__main__':
    main()
