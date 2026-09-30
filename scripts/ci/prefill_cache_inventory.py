"""Read pinned hybrid-model prefill ownership code without importing the runtime."""

import argparse
import hashlib
import json
from pathlib import Path


MODEL = 'models/demos/blackhole/qwen36/tt'
GENERATOR = 'models/common/sampling/generator.py'


def inventory(root, output):
    root = Path(root).resolve(strict=True)
    output = Path(output)
    if output.exists():
        raise ValueError('Fresh prefix-cache source inventory required')
    directory = root / MODEL
    if not directory.is_dir() or not directory.resolve(strict=True).is_relative_to(root):
        raise ValueError('Pinned model source directory required')
    paths = sorted(directory.rglob('*.py')) + [root / GENERATOR]
    if not 2 <= len(paths) <= 96:
        raise ValueError('Unexpected model source count')
    payloads = {}
    for path in paths:
        if not path.is_file():
            raise ValueError('Missing pinned source')
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Model source escaped pinned runtime')
        if path.stat().st_size > 512 * 1024:
            raise ValueError('Missing or oversized pinned source')
        payloads[path.relative_to(root).as_posix()] = path.read_bytes()
    report = dict(devices_opened=False, model_weights_loaded=False, prefix_cache_enabled=False,
        scope=__doc__, files={name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            for name, data in payloads.items()})
    for name, data in payloads.items():
        destination = output / 'sources' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/opt/tt-metal'))
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(inventory(options.root, options.output / 'prefill-cache')))
