"""Export the pinned convolution graft without importing TT-NN or opening cards."""

import argparse
import hashlib
import json
from pathlib import Path


COMPONENT = 'ttnn/cpp/ttnn/operations/transformer/gdn_conv_gates'


def inventory(root, output):
    root, output = Path(root).resolve(strict=True), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Fresh convolution inventory output required')
    component = root / COMPONENT
    paths = sorted(path for path in component.rglob('*') if path.is_file()
                   and path.suffix in ('.cpp', '.hpp', '.h'))
    if not paths or len(paths) > 40:
        raise ValueError('Bounded existing convolution graft required')
    payloads = {}
    for path in paths:
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Convolution source escaped pinned runtime')
        if path.stat().st_size > 512 * 1024:
            raise ValueError('Unexpectedly large convolution source')
        payloads[path.relative_to(root).as_posix()] = path.read_bytes()
    if not any('/kernels/' in name for name in payloads):
        raise ValueError('Actual convolution kernels required')
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        destination = output / 'sources' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    report = dict(devices_opened=False, model_weights_loaded=False,
        scope='Pinned graft source inventory, not numerical or performance validation',
        files=[dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
               for name, payload in payloads.items()])
    (output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/opt/tt-metal'))
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(inventory(options.root, options.output)))
