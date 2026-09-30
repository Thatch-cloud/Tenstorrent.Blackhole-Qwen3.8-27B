"""Stage direct-window simulation onto the frozen source-only runner recipe."""

import argparse
import hashlib
import json
from pathlib import Path


def stage(checkout):
    scripts = Path(checkout) / 'scripts/ci'
    if (scripts / 'gdn_direct_window_device.py').exists():
        raise ValueError('Fresh direct-window staging required')
    for name in ('gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'attention_batch.py', 'gdn_multitoken_conv.py'):
        if (scripts / name).read_bytes() != Path(__file__).with_name(name).read_bytes():
            raise ValueError('Frozen control source differs: ' + name)
    payloads = {name: Path(__file__).with_name(name).read_bytes()
                for name in ('gdn_direct_window.py', 'gdn_direct_window_device.py')}
    payloads['gdn-output-grid-probe.py'] = Path(__file__).with_name('gdn-direct-window-probe.py').read_bytes()
    for name, payload in payloads.items():
        compile(payload, name, 'exec')
    for name, payload in payloads.items():
        (scripts / name).write_bytes(payload)
    return dict(sources={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
                simulator_qualified=False, hardware_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().checkout), indent=2))
