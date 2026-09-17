"""Stage the full-vocabulary compact score simulator probe without learned weights."""

import argparse
import hashlib
import json
from pathlib import Path


def stage(checkout):
    scripts = Path(checkout) / 'scripts/ci'
    if (scripts / 'compact_score_device.py').exists():
        raise ValueError('Fresh compact score staging required')
    sources = {}
    for name in ('compact_score_device.py', 'compact_score_io.cpp', 'compact_score_compute.cpp',
            'compact_score_reduce.cpp', 'dspark_score_layout.py', 'dspark_score_layout_io.cpp',
            'dspark_score_layout_compute.cpp'):
        payload = Path(__file__).with_name(name).read_bytes()
        if name.startswith('dspark_') and (scripts / name).read_bytes() != payload:
            raise ValueError('Frozen control source differs: ' + name)
        sources[name] = payload
    sources['gdn-output-grid-probe.py'] = Path(__file__).with_name('compact-score-probe.py').read_bytes()
    for name, payload in sources.items():
        if name.endswith('.py'):
            compile(payload, name, 'exec')
    for name, payload in sources.items():
        (scripts / name).write_bytes(payload)
    return dict(candidate='compact_score_selection', vocabulary=248320,
        sources={name: hashlib.sha256(payload).hexdigest() for name, payload in sources.items()},
        simulator_qualified=False, hardware_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().checkout), indent=2))
