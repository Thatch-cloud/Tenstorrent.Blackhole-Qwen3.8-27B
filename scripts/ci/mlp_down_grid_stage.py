"""Reuse the bounded simulator projection entrypoint for an explicitly labelled MLP probe."""

import argparse
import hashlib
import json
from pathlib import Path


def stage(checkout):
    scripts = Path(checkout) / 'scripts/ci'
    destination = scripts / 'gdn-output-grid-probe.py'
    marker = scripts / 'mlp_down_grid.py'
    if not destination.is_file() or marker.exists():
        raise ValueError('Fresh pinned projection staging required')
    original = destination.read_bytes()
    probe = Path(__file__).with_name('mlp-down-grid-probe.py').read_bytes()
    helper = Path(__file__).with_name('mlp_down_grid.py').read_bytes()
    compile(probe, str(destination), 'exec')
    compile(helper, str(marker), 'exec')
    destination.write_bytes(probe)
    marker.write_bytes(helper)
    return dict(projection='mlp_down', entrypoint='gdn-output-grid-probe.py',
        before=hashlib.sha256(original).hexdigest(), probe=hashlib.sha256(probe).hexdigest(),
        helper=hashlib.sha256(helper).hexdigest(), simulator_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().checkout)))
