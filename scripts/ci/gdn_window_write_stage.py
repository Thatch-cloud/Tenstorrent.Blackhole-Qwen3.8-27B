"""Stage a source-bound window-only simulator probe without model loading."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frozen_recipe_context import REVISION
from gdn_window_write_overlap import transform_builder, transform_reader


def stage(checkout):
    scripts = Path(checkout) / 'scripts/ci'
    destination = scripts / 'window-write-candidate'
    if destination.exists():
        raise ValueError('Fresh window-write staging required')
    originals, candidates = {}, {}
    for name, transform in (('gdn_conv_windows.py', transform_builder), ('gdn_conv_windows.cpp', transform_reader)):
        original = subprocess.check_output(['git', '-C', str(checkout), 'show',
            f'{REVISION}:scripts/ci/{name}']).decode().replace('\r\n', '\n')
        if (scripts / name).read_text() != original:
            raise ValueError('Exact frozen window source required: ' + name)
        originals[name] = original.encode()
        candidates[name] = transform(original).encode()
    probe = Path(__file__).with_name('gdn-window-write-probe.py').read_bytes()
    compile(probe, 'gdn-window-write-probe.py', 'exec')
    compile(candidates['gdn_conv_windows.py'], 'gdn_conv_windows.py', 'exec')
    destination.mkdir()
    for name, payload in candidates.items():
        (destination / name).write_bytes(payload)
    (scripts / 'gdn-output-grid-probe.py').write_bytes(probe)
    return dict(candidate='gdn_window_write_overlap', rows=16, extra_scratch_bytes_per_worker=6144,
        before={name: hashlib.sha256(payload).hexdigest() for name, payload in originals.items()},
        after={name: hashlib.sha256(payload).hexdigest() for name, payload in candidates.items()},
        probe_sha256=hashlib.sha256(probe).hexdigest(), simulator_qualified=False, hardware_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().checkout), indent=2))
