"""Stage full-width direct-window replay without changing the T16 sources."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once
from gdn_direct_window_stage import stage as stage_baseline
from gdn_direct_window_t32_adapter import payloads


def adapt_probe(probe):
    for before, after in (
        ('from gdn_direct_window_device import execute, sources, HASHES',
            'from gdn_direct_window_t32_device import execute, sources, HASHES'),
        ("'gdn_direct_window.py', 'gdn_direct_window_device.py',",
            "'gdn_direct_window_t32.py', 'gdn_direct_window_t32_device.py',"),
        ('(2, 16, 8240)', '(2, 32, 8240)'),
        ('batch=16,', 'batch=32,'),
        ('projection_memory=options.projection_memory)',
            'projection_memory=options.projection_memory, rows=32, hardware_qualified=False)')):
        probe = replace_once(probe, before, after)
    return probe


def stage(checkout):
    baseline = stage_baseline(checkout)
    scripts = Path(checkout) / 'scripts/ci'
    originals = {name: (scripts / name).read_text() for name in
        ('gdn_direct_window.py', 'gdn_direct_window_device.py')}
    sources = payloads(originals)
    sources['gdn-output-grid-probe.py'] = adapt_probe((scripts / 'gdn-output-grid-probe.py').read_text())
    sources['simulator-suite.sh'] = replace_once((scripts / 'simulator-suite.sh').read_text(),
        'frozen_sim_phase.py --phase probe --seconds 510 ',
        'frozen_sim_phase.py --phase probe --seconds 420 ')
    for name, source in sources.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in sources.items():
        (scripts / name).write_bytes(source.encode())
    return dict(rows=32, baseline=baseline, simulator_qualified=False, hardware_qualified=False,
        sources={name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().checkout), indent=2))
