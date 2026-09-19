"""Prepare T32 numerical replay of the same native MLP-down grid change."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


def adapt_probe(source):
    for before, after in (
        ('Synthetic native T16 MLP-down', 'Synthetic native T32 MLP-down'),
        ('rows=16,', 'rows=32,'),
        ('(2, 1, 16, 8704)', '(2, 1, 32, 8704)'),
        ('create_matmul_1d_decode_progcfg(16,', 'create_matmul_1d_decode_progcfg(32,'),
        ('(2, 1, 16, 5120)', '(2, 1, 32, 5120)')):
        source = replace_once(source, before, after)
    compile(source, 'mlp-down-grid-t32-probe.py', 'exec')
    return source


def stage(checkout):
    from mlp_down_grid_stage import stage as stage_baseline

    baseline = stage_baseline(checkout)
    scripts = Path(checkout) / 'scripts/ci'
    probe = scripts / 'gdn-output-grid-probe.py'
    source = adapt_probe(probe.read_text())
    suite = scripts / 'simulator-suite.sh'
    launcher = replace_once(suite.read_text(), 'frozen_sim_phase.py --phase probe --seconds 510 ',
        'frozen_sim_phase.py --phase probe --seconds 420 ')
    probe.write_bytes(source.encode())
    suite.write_bytes(launcher.encode())
    return dict(rows=32, baseline=baseline, simulator_qualified=False, hardware_qualified=False,
        probe=hashlib.sha256(source.encode()).hexdigest(),
        suite=hashlib.sha256(launcher.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    print(json.dumps(stage(parser.parse_args().checkout), indent=2))
