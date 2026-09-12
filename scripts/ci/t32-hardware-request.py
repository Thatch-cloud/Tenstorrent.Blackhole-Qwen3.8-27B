"""Allocated T32 correctness screen using retained simulator-qualified proposal kernels."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from dspark_hardware_gate import digest
from dspark_intake import FILES
from dspark_rope_tables import DSparkRotary
from feature_projection import require_projection_environment
from t32_hardware_kernel import installed, request_admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timed', action='store_true', help='Run two timing repeats only after a fresh full audit')
    for name in ('checkpoint', 'config', 'target', 'output', 'evidence'):
        parser.add_argument('--' + name, type=Path, required=True)
    options = parser.parse_args()
    require_projection_environment(os.environ, True)
    if options.output.exists() or digest(options.config) != FILES['config.json'][1]:
        raise ValueError('Fresh output and pinned draft configuration required')
    rotary = DSparkRotary(json.loads(options.config.read_text()))
    root = Path(os.environ['TT_METAL_HOME'])
    directory = Path(__file__).parent
    with installed(root, options.evidence, directory) as evidence:
        with request_admission(root, evidence):
            spec = importlib.util.spec_from_file_location('t32_request_probe', directory / 't32-full-request-probe.py')
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
            probe.run_request(options, rotary, evidence, hardware=True)


if __name__ == '__main__':
    main()
