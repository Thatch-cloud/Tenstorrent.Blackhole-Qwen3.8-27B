"""Check gate-exp simulator evidence; never qualify hardware or performance."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_gate_exp_stage import adapt
from gdn_multitoken import HASHES, KERNEL_ROOT
from gdn_shared_qk_gate import LOCAL_SOURCES


CONTROL_SHA256 = 'ce404cf287f9962243dc65f30d9736f45b0c95cfff503c6c20d68f8981a41153'
CANDIDATE_SHA256 = 'b29904d1e6b3a082a85d01dbe33a94d89cd6a61b5003013d2b4d2389bb741a97'
KERNEL = dict(control_sha256=CONTROL_SHA256, candidate_sha256=CANDIDATE_SHA256,
    removed_intermediate_cb=23, extra_cb_bytes=0, precision_changed=False,
    numerical_qualification_required=True)


def validate_report(report):
    required = ('passed', 'closed_cleanly', 'norm_unchanged', 'state_math_unchanged',
                'shared_qk_preparation', 'gate_exp_fusion')
    if (any(report.get(key) is not True for key in required)
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('rows') != 16 or not report.get('sources')
            or report['sources'] != report.get('sources_after') or report.get('numerical_failures')
            or report.get('hardware_qualified') is not False
            or report.get('timing_qualified') is not False
            or report.get('generated_kernels') != [KERNEL]):
        raise ValueError('Complete unchanged-source gate-exp simulator evidence required')
    for field, operands in (('checks', 3), ('immutable_checks', 6)):
        expected = [dict(mode=mode, operand=operand, chip=chip, exact=True)
            for mode in ('eager', 'replay_1', 'replay_2', 'replay_0')
            for operand in range(operands) for chip in (0, 1)]
        if report.get(field) != expected:
            raise ValueError('Complete exact eager/replay matrix required: ' + field)


def inspect(evidence, directory):
    evidence, directory = Path(evidence), Path(directory)
    raw = (evidence / 'gdn-shared-recurrence.json').read_bytes()
    report = json.loads(raw)
    validate_report(report)
    if ((evidence / 'gdn-shared-recurrence.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip()
                != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_bytes()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Pinned runtime and clean simulator execution required')
    expected = {f'/opt/tt-metal/{KERNEL_ROOT}/{name}': digest
                for name, digest in HASHES.items()}
    for name in (*LOCAL_SOURCES, 'gdn_gate_exp_fusion.py'):
        payload = (directory / name).read_bytes()
        if name == 'gdn-shared-recurrence-probe.py':
            payload = adapt(payload.decode()).encode()
        expected[f'/experiment-scripts/ci/{name}'] = hashlib.sha256(payload).hexdigest()
    for name, digest in expected.items():
        if report['sources'].get(name) != digest:
            raise ValueError('Simulator source differs from reviewed source: ' + name)
    return dict(report_sha256=hashlib.sha256(raw).hexdigest(), kernel=dict(KERNEL),
        checked_sources=list(expected), simulator_qualified=True,
        hardware_qualified=False, performance_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(inspect(options.evidence, Path(__file__).parent), indent=2))
