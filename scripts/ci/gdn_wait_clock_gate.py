"""Admit only the reviewed synthetic diagnostic before combined hardware sampling."""

import hashlib
import json
from pathlib import Path

from gdn_multitoken import HASHES, KERNEL_ROOT
from gdn_shared_qk_gate import LOCAL_SOURCES, API_SOURCES
from gdn_wait_clock import instrument
from gdn_wait_clock_report import validate


REPORT_SHA256 = 'd1a9f1466adf0a764c4a655f94e699bca2939a54c17d5238b942d855f1c64628'
KERNEL = dict(control_sha256='ce404cf287f9962243dc65f30d9736f45b0c95cfff503c6c20d68f8981a41153',
    candidate_sha256='ce7a2541323de8e89dbd07e5e47b924d3d771bd7024a730a5a1abec0a52f9d95', token=8)
HELPERS = ('gdn_wait_clock.py', 'gdn_wait_clock_capture.py',
    'gdn_wait_clock_pipeline.py', 'mlp_compute_clock_projection.py',
    'mlp_clock_samples.py', 'frozen_mlp_wait_zones.py', 'frozen_recipe_context.py')


def retained(evidence, directory):
    evidence, directory = Path(evidence), Path(directory)
    raw = (evidence / 'gdn-shared-recurrence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact reviewed recurrence clock artifact required')
    report = json.loads(raw)
    validate(report)
    if (report['generated_kernels'] != [KERNEL]
            or (evidence / 'gdn-shared-recurrence.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_bytes()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Pinned clean simulator execution and generated compute required')
    names = tuple(name for name in LOCAL_SOURCES if not name.endswith('-probe.py')) + HELPERS
    for name in names:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != report['sources'].get('/experiment-scripts/ci/' + name):
            raise ValueError('Simulated diagnostic dependency changed: ' + name)
    return report


def qualify(evidence, directory, runtime):
    report = retained(evidence, directory)
    runtime = Path(runtime)
    native = [KERNEL_ROOT + '/' + name for name in HASHES]
    native += ['tt_metal/hw/inc/api/compute/' + name for name in API_SOURCES]
    for name in native:
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != report['sources'].get('/opt/tt-metal/' + name):
            raise ValueError('Native diagnostic dependency changed: ' + name)
    from gdn_shared_qk_recurrence import load_kernels
    before = load_kernels(runtime)['recurrence']['compute']
    if (hashlib.sha256(before.encode()).hexdigest() != KERNEL['control_sha256']
            or hashlib.sha256(instrument(before).encode()).hexdigest() != KERNEL['candidate_sha256']):
        raise ValueError('Constructed recurrence differs from simulator')
    return dict(report_sha256=REPORT_SHA256, kernel=dict(KERNEL), diagnostic_only=True,
        hardware_qualified=False, performance_qualified=False)
