"""Admit only the reviewed gate-exp replay and its exact generated kernel."""

import hashlib
import json
from pathlib import Path

from gdn_gate_exp_fusion import transform
from gdn_gate_exp_report import KERNEL, inspect
from gdn_shared_qk_gate import API_SOURCES


REPORT_SHA256 = 'ce1be67c12fc03a6514fca5e69882f9642d7ec902ebd5446123991cec4dc55c2'


def qualify(evidence, directory, runtime):
    evidence, runtime = Path(evidence), Path(runtime)
    raw = (evidence / 'gdn-shared-recurrence.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact reviewed gate-exp simulator artifact required')
    result = inspect(evidence, directory)
    report = json.loads(raw)
    for name in API_SOURCES:
        relative = 'tt_metal/hw/inc/api/compute/' + name
        if hashlib.sha256((runtime / relative).read_bytes()).hexdigest() != report['sources'].get(
                '/opt/tt-metal/' + relative):
            raise ValueError('Compute API differs from gate-exp simulation: ' + relative)
    from gdn_shared_qk_recurrence import load_kernels
    from gdn_vsplit_norm_batch import validate_runtime
    validate_runtime(runtime)
    before = load_kernels(runtime)['recurrence']['compute']
    after = transform(before)
    if (hashlib.sha256(before.encode()).hexdigest() != KERNEL['control_sha256']
            or hashlib.sha256(after.encode()).hexdigest() != KERNEL['candidate_sha256']):
        raise ValueError('Generated gate-exp recurrence differs from simulator qualification')
    return result
