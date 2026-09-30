"""Bind T32 native MLP-down admission to its own retained simulator evidence."""

import hashlib
import json
from pathlib import Path

from mlp_down_grid_gate import NATIVE_SOURCE, validate_report
from mlp_down_grid_t32_stage import adapt_probe


REPORT_SHA256 = '5768dc71a94c74424beae53919316700a1bc0b7b12e5bcfcc8d24487254cadac'


def qualify(evidence, directory, runtime):
    evidence, directory, runtime = Path(evidence), Path(directory), Path(runtime)
    raw = (evidence / 'gdn-output-grid.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained T32 MLP-down evidence required')
    report = json.loads(raw)
    validate_report(report, rows=32)
    if ((evidence / 'gdn-output-grid.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Clean pinned-runtime T32 simulator execution required')
    sources = {'/experiment-scripts/ci/' + name: (directory / name).read_bytes()
        for name in ('mlp_down_grid.py', 'attention_batch.py')}
    sources['/experiment-scripts/ci/gdn-output-grid-probe.py'] = adapt_probe(
        (directory / 'mlp-down-grid-probe.py').read_text()).encode()
    sources['/opt/tt-metal/' + NATIVE_SOURCE] = (runtime / NATIVE_SOURCE).read_bytes()
    if set(sources) != set(report['sources']):
        raise ValueError('Exact T32 MLP-down source closure required')
    for name, source in sources.items():
        if hashlib.sha256(source).hexdigest() != report['sources'][name]:
            raise ValueError('T32 simulator-qualified source changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(sources), rows=32,
        simulator_qualified=True, hardware_qualified=False, performance_qualified=False)
