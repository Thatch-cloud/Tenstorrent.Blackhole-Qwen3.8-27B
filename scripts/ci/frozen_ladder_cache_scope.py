"""Apply simulator-qualified page geometry only inside allocated offline requests."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

from frozen_context_geometry import selected_geometry
from frozen_ladder_cache_gate import qualify
from frozen_ladder_ordered_cache import page_geometry
from ordered_cache import load_kernels


@contextmanager
def runtime_scope(directory):
    if (any(os.environ.get(key) != '1' for key in
            ('QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Allocated offline combined hardware runtime required')
    directory = Path(directory)
    context = selected_geometry()['context']
    evidence_root = directory / 'frozen-cache-evidence'
    digests = json.loads((evidence_root / 'reports.json').read_text())
    digest = digests[str(context)]
    report = qualify(directory, evidence_root / str(context), digest, context)
    kernels = load_kernels(os.environ['TT_METAL_HOME'])
    generated = {role: hashlib.sha256(source.encode()).hexdigest() for role, source in kernels.items()}
    if report['generated_hashes'] != generated:
        raise ValueError('Exact simulator-qualified generated cache kernels required')
    with page_geometry(context) as evidence:
        evidence.update(simulator_report_sha256=digest, simulator_qualified=True,
            generated_hashes=generated, hardware_qualified=False)
        try:
            yield evidence
        finally:
            qualify(directory, evidence_root / str(context), digest, context)
    if not evidence['restored'] or not evidence['calls']:
        raise ValueError('Executed and restored offline cache geometry scope required')
