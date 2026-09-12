"""Experiment-only norm loader override with qualified inputs and restoration."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import gdn_norm_scatter as scatter
from gdn_norm_scatter_report import validate


@contextmanager
def scoped_reader(directory=None):
    directory = Path(directory) if directory is not None else Path(__file__).parent
    report = directory / 'gdn-norm-scatter-simulator.json'
    evidence = validate(json.loads(report.read_text()), directory)
    original = scatter.batch.load_kernels
    if getattr(original, '_norm_scatter_override', False):
        raise ValueError('Nested norm scatter override is forbidden')
    audit = dict(simulator=evidence, report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                 loads=[], restored=False)

    def load(root=scatter.batch.split.DEFAULT_ROOT):
        kernels = original(root)
        before = kernels['norm_gate']['reader']
        kernels['norm_gate']['reader'] = scatter.replace_reader(before)
        audit['loads'].append(dict(control_reader_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_reader_sha256=hashlib.sha256(kernels['norm_gate']['reader'].encode()).hexdigest()))
        return kernels

    load._norm_scatter_override = True
    scatter.batch.load_kernels = load
    try:
        yield audit
    finally:
        if scatter.batch.load_kernels is not load:
            raise ValueError('Norm loader changed externally during experiment')
        scatter.batch.load_kernels = original
        audit['restored'] = True
