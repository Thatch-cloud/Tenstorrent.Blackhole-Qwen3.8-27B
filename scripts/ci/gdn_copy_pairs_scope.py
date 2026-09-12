"""T16-only program construction override; all other widths retain native kernels."""

from contextlib import contextmanager
import hashlib

import gdn_vsplit as split
from gdn_copy_pairs import transform


@contextmanager
def scoped_copy_pairs(admission):
    original = split.build_program
    if getattr(original, '_copy_pairs_override', False):
        raise ValueError('Nested copy-pair experiments are forbidden')
    audit = dict(admission=admission, loads=[], restored=False)

    def build(operations, mesh, shards, kernels, stage, rows, **options):
        if stage == 'recurrence' and rows == 16:
            before = kernels[stage]['compute']
            after = transform(before)
            kernels = dict(kernels, recurrence=dict(kernels[stage], compute=after))
            audit['loads'].append(dict(rows=rows,
                control_sha256=hashlib.sha256(before.encode()).hexdigest(),
                candidate_sha256=hashlib.sha256(after.encode()).hexdigest()))
        return original(operations, mesh, shards, kernels, stage, rows, **options)

    build._copy_pairs_override = True
    split.build_program = build
    try:
        yield audit
    finally:
        if split.build_program is not build:
            raise ValueError('Program builder changed externally during experiment')
        split.build_program = original
        audit['restored'] = True
