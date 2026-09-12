"""T16-only program construction override; all other widths retain native kernels."""

from contextlib import contextmanager
import hashlib

import gdn_vsplit as split
from gdn_outer_add import transform


@contextmanager
def scoped_outer_add(admission):
    original = split.build_program
    if getattr(original, '_outer_add_override', False):
        raise ValueError('Nested outer-add experiments are forbidden')
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

    build._outer_add_override = True
    split.build_program = build
    try:
        yield audit
    finally:
        if split.build_program is not build:
            raise ValueError('Program builder changed externally during experiment')
        split.build_program = original
        audit['restored'] = True
