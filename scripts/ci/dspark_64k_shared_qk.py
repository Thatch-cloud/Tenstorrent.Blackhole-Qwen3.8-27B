"""Audit shared-Q/K preparation with the already-qualified 64K fused MLP."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch


def validate_shared(audit):
    loads = audit.get('loads', [])
    if (audit.get('restored') is not True or audit.get('released') is not True
            or not loads or len(loads) % 48
            or any(value != dict(rows=16, programs=3, retained_preparation_buffers=2) for value in loads)):
        raise ValueError('Complete restored T16 shared-Q/K pipelines required')


@contextmanager
def shared_factory_scope(module, shared_scope, admission, records):
    if (os.environ.get('QWEN_64K_SHARED_QK_AUDIT') != '1'
            or os.environ.get('QWEN_64K_MLP_AUDIT') != '1'
            or os.environ.get('QWEN_DSPARK_SFPU_REQUEST_SCREEN') != '1'
            or os.environ.get('QWEN_DSPARK_SFPU_TIMED', '0') != '0'):
        raise ValueError('Explicit combined MLP/shared-QK correctness audit required')
    original = module.FusedT16Arm

    class CombinedArm(original):
        @contextmanager
        def install(self):
            with shared_scope(self.operations, admission) as shared:
                with super().install():
                    yield self
            validate_shared(shared)
            records.append(shared)

    with patch.object(module, 'FusedT16Arm', CombinedArm):
        yield


def admission(directory, runtime):
    from gdn_shared_qk_gate import qualify

    return qualify(Path(directory) / 'gdn-shared-recurrence.json', directory, runtime)
