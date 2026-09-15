"""Clean combined shared-Q/K timing with immutable audit identity."""

from contextlib import contextmanager
import os
from unittest.mock import patch

import dspark_64k_mlp_timed as mlp
from dspark_64k_shared_gate import SCREEN_RUN, SCREEN_SHA256, qualify as qualify_shared
from dspark_64k_shared_qk import validate_shared


@contextmanager
def identity_scope():
    with patch.object(mlp, 'SCREEN_RUN', SCREEN_RUN), patch.object(mlp, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(mlp, 'qualify_mlp', qualify_shared):
        yield


def qualify(directory, report_directory=None):
    with identity_scope():
        return mlp.qualify(directory, report_directory)


@contextmanager
def factory_scope(module, shared_scope, admission, records):
    mlp.splitk.require_timed()
    if (os.environ.get('QWEN_64K_SHARED_QK_TIMED') != '1'
            or os.environ.get('QWEN_64K_MLP_TIMED') != '1'
            or os.environ.get('QWEN_64K_SHARED_QK_AUDIT', '0') != '0'):
        raise ValueError('Explicit clean combined shared-Q/K timing required')
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
