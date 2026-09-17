"""Timing identity for the fresh combined audit, never the old factory report."""

from contextlib import contextmanager
from unittest.mock import patch

import dspark_64k_score_timed as baseline
from matched_combined_build import identity_scope as build_identity
from matched_combined_gate import SCREEN_RUN, SCREEN_SHA256, qualify as qualify_combined


@contextmanager
def identity_scope():
    with build_identity(), patch.object(baseline, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(baseline, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(baseline, 'qualify_score', qualify_combined), baseline.identity_scope():
        yield


def qualify(directory, report_directory=None):
    with identity_scope():
        return baseline.qualify(directory, report_directory)
