"""Fresh sixteen-worker audit identity for complete repeated requests."""

from contextlib import contextmanager
from unittest.mock import patch

import matched_combined_gate as request_gate
import matched_combined_timed as timing
from workers_combined_gate import SCREEN_RUN, SCREEN_SHA256, qualify as qualify_combined


@contextmanager
def identity_scope():
    with patch.object(timing, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(timing, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(timing, 'qualify_combined', qualify_combined), \
            patch.object(request_gate, 'SCREEN_RUN', SCREEN_RUN), \
            patch.object(request_gate, 'SCREEN_SHA256', SCREEN_SHA256), \
            patch.object(request_gate, 'qualify', qualify_combined), timing.identity_scope():
        yield


def qualify(directory, report_directory=None):
    with identity_scope():
        return timing.qualify(directory, report_directory)
