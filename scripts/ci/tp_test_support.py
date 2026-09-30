"""Test support for the four-card twins: a process at QWEN_FAST_TP=4 as serving_startup.start leaves it.

serving_startup.start installs the tp_addresses seam before it attaches anything, which points the pair's pinned names
(the helpers recorded evidence hashes) at their four-card twins. A test that calls a pair module at four cards through
its own name has to see the same, so four_cards() sets the width AND installs the seam, and puts both back on exit -
only the outermost entry uninstalls, so nested use stays four-card until the outer one leaves. Not imported by any
served module (test only; not in the image copy lists).
"""

from contextlib import contextmanager
import os
from unittest.mock import patch

import tp_addresses


@contextmanager
def four_cards():
    with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
        installed = tp_addresses.install()
        try:
            yield
        finally:
            if installed:
                tp_addresses.uninstall()


def pair():
    return patch.dict(os.environ, {}, clear=True)
