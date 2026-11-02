"""pytest entry: the tests import their helpers by plain module name (``import harness``), so put this directory first
and let fakes.py prepare the rest of the environment (package path, no-GPU probe) before vLLM is imported."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fakes  # noqa: E402,F401
