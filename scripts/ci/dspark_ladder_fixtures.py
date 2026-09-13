"""Reuse full-history poison and frontier controls at explicit ladder capacities."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
from unittest.mock import patch

import dspark_full_attention
from dspark_ladder_geometry import geometry


@contextmanager
def fixture_probe(context, output_tokens=1024):
    fixture = geometry(context, output_tokens)
    path = Path(__file__).with_name('dspark-native-8k-attention-probe.py')
    spec = importlib.util.spec_from_file_location('ladder_fixture_probe', path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.CAPACITY = fixture['capacity']
    probe.POSITIONS = fixture['probe_positions']
    with patch.object(dspark_full_attention, 'MAX_CONTEXT', fixture['capacity']):
        yield probe
