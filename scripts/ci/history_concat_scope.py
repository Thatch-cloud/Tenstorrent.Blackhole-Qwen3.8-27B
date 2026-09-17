"""Explicit simulator-admitted concat lifetime candidate for the offline full window."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from unittest.mock import patch

from history_concat_gate import qualify
from history_concat_lifetime import join_rows


@contextmanager
def runtime_scope(directory):
    import dspark_history

    if (os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT') != '261888'
            or any(os.environ.get(key) != '1' for key in
                ('QWEN_FROZEN_COMBINED_RUNTIME', 'QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'))
            or os.environ.get('TT_METAL_SIMULATOR')):
        raise ValueError('Allocated offline full-window concat candidate required')
    directory = Path(directory)
    root = directory / 'history-concat-evidence'
    checksum = json.loads((root / 'admission.json').read_text())['report_sha256']
    qualify(directory, root, checksum)
    original = dspark_history.join_rows
    evidence = dict(calls=0, restored=False, simulator_report_sha256=checksum,
        performance_qualified=False, grouping_unchanged=True)

    def bounded(operations, values, retain):
        result = join_rows(operations, values, retain)
        evidence['calls'] += 1
        return result

    try:
        with patch.object(dspark_history, 'join_rows', bounded):
            yield evidence
    finally:
        evidence['restored'] = dspark_history.join_rows is original
        qualify(directory, root, checksum)
    if not evidence['restored'] or not evidence['calls']:
        raise ValueError('Executed and restored history concat candidate required')
