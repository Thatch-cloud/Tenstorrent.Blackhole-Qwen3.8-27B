"""Request-owned T16 window builder override; preserve native tails and bindings."""

from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

from gdn_window_write_gate import REPORT_SHA256


@contextmanager
def scoped_window_writes(admission, directory):
    import gdn_conv_windows

    if admission.get('report_sha256') != REPORT_SHA256 or admission.get('passed') is not True:
        raise ValueError('Source-qualified simulator admission required')
    directory = Path(directory)
    for name in ('gdn_conv_windows.py', 'gdn_conv_windows.cpp',
            'window-write-candidate/gdn_conv_windows.py', 'window-write-candidate/gdn_conv_windows.cpp'):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != admission.get('source_hashes', {}).get(name):
            raise ValueError('Window source differs from admission: ' + name)
    original = gdn_conv_windows.build_windows
    if getattr(original, '_window_write_override', False):
        raise ValueError('Nested window write overrides forbidden')
    spec = importlib.util.spec_from_file_location('qualified_window_write_candidate',
        directory / 'window-write-candidate/gdn_conv_windows.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    audit = dict(report_sha256=REPORT_SHA256, hits=0, fallbacks=0, restored=False, shapes={})

    def build(mesh, projected, history):
        shape = tuple(projected.shape)
        label = str(shape)
        audit['shapes'][label] = audit['shapes'].get(label, 0) + 1
        if len(shape) == 3 and shape[1] == 16 and shape != (1, 16, 8256):
            raise ValueError('Unqualified T16 window geometry: ' + label)
        if shape != (1, 16, 8256):
            audit['fallbacks'] += 1
            return original(mesh, projected, history)
        if len(history) != 4 or any(tuple(value.shape) != (1, 1, 5120) for value in history):
            raise ValueError('Simulator-covered compact history geometry required')
        result = candidate.build_windows(mesh, projected, history)
        audit['hits'] += 1
        return result

    build._window_write_override = True
    try:
        with patch.object(gdn_conv_windows, 'build_windows', build):
            try:
                yield audit
            finally:
                if gdn_conv_windows.build_windows is not build:
                    raise ValueError('Window builder changed outside owning scope')
    finally:
        audit['restored'] = gdn_conv_windows.build_windows is original
