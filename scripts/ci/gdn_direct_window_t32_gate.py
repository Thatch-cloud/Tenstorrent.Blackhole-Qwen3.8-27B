"""Source-bound T32 window admission including the generated tile-address reader."""

import hashlib
import json
from pathlib import Path

from gdn_direct_window_device import DIRECTORY, HASHES, sources as native_sources
from gdn_direct_window_t32_adapter import payloads
from gdn_direct_window_t32_stage import adapt_probe


REPORT_SHA256 = 'bdc8b31ec1ed80a401ecec1b581baf3006e3d72e3699c3e73417ef494a6962e1'


def qualify(evidence, directory, runtime):
    evidence, directory, runtime = Path(evidence), Path(directory), Path(runtime)
    raw = (evidence / 'gdn-output-grid.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained T32 direct-window report required')
    report = json.loads(raw)
    expected = dict(passed=True, closed_cleanly=True, rows=32, stage='complete', backend='simulator',
        performance_qualified=False, hardware_qualified=False, native_unchanged=True, projection_memory='L1')
    if (any(type(report.get(key)) is not type(value) or report.get(key) != value
            for key, value in expected.items()) or report.get('error')
            or report.get('native_sources') != HASHES or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete unchanged T32 direct-window replay required')
    for field, operands in (('checks', 7), ('immutable_checks', 11)):
        matrix = [dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True)
            for mode, seed in (('eager', 0), ('replay', 0), ('replay', 1), ('replay', 2))
            for operand in range(operands) for chip in (0, 1)]
        if report.get(field) != matrix:
            raise ValueError('Complete T32 native comparison required: ' + field)
    if ((evidence / 'gdn-output-grid.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
            or json.loads((evidence / 'container-cleanup.json').read_text()) != dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)):
        raise ValueError('Clean pinned-runtime T32 simulator execution required')
    generated = payloads({name: (directory / name).read_text() for name in
        ('gdn_direct_window.py', 'gdn_direct_window_device.py')})
    measured = {name: source.encode() for name, source in generated.items()}
    measured['gdn-output-grid-probe.py'] = adapt_probe((directory / 'gdn-direct-window-probe.py').read_text()).encode()
    measured.update({name: (directory / name).read_bytes() for name in
        ('gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'attention_batch.py', 'gdn_multitoken_conv.py')})
    if set(measured) != set(report['sources']):
        raise ValueError('Exact T32 direct-window source closure required')
    for name, source in measured.items():
        if hashlib.sha256(source).hexdigest() != report['sources'][name]:
            raise ValueError('Qualified T32 window source changed: ' + name)
    native_sources(runtime)
    namespace = {}
    exec(compile(generated['gdn_direct_window_t32.py'], 'qualified-t32-reader', 'exec'), namespace)
    reader = namespace['reader']((runtime / DIRECTORY / 'kernels/dataflow/reader_gdn_conv_gates.cpp').read_text())
    if hashlib.sha256(reader.encode()).hexdigest() != report.get('generated_reader_sha256'):
        raise ValueError('Generated T32 reader differs from simulator evidence')
    return dict(report_sha256=REPORT_SHA256, rows=32, simulator_qualified=True,
        hardware_qualified=False, performance_qualified=False, checked_sources=list(measured))
