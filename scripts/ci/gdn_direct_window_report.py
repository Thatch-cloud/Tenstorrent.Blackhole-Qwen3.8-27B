"""Check every direct-window output, prefix and immutable input before admission."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_direct_window_device import HASHES, sources


def validate(report, directory, native_root):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('backend') != 'simulator'
            or report.get('performance_qualified') is not False or report.get('native_unchanged') is not True
            or report.get('projection_memory') != 'L1'
            or report.get('native_sources') != HASHES or report.get('error')):
        raise ValueError('Successful unchanged native direct-window simulator evidence required')
    for name, count in (('checks', 7), ('immutable_checks', 11)):
        expected = [dict(seed=seed, mode=mode, operand=operand, chip=chip, exact=True)
                    for mode, seed in (('eager', 0), ('replay', 0), ('replay', 1), ('replay', 2))
                    for operand in range(count) for chip in (0, 1)]
        if report.get(name) != expected:
            raise ValueError('Complete exact two-chip matrix required: ' + name)
    names = ('gdn-output-grid-probe.py', 'gdn_direct_window.py', 'gdn_direct_window_device.py',
             'gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'attention_batch.py', 'gdn_multitoken_conv.py')
    if set(report.get('sources', {})) != set(names) or report['sources'] != report.get('sources_after'):
        raise ValueError('Complete unchanged script source closure required')
    for name in names:
        local = 'gdn-direct-window-probe.py' if name == 'gdn-output-grid-probe.py' else name
        if hashlib.sha256((Path(directory) / local).read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Qualified script differs from candidate: ' + name)
    generated = sources(native_root)
    if hashlib.sha256(generated['reader'].encode()).hexdigest() != report.get('generated_reader_sha256'):
        raise ValueError('Generated direct reader differs from simulator evidence')
    return dict(simulator_qualified=True, hardware_qualified=False, checks=56, immutable_checks=88,
                generated_reader_sha256=report['generated_reader_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--native-root', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(validate(json.loads(options.report.read_text()), Path(__file__).parent,
                              options.native_root), indent=2))
