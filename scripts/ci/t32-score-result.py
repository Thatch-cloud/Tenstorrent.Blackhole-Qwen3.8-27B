"""Validate fused T32 simulator evidence against the checked-out probe and kernels."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from dspark_t32_result import validate


def qualify(path):
    path = Path(path)
    raw = path.read_bytes()
    report = json.loads(raw)
    if path.with_suffix('.exit-status').read_text().strip() != '0' or report.get('score_layout') != 'fused':
        raise ValueError('Successful explicitly fused score replay required')
    probe_path = Path(__file__).with_name('dspark-t32-markov-probe.py')
    spec = importlib.util.spec_from_file_location('score_probe', probe_path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    expected = probe.source_hashes()
    if report.get('sources') != expected or report.get('sources_after') != expected:
        raise ValueError('Fused candidate source fingerprints differ')
    native = report.get('native_sources', {})
    if any(native.get(name) != digest for name, digest in probe.BINARY_SHA256.items()):
        raise ValueError('Pinned simulator binaries required')
    if native.get(probe.PACKER) != probe.ORIGINAL_PACKER:
        raise ValueError('Unchanged simulator packer required')
    result = validate(report)
    return dict(result, report_sha256=hashlib.sha256(raw).hexdigest(), score_layout='fused',
        hardware_qualified=False, performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        raise ValueError('Fresh qualification file required')
    result = qualify(options.report)
    options.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
