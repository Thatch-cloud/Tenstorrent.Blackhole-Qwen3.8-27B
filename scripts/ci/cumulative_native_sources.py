"""Fetch only hash-qualified normalization sources from the pinned native revision."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from gdn_multitoken import HASHES, KERNEL_ROOT
from gdn_shared_qk_gate import API_SOURCES
from shared_qk_norm_scatter_gate import REPORT_SHA256, validate_report
from mlp_register_epilogue_gate import TYPECAST, HARDWARE_PACKER
from mlp_rounding_policy import HEADER, PACKER


REVISION = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'
PATHS = tuple(KERNEL_ROOT + '/' + name for name in HASHES) + tuple(
    'tt_metal/hw/inc/api/compute/' + name for name in API_SOURCES)


def restore(report_path, destination, *, register_epilogue=False):
    if type(register_epilogue) is not bool:
        raise ValueError('Explicit register native-source policy required')
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained normalization simulator report required')
    report = json.loads(raw)
    validate_report(report)
    root = Path(destination).resolve()
    payloads = {}
    expected_sources = {relative: report['sources']['/opt/tt-metal/' + relative] for relative in PATHS}
    if register_epilogue:
        expected_sources.update({HEADER: TYPECAST, PACKER: HARDWARE_PACKER})
    for relative, expected in expected_sources.items():
        target = root / relative
        if target.exists():
            payload = target.read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/tenstorrent/tt-metal/{REVISION}/{relative}'
            with urlopen(url, timeout=30) as response:
                payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Pinned normalization native source mismatch: ' + relative)
        payloads[relative] = payload
    for relative, payload in payloads.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return dict(revision=REVISION, simulator_report_sha256=REPORT_SHA256,
        register_epilogue=register_epilogue,
        sources={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        hardware_qualified=False, performance_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--register-epilogue', action='store_true')
    options = parser.parse_args()
    print(json.dumps(restore(options.report, options.destination,
        register_epilogue=options.register_epilogue), indent=2))
