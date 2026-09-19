"""Stage synthetic input gather overlap with complete prefix-state replay checks."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_multitoken import replace_once


def adapt(source):
    source = replace_once(source, 'from gdn_shared_qk_pipeline import build as build_pipeline',
        'from gdn_input_overlap import build as build_pipeline, BUILD_RECORDS')
    source = replace_once(source,
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True)',
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True,\n'
        '        input_overlap=True, generated_kernels=BUILD_RECORDS)')
    source = replace_once(source, "        report['passed'] = True",
        "        if len(BUILD_RECORDS) != 1:\n"
        "            raise AssertionError('One transformed shared-Q/K recurrence required')\n"
        "        report['passed'] = True")
    compile(source, 'gdn-shared-recurrence-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    name = 'gdn-shared-recurrence-probe.py'
    original = (scripts / name).read_bytes()
    candidate = adapt(original.decode())
    (scripts / name).write_bytes(candidate.encode())
    helper = Path(__file__).with_name('gdn_input_overlap.py').read_bytes()
    (scripts / 'gdn_input_overlap.py').write_bytes(helper)
    print(json.dumps(dict(probe_before=hashlib.sha256(original).hexdigest(),
        probe_after=hashlib.sha256(candidate.encode()).hexdigest(),
        helper_sha256=hashlib.sha256(helper).hexdigest(), simulator_qualified=False)))


if __name__ == '__main__':
    main()
