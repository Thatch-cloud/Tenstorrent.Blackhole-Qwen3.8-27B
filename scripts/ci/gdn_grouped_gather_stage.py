"""Stage exact output, prefix-state and changed-input replay simulation."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_multitoken import replace_once


def adapt(source):
    source = replace_once(source, 'from gdn_shared_qk_pipeline import build as build_pipeline',
        'from gdn_grouped_gather import build as build_pipeline, BUILD_RECORDS')
    source = replace_once(source,
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True)',
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True,\n'
        '        grouped_gather=True, generated_kernels=BUILD_RECORDS)')
    source = replace_once(source, "        report['passed'] = True",
        "        if len(BUILD_RECORDS) != 1:\n"
        "            raise AssertionError('One grouped-copy recurrence build required')\n"
        "        report['passed'] = True")
    compile(source, 'gdn-shared-recurrence-probe.py', 'exec')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    probe = scripts / 'gdn-shared-recurrence-probe.py'
    original = probe.read_bytes()
    candidate = adapt(original.decode()).encode()
    helper = Path(__file__).with_name('gdn_grouped_gather.py').read_bytes()
    probe.write_bytes(candidate)
    (scripts / 'gdn_grouped_gather.py').write_bytes(helper)
    print(json.dumps(dict(probe_before=hashlib.sha256(original).hexdigest(),
        probe_after=hashlib.sha256(candidate).hexdigest(),
        helper_sha256=hashlib.sha256(helper).hexdigest(), simulator_qualified=False)))
