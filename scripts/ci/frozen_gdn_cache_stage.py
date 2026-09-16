"""Stage synthetic T16 cache qualification without changing baseline kernel files."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


def adapt_probe(source):
    source = replace_once(source,
        'from gdn_shared_qk_pipeline import build as build_pipeline',
        'from frozen_gdn_input_cache import build as build_pipeline')
    source = replace_once(source,
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True)',
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True,\n'
        '        v_beta_gate_cache=True, cache_pages_per_worker=3)')
    compile(source, 'gdn-shared-recurrence-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    name = 'gdn-shared-recurrence-probe.py'
    original = (scripts / name).read_text()
    candidate = adapt_probe(original)
    helper = Path(__file__).with_name('frozen_gdn_input_cache.py').read_bytes()
    (scripts / name).write_bytes(candidate.encode())
    (scripts / 'frozen_gdn_input_cache.py').write_bytes(helper)
    print(json.dumps(dict(probe_before=hashlib.sha256(original.encode()).hexdigest(),
        probe_after=hashlib.sha256(candidate.encode()).hexdigest(),
        helper_sha256=hashlib.sha256(helper).hexdigest(), simulator_qualified=False)))


if __name__ == '__main__':
    main()
