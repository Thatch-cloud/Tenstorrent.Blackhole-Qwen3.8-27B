"""Stage synthetic T16 cache qualification without changing baseline kernel files."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once


def adapt_probe(source, *, norm_prefetch=False, norm_scatter=False):
    if (type(norm_prefetch) is not bool or type(norm_scatter) is not bool
            or norm_prefetch and norm_scatter):
        raise ValueError('Explicit mutually exclusive norm selection required')
    module = ('shared_qk_norm_scatter' if norm_scatter else
        'frozen_gdn_norm_prefetch' if norm_prefetch else 'frozen_gdn_input_cache')
    source = replace_once(source,
        'from gdn_shared_qk_pipeline import build as build_pipeline',
        f'from {module} import build as build_pipeline')
    source = replace_once(source,
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True)',
        'rows=16, norm_unchanged=True, state_math_unchanged=True, shared_qk_preparation=True,\n' +
        ('        norm_bridge_scatter=True, norm_direct_read_bytes=8192)' if norm_scatter else
         '        norm_bridge_prefetch=True, norm_staging_bytes=8192)' if norm_prefetch else
         '        v_beta_gate_cache=True, cache_pages_per_worker=3)'))
    compile(source, 'gdn-shared-recurrence-probe.py', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--norm-prefetch', action='store_true')
    parser.add_argument('--norm-scatter', action='store_true')
    options = parser.parse_args()
    scripts = options.checkout / 'scripts/ci'
    name = 'gdn-shared-recurrence-probe.py'
    original = (scripts / name).read_text()
    candidate = adapt_probe(original, norm_prefetch=options.norm_prefetch, norm_scatter=options.norm_scatter)
    helper_name = ('shared_qk_norm_scatter.py' if options.norm_scatter else
        'frozen_gdn_norm_prefetch.py' if options.norm_prefetch else 'frozen_gdn_input_cache.py')
    helper = Path(__file__).with_name(helper_name).read_bytes()
    (scripts / name).write_bytes(candidate.encode())
    (scripts / helper_name).write_bytes(helper)
    if options.norm_scatter:
        (scripts / 'gdn_norm_scatter.py').write_bytes(Path(__file__).with_name('gdn_norm_scatter.py').read_bytes())
    print(json.dumps(dict(probe_before=hashlib.sha256(original.encode()).hexdigest(),
        probe_after=hashlib.sha256(candidate.encode()).hexdigest(),
        helper_sha256=hashlib.sha256(helper).hexdigest(), simulator_qualified=False)))


if __name__ == '__main__':
    main()
