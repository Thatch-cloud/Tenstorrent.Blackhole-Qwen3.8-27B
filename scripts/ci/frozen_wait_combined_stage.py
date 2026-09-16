"""Add admitted marker capture to the qualified 32K norm/incremental runtime."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_mlp_wait_zones import instrument
from frozen_recipe_context import replace_once
from frozen_verifier_profile import FILES, adapt_sources
from frozen_wait_zone_capture import adapt_capture
from frozen_wait_zone_gate import qualify


HELPERS = ('frozen_wait_zone_gate.py', 'frozen_wait_zone_scope.py', 'frozen_wait_zone_report.py',
    'frozen_mlp_wait_zones.py', 'frozen_recipe_context.py', 'frozen_wait_combined_report.py')


def adapt(sources):
    result = adapt_sources(sources)
    result['dspark_8k_scope.py'] = replace_once(result['dspark_8k_scope.py'],
        '        stack.enter_context(incremental_scope(directory))\n        yield evidence',
        '        stack.enter_context(incremental_scope(directory))\n'
        '        from frozen_wait_zone_scope import runtime_scope as marker_scope\n'
        '        stack.enter_context(marker_scope(directory))\n        yield evidence')
    result['request_verifier_profile_report.py'] = replace_once(result['request_verifier_profile_report.py'],
        'from gdn_shared_qk_variants import validate_route as publication_route',
        'from frozen_wait_zone_scope import validate_route as publication_route')
    result['dspark-combined-profile.sh'] = adapt_capture(result['dspark-combined-profile.sh'])
    result['dspark-combined-profile.sh'] += ('\npython3 /experiment-scripts/ci/frozen_wait_combined_report.py '
        '"$output"\n')
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    geometry = json.loads(options.geometry.read_text())
    if (geometry.get('norm_prefetch') is not True or geometry.get('combined_runtime') is not True
            or geometry.get('verifier_profile') is not False or geometry.get('performance_qualified') is not False
            or geometry.get('geometry', {}).get('context') != 32768 or options.manifest.exists()):
        raise ValueError('Fresh prepared 32K norm/incremental runtime required')
    scripts = options.checkout / 'scripts/ci'
    for name, checksum in geometry['after'].items():
        if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Prepared runtime source changed: ' + name)
    sources = {name: (scripts / name).read_text() for name in (*FILES, 'dspark_8k_scope.py')}
    result = adapt(sources)
    for name in HELPERS:
        result[name] = Path(__file__).with_name(name).read_text()
    prefix = 'frozen-wait-zone-candidate/'
    result[prefix + 'fused_1d.py'] = (scripts / 'fused_1d.py').read_text()
    for role in ('input', 'weights'):
        name = f'fused_1d_{role}.cpp'
        result[prefix + name] = instrument((scripts / name).read_text(), role)
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    for name, source in result.items():
        destination = scripts / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.encode())
    admission = qualify(scripts)
    options.manifest.write_text(json.dumps(dict(admission=admission, diagnostic_only=True,
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        serving_defaults_changed=False, performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
