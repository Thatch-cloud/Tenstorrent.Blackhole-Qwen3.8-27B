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
    'frozen_mlp_wait_zones.py', 'frozen_recipe_context.py', 'frozen_wait_combined_report.py',
    'frozen_wait_raw_export.py')
SOURCE_FILES = (*FILES, 'dspark_8k_scope.py', 'verifier_engine.py')


def drain_setup(sources):
    result = dict(sources)
    source = result['full_dspark_request.py']
    for anchor in (
            '            seed = prefill(tokens)\n',
            '                output = decode(token, position, False)\n',
            '        drafter.prepare_trace(seed, audit=audit_features)\n',
            '        drafter.propose(seed, 15)\n\n    def factory():'):
        if anchor.endswith('def factory():'):
            replacement = ('        drafter.propose(seed, 15)\n'
                '        operations.synchronize_device(model.mesh_device)\n'
                '        operations.ReadDeviceProfiler(model.mesh_device)\n\n    def factory():')
        else:
            indent = anchor[:len(anchor) - len(anchor.lstrip())]
            replacement = (anchor + indent + 'operations.synchronize_device(model.mesh_device)\n'
                + indent + 'operations.ReadDeviceProfiler(model.mesh_device)\n')
        source = replace_once(source, anchor, replacement)
    result['full_dspark_request.py'] = source
    source = result['verifier_engine.py']
    changes = (
        ('                    ttnn.synchronize_device(self.mesh)\n                finally:',
         '                    ttnn.synchronize_device(self.mesh)\n'
         '                    ttnn.ReadDeviceProfiler(self.mesh)\n                finally:'),
        ("                        feature_capture=bucket.get('feature_capture')))\n                if rows > 1:",
         "                        feature_capture=bucket.get('feature_capture')))\n"
         '                ttnn.synchronize_device(self.mesh)\n'
         '                ttnn.ReadDeviceProfiler(self.mesh)\n                if rows > 1:'),
        ('                    for publication in publications.values():\n                        publication()\n',
         '                    for publication in publications.values():\n                        publication()\n'
         '                        ttnn.synchronize_device(self.mesh)\n'
         '                        ttnn.ReadDeviceProfiler(self.mesh)\n'),
        ("                        bucket['commits'][prefix], unused = capture_operation(ttnn, self.mesh, publication)\n",
         "                        bucket['commits'][prefix], unused = capture_operation(ttnn, self.mesh, publication)\n"
         '                        ttnn.synchronize_device(self.mesh)\n'
         '                        ttnn.ReadDeviceProfiler(self.mesh)\n'),
    )
    for before, after in changes:
        source = replace_once(source, before, after)
    result['verifier_engine.py'] = source
    return result


def adapt(sources):
    result = drain_setup(adapt_sources(sources))
    result['dspark_8k_scope.py'] = replace_once(result['dspark_8k_scope.py'],
        '        stack.enter_context(incremental_scope(directory))\n        yield evidence',
        '        stack.enter_context(incremental_scope(directory))\n'
        '        from frozen_wait_zone_scope import runtime_scope as marker_scope\n'
        '        stack.enter_context(marker_scope(directory))\n        yield evidence')
    result['request_verifier_profile_report.py'] = replace_once(result['request_verifier_profile_report.py'],
        'from gdn_shared_qk_variants import validate_route as publication_route',
        'from frozen_wait_zone_scope import validate_route as publication_route')
    result['dspark-combined-profile.sh'] = adapt_capture(result['dspark-combined-profile.sh'])
    result['dspark-combined-profile.sh'] = replace_once(result['dspark-combined-profile.sh'],
        'if [ -f "$output/.logs/$name" ]; then cp "$output/.logs/$name" "$output/metadata/$name"; fi',
        'if [ -f "$output/.logs/$name" ]; then\n'
        '            if [ "$name" = profile_log_device.csv ]; then\n'
        '                if [ ! -f "$output/metadata/$name" ] || [ "$output/.logs/$name" -nt "$output/metadata/$name" ]; then\n'
        '                    python3 /experiment-scripts/ci/frozen_wait_raw_export.py "$output/.logs/$name" "$output/metadata/$name"\n'
        '                fi\n'
        '            else\n'
        '                cp -u "$output/.logs/$name" "$output/metadata/$name"\n'
        '            fi\n'
        '        fi')
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
    sources = {name: (scripts / name).read_text() for name in SOURCE_FILES}
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
