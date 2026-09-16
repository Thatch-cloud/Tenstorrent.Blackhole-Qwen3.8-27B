"""Add one admitted assembly comparison to an already prepared norm-prefetch recipe."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once
from frozen_draft_tail_gate import qualify_hardware


HELPERS = ('frozen_draft_tail.py', 'draft-tail-probe.py', 'frozen_draft_tail_gate.py',
    'frozen_draft_tail_hardware.py', 'frozen_draft_tail_scope.py', 'frozen_recipe_context.py')


def adapt(sources):
    result = dict(sources)
    result['dspark_8k_scope.py'] = replace_once(result['dspark_8k_scope.py'],
        'from frozen_gdn_norm_scope import runtime_scope as incremental_scope',
        'from frozen_draft_tail_scope import runtime_scope as incremental_scope')
    result['dspark_request_experiment.py'] = replace_once(result['dspark_request_experiment.py'],
        'Original versus prefetched norm bridge; incremental publication and shared Q/K in both arms',
        'Original versus tile-aligned proposal tail; norm prefetch, incremental publication and shared Q/K in both arms')
    for name, source in result.items():
        compile(source, name, 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    metadata = json.loads(options.geometry.read_text())
    if (metadata.get('norm_prefetch') is not True or metadata.get('combined_runtime') is not True
            or metadata.get('verifier_profile') is not False or metadata.get('geometry', {}).get('context') != 32768
            or options.manifest.exists()):
        raise ValueError('Fresh 32K norm-prefetch combined runtime required')
    scripts = options.checkout / 'scripts/ci'
    for name, digest in metadata['after'].items():
        if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared base runtime changed: ' + name)
    result = adapt({name: (scripts / name).read_text()
        for name in ('dspark_8k_scope.py', 'dspark_request_experiment.py')})
    result.update({name: Path(__file__).with_name(name).read_text() for name in HELPERS})
    for name, source in result.items():
        compile(source, name, 'exec')
        (scripts / name).write_bytes(source.encode())
    evidence = qualify_hardware(scripts, scripts / 'frozen-draft-tail-hardware.json')
    options.manifest.write_text(json.dumps(dict(admission=evidence,
        after={name: hashlib.sha256(source.encode()).hexdigest() for name, source in result.items()},
        performance_qualified=False, serving_defaults_changed=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
