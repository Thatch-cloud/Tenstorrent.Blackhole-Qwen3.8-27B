"""Stage a source-bound concat candidate after full-window ladder preparation."""

import argparse
import hashlib
import json
from pathlib import Path

from frozen_recipe_context import replace_once
from history_concat_gate import qualify


def stage(checkout, evidence, checksum, manifest):
    scripts = Path(checkout) / 'scripts/ci'
    evidence, manifest = Path(evidence), Path(manifest)
    if manifest.exists() or (scripts / 'history-concat-evidence').exists():
        raise ValueError('Fresh history concat staging required')
    report = qualify(Path(__file__).parent, evidence, checksum)
    for name in ('dspark_history.py', 'gdn_multitoken_conv.py'):
        if hashlib.sha256((scripts / name).read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Staged history source differs: ' + name)
    scope = (scripts / 'frozen_ladder_cache_scope.py').read_text()
    scope = replace_once(scope,
        '    with tail_scope() as tail, page_geometry(context) as evidence, audit_scope() as reference_audit:',
        '    from history_concat_scope import runtime_scope as concat_scope\n'
        '    with concat_scope(directory) as concat, tail_scope() as tail, page_geometry(context) as evidence, audit_scope() as reference_audit:\n'
        "        evidence['history_concat_lifetime'] = concat")
    compile(scope, 'frozen_ladder_cache_scope.py', 'exec')
    payloads = {name: Path(__file__).with_name(name).read_bytes() for name in
        ('history_concat_gate.py', 'history_concat_scope.py', 'history_concat_lifetime.py', 'history-concat-probe.py')}
    payloads['frozen_ladder_cache_scope.py'] = scope.encode()
    for name in ('history-concat.json', 'history-concat.exit-status', 'simulator-runtime.txt'):
        payloads['history-concat-evidence/' + name] = (evidence / name).read_bytes()
    payloads['history-concat-evidence/admission.json'] = json.dumps(dict(report_sha256=checksum)).encode()
    for name, payload in payloads.items():
        destination = scripts / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    manifest.write_text(json.dumps(dict(report_sha256=checksum,
        sources={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        performance_qualified=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.evidence, options.sha256, options.manifest)
