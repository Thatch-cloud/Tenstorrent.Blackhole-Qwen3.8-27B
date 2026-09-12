"""Retained proposal evidence for an audited hardware experiment, not serving admission."""

import ast
import hashlib
import json
from pathlib import Path


REPORT_SHA256 = 'e90aa5715fe105a3722f75b17d933db8be89e79de6c79c9f994d761671566559'


def source_closure(directory):
    directory = Path(directory)
    pending = ['dspark_t32_prepared.py']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(ast.parse((directory / name).read_text())):
            modules = [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                candidate = (module or '').split('.')[0] + '.py'
                if (directory / candidate).is_file():
                    pending.append(candidate)
    return visited


def qualify(path, directory):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Retained combined proposal simulator report required')
    report = json.loads(payload)
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('context') != 4096 or report.get('capacity') != 4384
            or report.get('proposals') != 31 or report.get('learned_layers') != 5
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete stable 31-query learned proposal evidence required')
    checks = report.get('checks', [])
    if ([check.get('anchor') for check in checks] != [20, 10]
            or any(check.get('exact') is not True or len(check.get('tokens', [])) != 31 for check in checks)
            or checks[0]['tokens'] == checks[1]['tokens']):
        raise ValueError('Two distinct changing-input exact proposal replays required')
    sources = source_closure(directory)
    changed = sorted(name for name in sources
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != report['sources'].get(name))
    if changed:
        raise ValueError('Proposal dependency changed: ' + ', '.join(changed))
    return dict(run=34660555430, report_sha256=REPORT_SHA256, source_count=len(sources),
        sources={name: report['sources'][name] for name in sorted(sources)},
        scope='Learned proposal replay with synthetic history; permits auditing integration, not performance promotion',
        full_request_qualified=False, serving_qualified=False)
