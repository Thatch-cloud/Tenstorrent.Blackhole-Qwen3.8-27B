"""Explicit experiment-only projection links; never infer hardware from chip count."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from sampling_link_policy import DESCRIPTOR, SOURCES


def validate(environment):
    requested = environment.get('QWEN_PROJECTION_LINKS', '1')
    if requested not in ('1', '2', '4'):
        raise ValueError('Projection links must be explicitly 1, 2 or 4')
    simulated = bool(environment.get('TT_METAL_SIMULATOR'))
    if simulated:
        if requested != '1':
            raise ValueError('Simulator cannot qualify physical multi-link operation')
    elif requested != '1':
        root = Path(environment.get('TT_METAL_HOME', ''))
        if (environment.get('QWEN_HARDWARE_TESTS') != '1'
                or environment.get('QWEN_CARDS_ALLOCATED') != '1'
                or environment.get('TT_METAL_MOCK_CLUSTER_DESC_PATH')
                or environment.get('TT_METAL_SLOW_DISPATCH_MODE')
                or environment.get('TT_MESH_GRAPH_DESC_PATH') != str(root / DESCRIPTOR)):
            raise ValueError('Allocated P150A-pair hardware and audited descriptor required')
        if hashlib.sha256((root / DESCRIPTOR).read_bytes()).hexdigest() != SOURCES[DESCRIPTOR]:
            raise ValueError('P150A-pair descriptor differs from audited four-link configuration')
    backend = 'simulator' if simulated else 'hardware' if environment.get('QWEN_HARDWARE_TESTS') == '1' else 'unallocated'
    return dict(requested_links=int(requested), backend=backend,
                selection='explicit; not native discovery or physical-link validation')


@lru_cache(maxsize=8)
def _resolve(values):
    report = validate(dict(values))
    print(json.dumps(dict(stage='projection_link_policy', **report)), flush=True)
    return report['requested_links']


def projection_links():
    keys = ('QWEN_PROJECTION_LINKS', 'TT_METAL_SIMULATOR', 'TT_METAL_HOME',
            'QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED', 'TT_METAL_MOCK_CLUSTER_DESC_PATH',
            'TT_METAL_SLOW_DISPATCH_MODE', 'TT_MESH_GRAPH_DESC_PATH')
    return _resolve(tuple((key, os.environ[key]) for key in keys if key in os.environ))
