from frozen_context_geometry import selected_geometry
"""Offline hardware entry routing; ordinary 4K and preflight entry remain unchanged."""

import json
import os
from pathlib import Path
import sys

from dspark_context_selection import request_context


def run(main):
    if request_context() != selected_geometry()['context'] or '--preflight' in sys.argv:
        return main()
    if (os.environ.get('QWEN_HARDWARE_TESTS') != '1'
            or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('TT_METAL_SIMULATOR') or '--captured-publication' not in sys.argv):
        raise ValueError('8K integration requires the allocated offline captured-publication experiment')
    if os.environ.get('QWEN_FROZEN_COMBINED_RUNTIME') != '1' or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1':
        raise ValueError('Explicit combined candidate and compact scratch required')
    from dspark_8k_scope import runtime_scope
    evidence = json.loads(Path('/experiment/results/dspark-8k-hardware-build.json').read_text())
    with runtime_scope(Path(__file__).parent, context=selected_geometry()['context'], output_tokens=256,
            factory_root=os.environ['TT_METAL_HOME'], build_evidence=evidence):
        return main()
