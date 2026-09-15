"""Opt-in 64K combined build for validated maxima components, not timing admission."""

from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch

import dspark_splitk_combined_build as baseline
from dspark_splitk_maxima_factory import transform
from dspark_splitk_maxima_gate import qualify as qualify_simulator
from matched_draft_gate import qualify as qualify_draft
from matched_target_gate import qualify as qualify_target


BUILDERS = (*baseline.BUILDERS, 'matched_combined_build.py', 'matched_draft_gate.py',
    'matched_target_gate.py', 'dspark_splitk_maxima_factory.py', 'dspark_splitk_maxima_gate.py')


def admission(directory):
    directory = Path(directory)
    simulator = qualify_simulator(directory, directory / 'dspark-maxima-simulator.json')
    draft = qualify_draft(directory, directory / 'matched-context-attention-65536.json', 65536)
    target = qualify_target(directory, directory / 'matched-context-target-65536.json', 65536)
    if simulator['factory_source_after'] != draft['factory_sha256']:
        raise ValueError('Simulator and physical draft factory must match exactly')
    return dict(run_id=35035520826, report_sha256=draft['report_sha256'],
        simulator_report_sha256=simulator['report_sha256'],
        factory_source_before=simulator['factory_source_before'],
        factory_source_after=draft['factory_sha256'], kernel=draft['kernel'],
        draft=draft, target=target, component_qualified=True,
        full_request_qualified=False, performance_qualified=False, serving_qualified=False)


@contextmanager
def identity_scope():
    if os.environ.get('QWEN_MATCHED_COMBINED') != '1':
        raise ValueError('Explicit fresh matched combined experiment required')
    original = baseline.transform
    with patch.object(baseline, 'admission', admission), patch.object(baseline, 'BUILDERS', BUILDERS), \
            patch.object(baseline, 'transform', lambda source: transform(original(source))):
        yield


if __name__ == '__main__':
    from dspark_runtime_cache import main
    from dspark_sum_sfpu_hardware import hardware_scope

    directory = Path(__file__).parent
    with identity_scope(), hardware_scope(directory), baseline.build_scope():
        main()
        baseline.validate_combined('/opt/tt-metal', directory, '/experiment/results/dspark-64k-hardware-build.json')
