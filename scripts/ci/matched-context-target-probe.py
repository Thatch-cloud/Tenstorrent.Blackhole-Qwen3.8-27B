"""Full-context folded-verifier comparison to native B1; no model/TG acceptance."""

import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from dspark_hardware_gate import digest
from dspark_splitk_hardware_build import REPORT, validate_build
from dspark_splitk_maxima_hardware import hardware_identity
from matched_target_geometry import geometry, geometry_scope


def main():
    if '--hardware' not in sys.argv or not Path('/dev/tenstorrent').exists():
        raise ValueError('Explicit physical target-context experiment required')
    if os.environ.get('QWEN_SPLITK_FP32_INTERMEDIATES', '0') != '0':
        raise ValueError('Draft-only factory selector must not affect native target attention')
    context = int(os.environ['QWEN_MATCHED_CONTEXT'])
    plan = geometry(context)
    directory = Path(__file__).parent
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise ValueError('Fresh target-context report required')
    names = (Path(__file__).name, 'matched_target_geometry.py', 'matched_context_geometry.py',
        'dspark_splitk_maxima_hardware.py', 'dspark_splitk_maxima_gate.py')
    sources = {name: digest(directory / name) for name in names}
    with hardware_identity():
        build = validate_build(os.environ['TT_METAL_HOME'], directory, REPORT,
            directory / 'dspark-splitk-simulator.json')
        spec = importlib.util.spec_from_file_location('target_context_fixture',
            directory / 'target-t16-attention-64k-probe.py')
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)

        def selected_geometry(*, hardware):
            if hardware is not True:
                raise ValueError('Hardware context selection required')
            return plan

        try:
            with geometry_scope(context), patch.object(fixture, 'geometry', selected_geometry):
                fixture.main()
        finally:
            if output.exists():
                report = json.loads(output.read_text())
                after = {name: digest(directory / name) for name in names}
                report.update(scope=__doc__, geometry=plan, context_sources=sources,
                    context_sources_after=after, build=build, full_request_qualified=False,
                    performance_qualified=False)
                report['passed'] = report.get('passed') is True and sources == after
                output.write_text(json.dumps(report, indent=2) + '\n')
                if sources != after:
                    raise ValueError('Target context adapter changed during execution')


if __name__ == '__main__':
    main()
