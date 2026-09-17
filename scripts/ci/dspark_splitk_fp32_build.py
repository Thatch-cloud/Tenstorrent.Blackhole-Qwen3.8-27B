"""Source-keyed simulator build for opt-in FP32 decode intermediates."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import dspark_sim_build_cache
from dspark_hardware_gate import digest
from dspark_score_sfpu_build import main as build
from dspark_splitk_fp32_factory import SOURCE, transform
from dspark_sum_sfpu import sum_scope


BUILDERS = ('dspark_splitk_fp32_build.py', 'dspark_splitk_fp32_factory.py')
REPORT = '/experiment/results/dspark-splitk-fp32-build.json'


def validate(root, report_path=REPORT):
    root = Path(root)
    report = json.loads(Path(report_path).read_text())
    scripts = Path(__file__).parent
    if report.get('passed') is not True or report.get('source_after') != digest(root / SOURCE):
        raise ValueError('Completed exact split-K factory rebuild required')
    if report.get('builders') != {name: digest(scripts / name) for name in BUILDERS}:
        raise ValueError('Split-K factory builders changed')
    binaries = report.get('binaries', {})
    expected = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
    if set(binaries) != set(expected) or len(set(binaries.values())) != 1:
        raise ValueError('Both rebuilt binary paths required')
    if any(digest(root / name) != binaries[name] for name in expected):
        raise ValueError('Split-K loaded binary changed')
    return report


def main():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_SPLITK_ATTENTION') != '1' or Path('/dev/tenstorrent').exists()):
        raise ValueError('Explicit device-free split-K build required')
    root = Path('/opt/tt-metal')
    source = root / SOURCE
    scripts = Path(__file__).parent
    before = digest(source)
    source.write_text(transform(source.read_text()))
    report = dict(passed=False, source_before=before, source_after=digest(source),
        builders={name: digest(scripts / name) for name in BUILDERS}, device_access=False)
    original_inputs = dspark_sim_build_cache.build_inputs

    def inputs(runtime, directory):
        result = original_inputs(runtime, directory)
        result['splitk_factory'] = dict(report)
        return result

    try:
        with sum_scope(), patch.object(dspark_sim_build_cache, 'build_inputs', inputs):
            build()
        report['binaries'] = {name: digest(root / name) for name in
            ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')}
        report['passed'] = True
    finally:
        Path(REPORT).write_text(json.dumps(report, indent=2) + '\n')
    validate(root)


if __name__ == '__main__':
    main()
