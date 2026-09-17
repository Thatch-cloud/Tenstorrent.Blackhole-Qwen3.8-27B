"""Opt-in combined prefill/decode build using one content-addressed runtime."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import dspark_64k_build
from dspark_splitk_fp32_factory import SOURCE, transform
from dspark_splitk_hardware_gate import qualify


BUILDERS = ('dspark_splitk_combined_build.py', 'dspark_splitk_hardware_gate.py',
    'dspark_splitk_fp32_factory.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission(directory):
    directory = Path(directory)
    return qualify(directory, directory / 'dspark-splitk-hardware.json',
        directory / 'dspark-splitk-simulator.json')


def factory_identity(directory, evidence):
    return dict(source=SOURCE, source_before=evidence['factory_source_before'],
        source_after=evidence['factory_source_after'],
        hardware_report_sha256=evidence['report_sha256'],
        simulator_report_sha256=evidence['simulator_report_sha256'],
        builders={name: digest(Path(directory) / name) for name in BUILDERS})


def prepare(root, directory, report_path, *, prepare_prefill):
    evidence = admission(directory)
    source = Path(root) / SOURCE
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != evidence['factory_source_before']:
        raise ValueError('Pristine qualified split-K decode factory required')
    candidate = transform(original.decode()).encode()
    if hashlib.sha256(candidate).hexdigest() != evidence['factory_source_after']:
        raise ValueError('Exact hardware-qualified decode factory required')
    prefill = prepare_prefill(root, directory, report_path)
    if prefill.get('capacity') != 66560 or 'splitk_factory' in prefill:
        raise ValueError('Fresh full-history prefill factory identity required')
    source.write_bytes(candidate)
    return dict(prefill, splitk_factory=factory_identity(directory, evidence))


def validate_combined(root, directory, report_path):
    evidence = admission(directory)
    build = dspark_64k_build.validate_build(root, report_path, directory)
    expected = factory_identity(directory, evidence)
    if (build['factory_inputs'].get('splitk_factory') != expected
            or digest(Path(root) / SOURCE) != expected['source_after']):
        raise ValueError('Both attention factories must belong to the same combined build')
    return dict(component=evidence, build=build,
        full_request_qualified=False, performance_qualified=False, serving_qualified=False)


def require_selected():
    required = ('QWEN_SPLITK_COMBINED', 'QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED',
        'QWEN_DSPARK_64K_TRIAL', 'QWEN_DSPARK_SUM_SFPU')
    if (any(os.environ.get(name) != '1' for name in required)
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_HOME') != '/opt/tt-metal'):
        raise ValueError('Explicit allocated combined split-K hardware experiment required')


@contextmanager
def build_scope():
    require_selected()
    original = dspark_64k_build.prepare

    def combined(root, directory, report_path):
        return prepare(root, directory, report_path, prepare_prefill=original)

    with patch.object(dspark_64k_build, 'prepare', combined):
        yield


if __name__ == '__main__':
    from dspark_runtime_cache import main
    from dspark_sum_sfpu_hardware import hardware_scope

    directory = Path(__file__).parent
    with hardware_scope(directory), build_scope():
        main()
        validate_combined('/opt/tt-metal', directory, '/experiment/results/dspark-64k-hardware-build.json')
