"""Disposable simulator factory build for the finite context ladder."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from unittest.mock import patch

import dspark_fp32_build as baseline
from dspark_hardware_gate import digest
from dspark_ladder_factory import geometry_predicate
from dspark_ladder_sum_unpack import INSERT, remove_unpack_transform, transform


BUILDERS = ('dspark_ladder_build.py', 'dspark_ladder_factory.py', 'dspark_ladder_geometry.py',
    'dspark_ladder_sum_unpack.py')
BASELINE_VALIDATE = baseline.validate_manifest


@contextmanager
def factory_scope():
    replacement = baseline.REPLACEMENT.replace('Skt == 272 && Sq_chunk_t == 1 && Sk_chunk_t == 8',
        geometry_predicate('Skt', 'Sk_chunk_t') + ' && Sq_chunk_t == 1')
    if replacement == baseline.REPLACEMENT:
        raise ValueError('Original baseline geometry selector required')

    def selected(source, *, enabled=True):
        if enabled is not True:
            raise ValueError('Ladder requires the enabled FP32 statistics factory')
        if INSERT.encode() in source:
            source = remove_unpack_transform(source)
        return transform(source)

    with patch.object(baseline, 'REPLACEMENT', replacement), patch.object(baseline, 'transform', selected):
        yield


def validate_manifest(root, output):
    with factory_scope():
        report = BASELINE_VALIDATE(root, output)
    if report.get('ladder_builders') != {name: digest(Path(__file__).with_name(name)) for name in BUILDERS}:
        raise ValueError('Exact ladder builder provenance required')
    if report.get('experiment') != 'full-context-ladder':
        raise ValueError('Explicit ladder build required')
    if report.get('sum_unpack_mode') != 'fp32-direct-sum-a-b':
        raise ValueError('Explicit sum-buffer unpack candidate required')
    return report


def main():
    if os.environ.get('QWEN_SIM_CASE') != 'dspark-ladder-attention':
        raise ValueError('Dedicated ladder simulator case required')
    if os.environ.get('QWEN_DRAFT_FP32_CONTROL', '0') != '0':
        raise ValueError('Ladder precision control override is unsupported')
    output = Path('/experiment/results/dspark-fp32-build.json')
    builders = {name: digest(Path(__file__).with_name(name)) for name in BUILDERS}
    with factory_scope(), patch.dict(os.environ, {'QWEN_SIM_CASE': 'dspark-native-8k-attention'}):
        baseline.main()
    report = json.loads(output.read_text())
    report['experiment'] = 'full-context-ladder'
    report['ladder_builders'] = builders
    report['sum_unpack_mode'] = 'fp32-direct-sum-a-b'
    output.write_text(json.dumps(report, indent=2) + '\n')
    validate_manifest('/opt/tt-metal', output)


if __name__ == '__main__':
    main()
