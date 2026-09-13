"""Synthetic full-history ladder attention; no model weights or hardware speed claims."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspark_fp32_build
import dspark_stats_pack
from dspark_ladder_attention import adapter
from dspark_ladder_build import validate_manifest
from dspark_ladder_factory import selector_assert
from dspark_ladder_fixtures import fixture_probe
from dspark_ladder_geometry import CONTEXTS, geometry
from dspark_ladder_stage_print import stage_snapshots


def main():
    context_text = os.environ.get('QWEN_LADDER_CONTEXT')
    if context_text not in tuple(map(str, CONTEXTS)):
        raise ValueError('Explicit ladder context required')
    if (os.environ.get('QWEN_SIM_CASE') != 'dspark-ladder-attention'
            or os.environ.get('QWEN_DRAFT_FP32_INTERMEDIATES') != '1'):
        raise ValueError('Dedicated rebuilt ladder simulator required')
    context = int(context_text)
    fixture = geometry(context)

    def dumps(value, *arguments, **keywords):
        if isinstance(value, dict) and 'capacity' in value and 'numerical_tolerances' in value:
            value = dict(value, scope=__doc__, ladder_geometry=fixture,
                stage_instrumented=True, performance_qualified=False,
                sum_unpack_mode='fp32-direct-sum-a-b',
                output_recurrence='native-l1-pack-accumulation-bf16',
                key_chunk_size=fixture['key_chunk'],
                native_padded_keys=fixture['native_keys'], added_masked_poison_rows=fixture['extra_masked_keys'])
        return json.dumps(value, *arguments, **keywords)

    with fixture_probe(context) as probe:
        probe.execute = adapter(context)
        probe.SOURCES = tuple(sorted(set(probe.SOURCES + (
            'dspark-native-8k-attention-probe.py', 'dspark-ladder-attention-probe.py',
            'dspark_ladder_attention.py', 'dspark_ladder_geometry.py', 'dspark_ladder_fixtures.py',
            'dspark_ladder_factory.py', 'dspark_ladder_build.py', 'dspark_ladder_stage_print.py',
            'dspark_ladder_sum_unpack.py'))))
        probe.__file__ = str(Path(__file__).resolve())
        with stage_snapshots(), patch.object(dspark_stats_pack, 'SELECTOR_ASSERT', selector_assert()), \
                patch.object(dspark_fp32_build, 'validate_manifest', validate_manifest), \
                patch.object(probe, 'json', SimpleNamespace(dumps=dumps)):
            probe.main()


if __name__ == '__main__':
    main()
