"""Synthetic full-history ladder attention; no model weights or hardware speed claims."""

import json
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspark_fp32_build
import dspark_stats_pack
import dspark_attention_value_diagnostics
from dspark_ladder_attention import adapter
from dspark_ladder_build import validate_manifest
from dspark_ladder_factory import selector_assert
from dspark_ladder_fixtures import fixture_probe
from dspark_ladder_geometry import CONTEXTS, geometry
from dspark_ladder_scalar_reciprocal import scalar_reciprocal
from dspark_ladder_stage_print import stage_snapshots


def main():
    diagnostics = os.environ.get('QWEN_LADDER_VALUE_DIAGNOSTICS', '0')
    if diagnostics not in ('0', '1'):
        raise ValueError('Value diagnostics must be explicitly 0 or 1')
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
                stage_instrumented=context == 65536, performance_qualified=False,
                stage_coordinates=dict(row=5, column=0) if context == 65536 else None,
                value_diagnostics_enabled=diagnostics == '1',
                sum_unpack_mode='native-tf32',
                reciprocal_mode='tr0-scalar-fp32-diagnostic',
                reciprocal_reload_rounding='native-truncate',
                output_recurrence='native-l1-pack-accumulation-bf16',
                key_chunk_size=fixture['key_chunk'],
                native_padded_keys=fixture['native_keys'], added_masked_poison_rows=fixture['extra_masked_keys'])
        return json.dumps(value, *arguments, **keywords)

    with fixture_probe(context) as probe:
        probe.execute = adapter(context)
        probe.SOURCES = tuple(sorted(set(probe.SOURCES + (
            'dspark-native-8k-attention-probe.py', 'dspark-ladder-attention-probe.py',
            'dspark_ladder_attention.py', 'dspark_ladder_geometry.py', 'dspark_ladder_fixtures.py',
            'dspark_ladder_factory.py', 'dspark_ladder_build.py', 'dspark_ladder_scalar_reciprocal.py',
            'dspark_ladder_stage_print.py'))))
        probe.__file__ = str(Path(__file__).resolve())
        with scalar_reciprocal(), patch.object(dspark_stats_pack, 'SELECTOR_ASSERT', selector_assert()), \
                (stage_snapshots(row=5, column=0) if context == 65536 else nullcontext()), \
                patch.object(dspark_attention_value_diagnostics, 'KINDS',
                    dspark_attention_value_diagnostics.KINDS if diagnostics == '1' else ()), \
                patch.object(dspark_fp32_build, 'validate_manifest', validate_manifest), \
                patch.object(probe, 'json', SimpleNamespace(dumps=dumps)):
            probe.main()


if __name__ == '__main__':
    main()
