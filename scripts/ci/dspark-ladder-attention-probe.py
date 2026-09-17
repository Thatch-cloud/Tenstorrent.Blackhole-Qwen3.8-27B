"""Synthetic full-history ladder attention; no model weights or hardware speed claims."""

import json
from contextlib import nullcontext
import os
import sys
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
from dspark_ladder_sum_update import scalar_sum_update
from dspark_ladder_score_center import scalar_score_center
from dspark_ladder_normalization import scratch_normalization


def main():
    diagnostics = os.environ.get('QWEN_LADDER_VALUE_DIAGNOSTICS', '0')
    if diagnostics not in ('0', '1'):
        raise ValueError('Value diagnostics must be explicitly 0 or 1')
    context_text = os.environ.get('QWEN_LADDER_CONTEXT')
    if context_text not in tuple(map(str, CONTEXTS)):
        raise ValueError('Explicit ladder context required')
    if '--hardware' in sys.argv:
        from dspark_ladder_backend import require_backend
        require_backend(os.environ, hardware=True, device_present=Path('/dev/tenstorrent').exists())
    elif os.environ.get('QWEN_SIM_CASE') != 'dspark-ladder-attention':
        raise ValueError('Dedicated ladder simulator required')
    if os.environ.get('QWEN_DRAFT_FP32_INTERMEDIATES') != '1':
        raise ValueError('Rebuilt ladder statistics factory required')
    context = int(context_text)
    diagnostic_row, diagnostic_column = (3, 0) if '--hardware' in sys.argv else (8, 116)
    fixture = geometry(context)
    smoke = os.environ.get('QWEN_LADDER_SCORE_SMOKE', '0')
    if smoke not in ('0', '1') or (smoke == '1' and context != 128):
        raise ValueError('Score smoke requires the 128-token fixture')

    def dumps(value, *arguments, **keywords):
        if isinstance(value, dict) and 'stage' in value:
            value = dict(value, context=context, score_center_smoke=smoke == '1')
        if isinstance(value, dict) and 'capacity' in value and 'numerical_tolerances' in value:
            value = dict(value, scope=__doc__, ladder_geometry=fixture,
                score_center_smoke=smoke == '1',
                stage_instrumented=context == 65536, performance_qualified=False,
                stage_coordinates=dict(row=diagnostic_row, column=diagnostic_column) if context == 65536 else None,
                value_diagnostics_enabled=diagnostics == '1',
                sum_unpack_mode='native-tf32',
                sum_update_mode='tr0-scalar-fp32' if context == 65536 else 'native',
                score_center_mode='tr0-fp32-before-reload' if context == 65536 or smoke == '1' else 'native',
                reciprocal_mode='tr0-scalar-fp32-diagnostic',
                reciprocal_reload_rounding='dedicated-fp32-scratch' if context == 65536 else 'native-truncate',
                normalization_mode='sfpu-column-scratch' if context == 65536 else 'native',
                output_recurrence='native-l1-pack-accumulation-bf16',
                final_output_rounding='unchanged',
                key_chunk_size=fixture['key_chunk'],
                native_padded_keys=fixture['native_keys'], added_masked_poison_rows=fixture['extra_masked_keys'])
        return json.dumps(value, *arguments, **keywords)

    with fixture_probe(context) as probe:
        probe.execute = adapter(context)
        probe.SOURCES = tuple(sorted(set(probe.SOURCES + (
            'dspark-native-8k-attention-probe.py', 'dspark-ladder-attention-probe.py',
            'dspark_ladder_attention.py', 'dspark_ladder_geometry.py', 'dspark_ladder_fixtures.py',
            'dspark_ladder_factory.py', 'dspark_ladder_build.py', 'dspark_ladder_scalar_reciprocal.py',
            'dspark_ladder_stage_print.py', 'dspark_ladder_sum_update.py', 'dspark_ladder_score_center.py',
            'dspark_ladder_output_rounding.py', 'dspark_ladder_normalization.py'))))
        probe.__file__ = str(Path(__file__).resolve())
        with scalar_reciprocal(), patch.object(dspark_stats_pack, 'SELECTOR_ASSERT', selector_assert()), \
                scalar_sum_update(), \
                (stage_snapshots(row=diagnostic_row, column=diagnostic_column) if context == 65536 else nullcontext()), \
                scalar_score_center(key_tiles=40 if smoke == '1' else 2112), \
                (scratch_normalization() if context == 65536 else nullcontext()), \
                patch.object(dspark_attention_value_diagnostics, 'KINDS',
                    dspark_attention_value_diagnostics.KINDS if diagnostics == '1' else ()), \
                patch.object(dspark_fp32_build, 'validate_manifest', validate_manifest), \
                patch.object(probe, 'json', SimpleNamespace(dumps=dumps)):
            probe.main()


if __name__ == '__main__':
    main()
