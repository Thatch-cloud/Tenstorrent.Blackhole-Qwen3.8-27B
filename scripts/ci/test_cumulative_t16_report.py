import copy
import unittest
from unittest.mock import patch

import cumulative_t16_report as validator


class CumulativeReportTests(unittest.TestCase):
    def test_declared_down_grid_requires_route_and_all_layers(self):
        report = dict(cumulative_t16=True,
            cumulative_components=['direct_windows', 'compact_scores', 'wider_mlp_down'],
            cumulative_sources={'source': 'hash'}, cumulative_sources_after={'source': 'hash'},
            cumulative_measurement_quality={}, cumulative_route_diagnostics=[dict(
                direct=dict(hits=96, restored=True), compact=dict(calls=2, steps=30, restored=True),
                down=dict(hits=[2] * 64, restored=True)) for unused in range(3)],
            request_checks=[dict(gdn_direct_window=dict(direct=enabled), compact_score=dict(compact=enabled),
                mlp_down_grid=dict(wider_down=enabled), gdn_shared_qk=dict(loads=[{}] * 96),
                score_layout=dict(calls=2), fused_t16_mlp=dict(hits=[2] * 64))
                for enabled in (False, True, False, True, True, False)])
        with patch.object(validator, 'validate_direct', return_value=dict(measurement_quality={})), \
                patch.object(validator, 'validate_compact'), patch.object(validator, 'validate_down') as down:
            validator.validate(report)
            self.assertEqual(down.call_count, 6)
            mutations = (
                lambda value: value['request_checks'][1]['mlp_down_grid'].update(wider_down=False),
                lambda value: value['cumulative_route_diagnostics'][0].pop('down'),
                lambda value: value['cumulative_route_diagnostics'][0]['down'].update(hits=[0] * 64),
                lambda value: value['cumulative_components'].pop(),
            )
            for mutate in mutations:
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validator.validate(changed)

    def test_both_routes_and_stable_sources_required(self):
        enabled = [False, True, False, True, True, False]
        report = dict(cumulative_t16=True, cumulative_components=['direct_windows', 'compact_scores'],
            cumulative_sources={'source': 'hash'}, cumulative_sources_after={'source': 'hash'},
            cumulative_measurement_quality={'repeatability_passed': True},
            cumulative_route_diagnostics=[dict(direct=dict(hits=96, restored=True),
                compact=dict(calls=2, steps=30, restored=True)) for unused in range(3)],
            request_checks=[dict(gdn_direct_window=dict(direct=value), compact_score=dict(compact=value),
                gdn_shared_qk=dict(loads=[{}] * 96), score_layout=dict(calls=2)) for value in enabled])
        with patch.object(validator, 'validate_direct', return_value=dict(
                measurement_quality=report['cumulative_measurement_quality'])), \
                patch.object(validator, 'validate_compact') as compact:
            validator.validate(report)
            self.assertEqual(compact.call_count, 6)
            mutations = (
                lambda value: value['request_checks'][1]['compact_score'].update(compact=False),
                lambda value: value['cumulative_route_diagnostics'][0]['compact'].update(calls=0),
                lambda value: value['cumulative_sources_after'].update(source='changed'),
                lambda value: value['cumulative_route_diagnostics'].pop(),
                lambda value: value.update(cumulative_measurement_quality={}),
            )
            for mutate in mutations:
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validator.validate(changed)
