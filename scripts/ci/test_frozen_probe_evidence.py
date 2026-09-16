from copy import deepcopy
import unittest

from frozen_context_geometry import geometry
from frozen_probe_evidence import validate_pair, validate_diagnostics


def fixture(context):
    shape = geometry(context)
    base = dict(closed_cleanly=True, backend='simulator', capacity=shape['capacity'],
        positions=list(shape['positions']), proposal_rows=15, key_chunk_size=256,
        native_padded_keys=shape['padded_keys'], numerical_tolerances=dict(rtol=.01, atol=.01),
        sources={'fixture': 'source'}, sources_after={'fixture': 'source'},
        native_sources={'fixture': 'native'}, native_sources_after={'fixture': 'native'},
        factory_build=dict(passed=True, import_passed=True, factory_enabled=True, precision_variant='stats-only'))
    numerical = dict(deepcopy(base), passed=True, probe_part='numerical', input_checks=[], layout_checks=[])
    for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
        numerical[mode + '_checks'] = []
        for ordinal, case in enumerate(cases):
            for chip in range(2):
                coordinate = dict(ordinal=ordinal, case=case, chip=chip)
                numerical[mode + '_checks'].append(dict(coordinate, passed=True,
                    failed_elements=0, numerical_close=True, replay_exact=True))
                for name in ('query', 'history_key', 'history_value', 'query_key', 'query_value', 'mask'):
                    numerical['input_checks'].append(dict(coordinate, mode=mode, name=name, exact=True))
                for name in ('key', 'value'):
                    numerical['layout_checks'].append(dict(coordinate, mode=mode, name=name,
                        passed=True, sha256='fixture', expected_sha256='fixture'))
    numerical['fixture_controls'] = [dict(name=name, case=int(name == 'frontier_update'), chip=chip,
        detected=True) for name in ('oldest', 'last_proposal', 'gap_poison', 'frontier_update') for chip in range(2)]
    numerical['stale_controls'] = [dict(chip=chip, detected=True) for chip in range(2)]
    diagnostics = dict(deepcopy(base), probe_part='diagnostics', diagnostics_complete=True,
        value_diagnostics=[dict(kind=kind, chip=chip, finite=True, failed_elements=0)
            for kind in ('constant', 'oldest', 'last_proposal') for chip in range(2)])
    return numerical, diagnostics


class EvidenceTests(unittest.TestCase):
    def test_diagnostics_reject_mismatch_nonfinite_or_incomplete_output(self):
        records = fixture(32768)[1]['value_diagnostics']
        validate_diagnostics(records)
        for field, value in (('failed_elements', 1), ('finite', False), ('chip', 3)):
            changed = deepcopy(records)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                validate_diagnostics(changed)
        with self.assertRaises(ValueError):
            validate_diagnostics(records[:-1])

    def test_complete_pair_is_not_model_or_provenance_qualification(self):
        result = validate_pair(*fixture(32768), 32768)
        self.assertTrue(result['complete_probe_coverage'])
        self.assertFalse(result['performance_qualified'])
        self.assertFalse(result['source_files_verified'])

    def test_missing_mixed_or_failed_evidence_rejected(self):
        for mutate in (
                lambda numerical, diagnostic: numerical['replay_checks'].pop(),
                lambda numerical, diagnostic: diagnostic.update(capacity=8448),
                lambda numerical, diagnostic: diagnostic.update(closed_cleanly=False),
                lambda numerical, diagnostic: diagnostic.update(reciprocal_variant='scalar-fp32'),
                lambda numerical, diagnostic: diagnostic['sources'].update(fixture='other'),
                lambda numerical, diagnostic: diagnostic['value_diagnostics'][0].update(failed_elements=1),
                lambda numerical, diagnostic: numerical['layout_checks'][0].update(expected_sha256='other')):
            numerical, diagnostic = fixture(32768)
            mutate(numerical, diagnostic)
            with self.assertRaises(ValueError):
                validate_pair(numerical, diagnostic, 32768)


if __name__ == '__main__':
    unittest.main()
