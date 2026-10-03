from copy import deepcopy
import unittest
from unittest.mock import patch

from gdn_output_grid_gate import REPORT_SHA256
from gdn_output_grid_variants import POLICIES, validate_route


class OutputVariantsTests(unittest.TestCase):
    def test_only_output_placement_differs(self):
        candidate = dict(POLICIES['publication'])
        self.assertTrue(candidate.pop('gdn_output_grid'))
        self.assertEqual(candidate, POLICIES['control'])
        self.assertTrue(candidate['captured_publication'])

    @patch('gdn_output_grid_variants.publication_route')
    def test_each_layer_and_restoration_required(self, publication):
        record = dict(gdn_output_grid=dict(hits=[2] * 48, restored=True,
            admission=dict(report_sha256=REPORT_SHA256)))
        validate_route(record, 'publication')
        publication.assert_called_with(record, 'publication')
        for defect in ('hit', 'restored', 'hash'):
            changed = deepcopy(record)
            if defect == 'hit':
                changed['gdn_output_grid']['hits'][47] = 0
            elif defect == 'restored':
                changed['gdn_output_grid']['restored'] = False
            else:
                changed['gdn_output_grid']['admission']['report_sha256'] = 'different'
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_route(changed, 'publication')
        with self.assertRaises(ValueError):
            validate_route(record, 'control')
        validate_route({}, 'control')
